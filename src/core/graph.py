import yaml
import time
from functools import lru_cache
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.core.guardrails import check_input_safety, check_output_safety
import json

load_dotenv()

with open("catalog.json", "r") as f:
    catalog_data = json.load(f)

# Load configurations
with open("config/retrieval.yaml", "r") as f:
    config = yaml.safe_load(f)
    
with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

@lru_cache(maxsize=1)
def get_retriever():
    """Initializes retrieval resources only when the first RAG query needs them."""
    from src.core.retriever import HybridRetriever

    return HybridRetriever()

@lru_cache(maxsize=1)
def get_local_llm():
    """Keeps the MLX model resident after the first local generation call."""
    from mlx_lm import load

    return load(config["system"]["local_llm"])

@lru_cache(maxsize=None)
def get_openai_llm(model_name: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=0)

class RouteExpansionResult(BaseModel):
    route: str = Field(description="One of finance, relationships, philosophy, out_of_scope, greeting.")
    expanded_queries: list[str] = Field(description="Two to three concise retrieval queries, including the original user intent.")

def local_generate(prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate

    model, tokenizer = get_local_llm()
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start:end + 1])

def _merge_timings(state: "AgentState", updates: dict[str, float]) -> dict[str, float]:
    return {**state.get("timings", {}), **updates}

def _timed_update(state: "AgentState", node_name: str, started_at: float, update: dict[str, Any]) -> dict[str, Any]:
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    update["timings"] = _merge_timings(state, {node_name: elapsed_ms})
    return update

def _dedupe_queries(query: str, expanded_queries: list[str]) -> list[str]:
    max_queries = int(config.get("retrieval", {}).get("max_expanded_queries", 2))
    queries = [query, *expanded_queries]
    deduped = []
    seen = set()
    for item in queries:
        cleaned = " ".join(str(item).split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            deduped.append(cleaned)
            seen.add(key)
        if len(deduped) >= max_queries:
            break
    return deduped or [query]

def build_synthesis_prompt(state: "AgentState") -> str | None:
    if not state.get("retrieved_chunks"):
        return None

    context_parts = []
    for c in state["retrieved_chunks"]:
        title = catalog_data.get(c["book_id"], {}).get("title", c["book_id"])
        context_parts.append(f"[{title}] {c['text']}")

    context = "\n\n".join(context_parts)
    return prompts["synthesis"].format(context=context, query=state["query"])

def stream_synthesis_answer(state: "AgentState"):
    """Yields answer text incrementally where the active model supports streaming."""
    formatted_prompt = build_synthesis_prompt(state)
    if formatted_prompt is None:
        yield "I don't have enough information in my library to answer this."
        return

    if config["system"]["inference_mode"] == "openai":
        llm = get_openai_llm(config["openai"]["synthesis_model"])
        for chunk in llm.stream(formatted_prompt):
            text = chunk.content
            if text:
                yield text
        return

    from mlx_lm import stream_generate

    model, tokenizer = get_local_llm()
    for chunk in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=1024):
        text = getattr(chunk, "text", "")
        if text:
            yield text

# 1. Define State
class AgentState(TypedDict):
    query: str
    expanded_queries: List[str]
    route: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    is_safe: bool
    guardrail_reason: str
    output_is_safe: bool
    output_guardrail_reason: str
    timings: Dict[str, float]

# 2. Node Functions
def input_guard_node(state: AgentState):
    """Validates the incoming user query."""
    started_at = time.perf_counter()
    safety_check = check_input_safety(state["query"])
    return _timed_update(state, "input_guard", started_at, {
        "is_safe": safety_check["is_safe"], 
        "guardrail_reason": safety_check["reason"]
    })

def router_node(state: AgentState):
    """Classifies the domain and generates semantic expansions in one model call."""
    started_at = time.perf_counter()
    query = state["query"]
    route_prompt = prompts["router_expansion"].format(query=query)
    
    if config["system"]["inference_mode"] == "openai":
        llm = get_openai_llm(config["openai"]["router_model"]).with_structured_output(
            RouteExpansionResult,
            method="json_schema",
            strict=True,
        )
        parsed = llm.invoke(route_prompt)
        route = parsed.route.strip().lower()
        expanded_raw = parsed.expanded_queries
    else:
        raw = local_generate(route_prompt, max_tokens=120).strip()
        try:
            parsed = _extract_json_object(raw)
            route = str(parsed.get("route", "")).strip().lower()
            expanded_raw = parsed.get("expanded_queries", [])
        except Exception:
            route = "in_scope"
            expanded_raw = [query]
        
    valid_routes = ["finance", "relationships", "philosophy", "out_of_scope", "greeting"]
    final_route = route if route in valid_routes else "in_scope"

    if final_route in {"greeting", "out_of_scope"}:
        return _timed_update(state, "router", started_at, {"route": final_route, "expanded_queries": [query]})

    if isinstance(expanded_raw, str):
        expanded_raw = expanded_raw.split("|")
    expanded = _dedupe_queries(query, expanded_raw)
        
    return _timed_update(state, "router", started_at, {"route": final_route, "expanded_queries": expanded})

def retriever_node(state: AgentState):
    """Executes the Hybrid Search & RRF."""
    started_at = time.perf_counter()
    retriever_engine = get_retriever()
    chunks = retriever_engine.retrieve(state["expanded_queries"])
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    timings = _merge_timings(state, {"retriever": elapsed_ms})
    timings.update(getattr(retriever_engine, "last_timings", {}))
    return {"retrieved_chunks": chunks, "timings": timings}

def synthesis_node(state: AgentState):
    """Drafts the final response using only the retrieved context."""
    started_at = time.perf_counter()
    formatted_prompt = build_synthesis_prompt(state)
    if formatted_prompt is None:
        return _timed_update(
            state,
            "synthesis",
            started_at,
            {"answer": "I don't have enough information in my library to answer this."},
        )
    
    if config["system"]["inference_mode"] == "openai":
        llm = get_openai_llm(config["openai"]["synthesis_model"])
        answer = llm.invoke(formatted_prompt).content
    else:
        # Local MLX Execution
        answer = local_generate(formatted_prompt, max_tokens=1024)
        
    return _timed_update(state, "synthesis", started_at, {"answer": answer})

def output_guard_node(state: AgentState):
    """Validates the generated answer before returning it."""
    started_at = time.perf_counter()
    safety_check = check_output_safety(state["answer"], state.get("retrieved_chunks", []))
    answer = state["answer"]
    if not safety_check["is_safe"]:
        answer = f"I cannot safely return that answer: {safety_check['reason']}"
    return _timed_update(state, "output_guard", started_at, {
        "answer": answer,
        "output_is_safe": safety_check["is_safe"],
        "output_guardrail_reason": safety_check["reason"],
    })

def greeting_node(state: AgentState):
    """Fast response for conversational greetings."""
    started_at = time.perf_counter()
    return _timed_update(state, "greeting", started_at, {"answer": "Hi! I am CiteMentor. I can offer advice based on my curated library of non-fiction books covering finance, philosophy, and relationships. How can I help you today?"})

def out_of_scope_node(state: AgentState):
    """Fast fallback for topics not covered in the library."""
    started_at = time.perf_counter()
    return _timed_update(state, "out_of_scope", started_at, {"answer": "This topic falls outside my curated non-fiction library. Please consult external resources."})

def unsafe_node(state: AgentState):
    """Fast fallback for blocked inputs."""
    started_at = time.perf_counter()
    return _timed_update(state, "unsafe", started_at, {"answer": f"Request blocked: {state['guardrail_reason']}"})

# 3. Conditional Routing Logic
def route_after_guard(state: AgentState):
    if not state.get("is_safe", True): return "unsafe"
    return "router"

def route_after_router(state: AgentState):
    if state["route"] == "greeting": return "greeting"
    if state["route"] == "out_of_scope": return "out_of_scope"
    return "retriever"

# 4. Build and Compile Graph
workflow = StateGraph(AgentState)

workflow.add_node("input_guard", input_guard_node)
workflow.add_node("router", router_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("output_guard", output_guard_node)
workflow.add_node("out_of_scope", out_of_scope_node)
workflow.add_node("unsafe", unsafe_node)

workflow.set_entry_point("input_guard")

workflow.add_conditional_edges("input_guard", route_after_guard, {
    "unsafe": "unsafe",
    "router": "router"
})

workflow.add_conditional_edges("router", route_after_router, {
    "greeting": "greeting",
    "out_of_scope": "out_of_scope",
    "retriever": "retriever"
})

workflow.add_edge("retriever", "synthesis")
workflow.add_edge("synthesis", "output_guard")
workflow.add_edge("output_guard", END)
workflow.add_edge("out_of_scope", END)
workflow.add_edge("unsafe", END)
workflow.add_edge("greeting", END)

app_graph = workflow.compile()
