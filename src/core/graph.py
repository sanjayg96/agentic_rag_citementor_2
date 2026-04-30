import yaml
from functools import lru_cache
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from src.core.guardrails import check_input_safety
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

def local_generate(prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate

    model, tokenizer = get_local_llm()
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

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

# 2. Node Functions
def input_guard_node(state: AgentState):
    """Validates the incoming user query."""
    safety_check = check_input_safety(state["query"])
    return {
        "is_safe": safety_check["is_safe"], 
        "guardrail_reason": safety_check["reason"]
    }

def router_node(state: AgentState):
    """Classifies the domain and generates semantic expansions."""
    query = state["query"]
    router_prompt = prompts["router"].format(query=query)
    
    if config["system"]["inference_mode"] == "openai":
        llm = get_openai_llm(config["openai"]["router_model"])
        route = llm.invoke(router_prompt).content.strip().lower()
    else:
        # Local MLX Execution
        route = local_generate(router_prompt, max_tokens=15).strip().lower()
        
    valid_routes = ["finance", "relationships", "philosophy", "out_of_scope", "greeting"]
    final_route = route if route in valid_routes else "in_scope"

    if final_route in {"greeting", "out_of_scope"}:
        return {"route": final_route, "expanded_queries": [query]}

    expansion_prompt = prompts["query_expansion"].format(query=query)
    if config["system"]["inference_mode"] == "openai":
        expansions_raw = get_openai_llm(config["openai"]["query_expansion_model"]).invoke(expansion_prompt).content.strip()
    else:
        expansions_raw = local_generate(expansion_prompt, max_tokens=60).strip()
    
    expanded = expansions_raw.split("|")
    expanded = [e.strip() for e in expanded if e.strip()]
    if not expanded:
        expanded = [query]
        
    return {"route": final_route, "expanded_queries": expanded}

def retriever_node(state: AgentState):
    """Executes the Hybrid Search & RRF."""
    retriever_engine = get_retriever()
    chunks = retriever_engine.retrieve(state["expanded_queries"])
    return {"retrieved_chunks": chunks}

def synthesis_node(state: AgentState):
    """Drafts the final response using only the retrieved context."""
    formatted_prompt = build_synthesis_prompt(state)
    if formatted_prompt is None:
        return {"answer": "I don't have enough information in my library to answer this."}
    
    if config["system"]["inference_mode"] == "openai":
        llm = get_openai_llm(config["openai"]["synthesis_model"])
        answer = llm.invoke(formatted_prompt).content
    else:
        # Local MLX Execution
        answer = local_generate(formatted_prompt, max_tokens=1024)
        
    return {"answer": answer}

def greeting_node(state: AgentState):
    """Fast response for conversational greetings."""
    return {"answer": "Hi! I am CiteMentor. I can offer advice based on my curated library of non-fiction books covering finance, philosophy, and relationships. How can I help you today?"}

def out_of_scope_node(state: AgentState):
    """Fast fallback for topics not covered in the library."""
    return {"answer": "This topic falls outside my curated non-fiction library. Please consult external resources."}

def unsafe_node(state: AgentState):
    """Fast fallback for blocked inputs."""
    return {"answer": f"Request blocked: {state['guardrail_reason']}"}

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
workflow.add_edge("synthesis", END)
workflow.add_edge("out_of_scope", END)
workflow.add_edge("unsafe", END)
workflow.add_edge("greeting", END)

app_graph = workflow.compile()
