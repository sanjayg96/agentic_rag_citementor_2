import json
import math
import os
import sys
import time
import yaml
import streamlit as st

from src.core.graph import (
    input_guard_node,
    router_node,
    retriever_node,
    stream_synthesis_answer,
)
from src.core.guardrails import check_output_safety
from src.core.ledger import record_transaction
from src.core.semantic_cache import SemanticAnswerCache

with open("config/retrieval.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

st.title("💬 CiteMentor 2.0")
st.caption("Agentic Mentorship powered by Local MLX & Hybrid RAG")

# Load Catalog for Titles
try:
    with open("catalog.json", "r") as f:
        catalog_data = json.load(f)
except Exception:
    catalog_data = {}

def _safe_float(value):
    score = float(value)
    if math.isnan(score):
        raise ValueError("Evaluator returned NaN for at least one metric.")
    return score

@st.cache_resource(show_spinner=False)
def get_answer_cache():
    return SemanticAnswerCache()

def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)

def _merge_timings(state: dict, updates: dict[str, float]) -> dict[str, float]:
    state["timings"] = {**state.get("timings", {}), **updates}
    return state["timings"]

def run_live_eval(prompt: str, answer: str, retrieved_chunks: list[dict]) -> tuple[dict | None, str | None]:
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ModuleNotFoundError as import_error:
        return None, (
            f"DeepEval is not installed in the Python environment running Streamlit: {sys.executable}. "
            "Run the app from this project with `uv run streamlit run src/app.py` after `uv sync`."
        )

    try:
        contexts = [c["text"] for c in retrieved_chunks]
        test_case = LLMTestCase(
            input=prompt,
            actual_output=answer,
            retrieval_context=contexts,
        )

        eval_model = config["openai"]["eval_model"]
        faithfulness = FaithfulnessMetric(model=eval_model, include_reason=True, async_mode=False)
        answer_relevancy = AnswerRelevancyMetric(model=eval_model, include_reason=True, async_mode=False)
        faithfulness.measure(test_case)
        answer_relevancy.measure(test_case)

        eval_data = {
            "query": prompt,
            "faithfulness": _safe_float(faithfulness.score),
            "answer_relevancy": _safe_float(answer_relevancy.score),
            "method": "deepeval",
            "faithfulness_reason": faithfulness.reason,
            "answer_relevancy_reason": answer_relevancy.reason,
        }
        return eval_data, None
    except Exception as eval_error:
        return None, f"DeepEval failed: {type(eval_error).__name__}: {eval_error}"

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "royalties" not in st.session_state:
    st.session_state["royalties"] = 0.0
if "ledger_details" not in st.session_state:
    st.session_state["ledger_details"] = []
if "gaps_log" not in st.session_state:
    st.session_state["gaps_log"] = []
if "evals" not in st.session_state:
    st.session_state["evals"] = []
if "latency_spans" not in st.session_state:
    st.session_state["latency_spans"] = []

# Sidebar Controls & Ledger
st.sidebar.markdown("### ⚙️ Controls")
openai_mode = config["system"].get("inference_mode") == "openai"
run_eval = st.sidebar.toggle(
    "🔬 Enable Live DeepEval",
    value=False,
    disabled=not openai_mode,
    help=(
        "Uses DeepEval with OpenAI to grade the response. Adds evaluation latency."
        if openai_mode
        else "Live API evals are disabled in local mode so the full app stays local."
    )
)

if st.sidebar.button("🗑️ Reset Session", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Micro-Royalties")
st.sidebar.metric("Accumulated Knowledge Cost", f"${st.session_state.get('royalties', 0.0):.6f}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("cache_hit"):
            st.caption(f"Semantic cache hit ({message.get('cache_similarity', 0.0):.2f} similarity)")
        
        # Render Sources
        if "sources" in message and message["sources"]:
            st.markdown("---")
            for chunk in message["sources"]:
                title = catalog_data.get(chunk["book_id"], {}).get("title", chunk["book_id"])
                with st.expander(f"📖 {title} | Author: {chunk['author']}"):
                    st.markdown(f"**Snippet Cost:** `${chunk.get('cost', 0.0):.6f}`")
                    st.write(chunk["text"])
                    
        # Render Eval Results
        if "eval" in message and message["eval"]:
            method = message["eval"].get("method", "deepeval")
            st.info(
                f"🔬 **Eval Complete ({method})** — "
                f"Faithfulness: {message['eval']['faithfulness']:.2f} | "
                f"Relevance: {message['eval']['answer_relevancy']:.2f}"
            )
        elif "eval_error" in message and message["eval_error"]:
            st.warning(f"⚠️ **Eval Failed:** {message['eval_error']}")

# Chat Input
if prompt := st.chat_input("Ask for mentorship or advice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        state = {"query": prompt, "timings": {}}
        eval_data_to_save = None
        eval_error_to_save = None
        processed_sources = []
        cache_hit = False
        cache_similarity = 0.0

        with st.status("Running CiteMentor pipeline...", expanded=True) as status:
            st.write("Input guardrail: checking safety boundaries")
            started = time.perf_counter()
            guard_update = input_guard_node(state)
            state.update(guard_update)
            state["timings"]["input_guard"] = guard_update.get("timings", {}).get("input_guard", _elapsed_ms(started))

            if state.get("is_safe") is False:
                answer = f"Request blocked: {state['guardrail_reason']}"
                status.update(label="Blocked by input guardrail", state="error")
                st.error(answer)
                st.session_state.latency_spans.append({"query": prompt, **state.get("timings", {})})
                st.session_state.messages.append({"role": "assistant", "content": answer, "timings": state.get("timings", {})})
                st.stop()

            st.write("Semantic cache: checking previous grounded answers")
            started = time.perf_counter()
            cache_result = None
            try:
                cache_result = get_answer_cache().lookup(prompt)
            except Exception as cache_error:
                st.caption(f"Semantic cache skipped: {type(cache_error).__name__}")
            _merge_timings(state, {"semantic_cache": _elapsed_ms(started)})

            if cache_result:
                cache_hit = True
                cache_similarity = cache_result["similarity"]
                answer = cache_result["answer"]
                retrieved_chunks = cache_result["sources"]
                status.update(label=f"Served from semantic cache ({cache_similarity:.2f})", state="complete")
                st.caption(f"Semantic cache hit from: {cache_result.get('matched_query', prompt)}")
                st.markdown(answer)

                if retrieved_chunks:
                    st.markdown("---")
                    st.markdown("### 📚 Source Citations & Ledger")
                    for chunk in retrieved_chunks:
                        cost = record_transaction(st.session_state, chunk["book_id"])
                        chunk["cost"] = cost
                        processed_sources.append(chunk)
                        title = catalog_data.get(chunk["book_id"], {}).get("title", chunk["book_id"])
                        with st.expander(f"📖 {title} | Author: {chunk['author']}"):
                            st.markdown(f"**Snippet Cost:** `${cost:.6f}`")
                            st.write(chunk["text"])

                if run_eval and retrieved_chunks:
                    with st.status("Running DeepEval evaluation...", expanded=True) as eval_status:
                        eval_data_to_save, eval_error_to_save = run_live_eval(prompt, answer, retrieved_chunks)
                        if eval_data_to_save:
                            st.session_state.evals.append(eval_data_to_save)
                            eval_status.update(label="DeepEval evaluation complete", state="complete")
                            st.info(
                                f"🔬 **Eval Complete (deepeval)** — "
                                f"Faithfulness: {eval_data_to_save['faithfulness']:.2f} | "
                                f"Relevance: {eval_data_to_save['answer_relevancy']:.2f}"
                            )
                        else:
                            eval_status.update(label="DeepEval evaluation failed", state="error")
                            st.warning(f"⚠️ **Eval Failed:** {eval_error_to_save}")

                st.session_state.latency_spans.append({"query": prompt, "cache_hit": True, **state.get("timings", {})})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": processed_sources,
                    "eval": eval_data_to_save,
                    "eval_error": eval_error_to_save,
                    "timings": state.get("timings", {}),
                    "cache_hit": cache_hit,
                    "cache_similarity": cache_similarity,
                })
                st.stop()

            st.write("Router: classifying intent and expanding the query")
            started = time.perf_counter()
            route_update = router_node(state)
            state.update(route_update)
            state["timings"]["router"] = route_update.get("timings", {}).get("router", _elapsed_ms(started))
            st.write(f"Route selected: `{state['route']}`")

            if state.get("route") == "greeting":
                answer = "Hi! I am CiteMentor. I can offer advice based on my curated library of non-fiction books covering finance, philosophy, and relationships. How can I help you today?"
                status.update(label="Greeting handled", state="complete")
                st.write(answer)
                st.session_state.latency_spans.append({"query": prompt, **state.get("timings", {})})
                st.session_state.messages.append({"role": "assistant", "content": answer, "timings": state.get("timings", {})})
                st.stop()

            if state.get("route") == "out_of_scope":
                answer = "This topic falls outside my curated non-fiction library. Please consult external resources."
                status.update(label="Out-of-scope route handled", state="complete")
                st.info("Badge: Public Domain Knowledge (Zero Charge)")
                st.write(answer)
                st.session_state.latency_spans.append({"query": prompt, **state.get("timings", {})})
                st.session_state.messages.append({"role": "assistant", "content": answer, "timings": state.get("timings", {})})
                st.session_state.gaps_log.append({"query": prompt})
                st.stop()

            st.write("Retriever: running semantic search, BM25, fusion, and reranking")
            started = time.perf_counter()
            retrieval_update = retriever_node(state)
            state.update(retrieval_update)
            state["timings"].update(retrieval_update.get("timings", {}))
            state["timings"]["retriever"] = retrieval_update.get("timings", {}).get("retriever", _elapsed_ms(started))
            retrieved_chunks = state.get("retrieved_chunks", [])
            st.write(f"Retriever returned `{len(retrieved_chunks)}` final source chunks")

            st.write("Synthesizer: streaming grounded answer")

        answer_placeholder = st.empty()
        streamed_answer = ""
        started = time.perf_counter()
        for token in stream_synthesis_answer(state):
            streamed_answer += token
            answer_placeholder.markdown(streamed_answer + "▌")
        _merge_timings(state, {"synthesis": _elapsed_ms(started)})

        started = time.perf_counter()
        output_check = check_output_safety(streamed_answer, state.get("retrieved_chunks", []))
        _merge_timings(state, {"output_guard": _elapsed_ms(started)})
        if not output_check["is_safe"]:
            streamed_answer = f"I cannot safely return that answer: {output_check['reason']}"
            st.warning("Output guardrail revised the response before saving it.")

        answer_placeholder.markdown(streamed_answer)
        status.update(label="Pipeline complete", state="complete")
        result = {**state, "answer": streamed_answer}

        retrieved_chunks = result.get("retrieved_chunks", [])

        if retrieved_chunks:
            st.markdown("---")
            st.markdown("### 📚 Source Citations & Ledger")
            
            for chunk in retrieved_chunks:
                cost = record_transaction(st.session_state, chunk["book_id"])
                chunk["cost"] = cost 
                processed_sources.append(chunk)
                
                title = catalog_data.get(chunk["book_id"], {}).get("title", chunk["book_id"])
                with st.expander(f"📖 {title} | Author: {chunk['author']}"):
                    st.markdown(f"**Snippet Cost:** `${cost:.6f}`")
                    st.write(chunk["text"])

        if retrieved_chunks and output_check["is_safe"]:
            started = time.perf_counter()
            try:
                get_answer_cache().store(prompt, result["answer"], retrieved_chunks)
            except Exception as cache_error:
                st.caption(f"Semantic cache store skipped: {type(cache_error).__name__}")
            _merge_timings(state, {"semantic_cache_store": _elapsed_ms(started)})

        # --- LIVE EVALUATION ---
        if run_eval and retrieved_chunks:
            with st.status("Running DeepEval evaluation...", expanded=True) as eval_status:
                st.write("Evaluating faithfulness against retrieved contexts")
                st.write("Evaluating answer relevancy against the user query")
                eval_data_to_save, eval_error_to_save = run_live_eval(prompt, result["answer"], retrieved_chunks)
                if eval_data_to_save:
                    st.session_state.evals.append(eval_data_to_save)
                    eval_status.update(label="DeepEval evaluation complete", state="complete")
                    st.info(
                        f"🔬 **Eval Complete (deepeval)** — "
                        f"Faithfulness: {eval_data_to_save['faithfulness']:.2f} | "
                        f"Relevance: {eval_data_to_save['answer_relevancy']:.2f}"
                    )
                else:
                    eval_status.update(label="DeepEval evaluation failed", state="error")
                    st.warning(f"⚠️ **Eval Failed:** {eval_error_to_save}")

        # Save state
        st.session_state.latency_spans.append({"query": prompt, "cache_hit": cache_hit, **state.get("timings", {})})
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "sources": processed_sources,
            "eval": eval_data_to_save,
            "eval_error": eval_error_to_save,
            "timings": state.get("timings", {}),
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
        })
        
        st.rerun()
