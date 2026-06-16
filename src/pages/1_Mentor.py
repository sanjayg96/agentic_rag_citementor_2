import json
import math
import os
import sys
import yaml
import streamlit as st

from src.core.graph import app_graph
from src.core.ledger import record_transaction

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

    # Status panel messages keyed by graph node name. The cache node is named
    # "cache_lookup" in the graph but records its span under "semantic_cache".
    NODE_STATUS_MESSAGES = {
        "input_guard": "Input guardrail: checking safety boundaries",
        "cache_lookup": "Semantic cache: checking previous grounded answers",
        "router": "Router: classifying intent and expanding the query",
        "retriever": "Retriever: running semantic search, BM25, fusion, and reranking",
        "output_guard": "Output guardrail: validating the grounded answer",
    }

    with st.chat_message("assistant"):
        eval_data_to_save = None
        eval_error_to_save = None
        processed_sources = []

        status = st.status("Running CiteMentor pipeline...", expanded=True)
        answer_placeholder = st.empty()
        streamed_answer = ""
        synthesis_announced = False
        final = {}

        # Drive the compiled LangGraph: "updates" powers the status panel,
        # "custom" carries streamed synthesis tokens from get_stream_writer().
        for mode, chunk in app_graph.stream(
            {"query": prompt, "timings": {}}, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                if not synthesis_announced:
                    status.write("Synthesizer: streaming grounded answer")
                    synthesis_announced = True
                streamed_answer += chunk
                answer_placeholder.markdown(streamed_answer + "▌")
                continue

            for node, update in chunk.items():
                update = update or {}
                final.update(update)

                message = NODE_STATUS_MESSAGES.get(node)
                if message:
                    status.write(message)
                if node == "cache_lookup" and final.get("cache_hit"):
                    status.write(
                        f"Semantic cache hit ({final.get('cache_similarity', 0.0):.2f} similarity)"
                    )
                if node == "router":
                    status.write(f"Route selected: `{final.get('route')}`")
                if node == "retriever":
                    status.write(
                        f"Retriever returned `{len(final.get('retrieved_chunks', []))}` final source chunks"
                    )

        # Reconcile final answer: output guard may revise it, and cache hits /
        # terminal routes never stream tokens.
        answer = final.get("answer", streamed_answer)
        cache_hit = bool(final.get("cache_hit", False))
        cache_similarity = float(final.get("cache_similarity", 0.0))
        retrieved_chunks = final.get("retrieved_chunks", [])
        route = final.get("route")
        timings = final.get("timings", {})

        if final.get("is_safe") is False:
            answer_placeholder.error(answer)
            status.update(label="Blocked by input guardrail", state="error")
        elif cache_hit:
            answer_placeholder.markdown(answer)
            status.update(label=f"Served from semantic cache ({cache_similarity:.2f})", state="complete")
            st.caption(f"Semantic cache hit from: {final.get('matched_query', prompt)}")
        else:
            answer_placeholder.markdown(answer)
            status.update(label="Pipeline complete", state="complete")
            if final.get("output_is_safe") is False:
                st.warning("Output guardrail revised the response before saving it.")

        if route == "out_of_scope":
            st.info("Badge: Public Domain Knowledge (Zero Charge)")
            st.session_state.gaps_log.append({"query": prompt})

        # Source cards + micro-royalty ledger (UI concern, reads final state).
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

        # --- LIVE EVALUATION ---
        if run_eval and retrieved_chunks:
            with st.status("Running DeepEval evaluation...", expanded=True) as eval_status:
                st.write("Evaluating faithfulness against retrieved contexts")
                st.write("Evaluating answer relevancy against the user query")
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

        # Save state
        st.session_state.latency_spans.append({"query": prompt, "cache_hit": cache_hit, **timings})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": processed_sources,
            "eval": eval_data_to_save,
            "eval_error": eval_error_to_save,
            "timings": timings,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
        })

        st.rerun()
