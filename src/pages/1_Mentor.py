import json
import math
import yaml
import streamlit as st

from src.core.graph import (
    input_guard_node,
    router_node,
    retriever_node,
    stream_synthesis_answer,
)
from src.core.ledger import record_transaction

with open("config/retrieval.yaml", "r") as f:
    config = yaml.safe_load(f)

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
        raise ValueError("RAGAS returned NaN for at least one metric.")
    return score

def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start:end + 1])

def run_live_eval(prompt: str, answer: str, retrieved_chunks: list[dict]) -> tuple[dict | None, str | None]:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, AnswerRelevancy

        contexts = [c["text"] for c in retrieved_chunks]
        dataset = Dataset.from_dict({
            "user_input": [prompt],
            "response": [answer],
            "retrieved_contexts": [contexts]
        })

        base_llm = ChatOpenAI(model=config["openai"]["eval_model"], temperature=0)
        base_embeddings = OpenAIEmbeddings(model=config["openai"]["eval_embedding_model"])
        ragas_llm = LangchainLLMWrapper(base_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

        score = evaluate(
            dataset,
            metrics=[
                Faithfulness(llm=ragas_llm),
                AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ],
            raise_exceptions=True,
            show_progress=False,
        )

        score_dict = score.to_pandas().iloc[0].to_dict()
        eval_data = {
            "query": prompt,
            "faithfulness": _safe_float(score_dict.get("faithfulness")),
            "answer_relevancy": _safe_float(score_dict.get("answer_relevancy")),
            "method": "ragas",
        }
        return eval_data, None
    except Exception as ragas_error:
        fallback_data = run_fallback_eval(prompt, answer, retrieved_chunks)
        if fallback_data:
            return fallback_data, f"RAGAS unavailable: {type(ragas_error).__name__}: {ragas_error}. Used fallback judge."
        return None, f"{type(ragas_error).__name__}: {ragas_error}"

def run_fallback_eval(prompt: str, answer: str, retrieved_chunks: list[dict]) -> dict | None:
    try:
        from langchain_openai import ChatOpenAI

        contexts = "\n\n".join(f"[Context {idx + 1}] {chunk['text']}" for idx, chunk in enumerate(retrieved_chunks))
        judge_prompt = f"""
You are evaluating a retrieval-augmented answer.
Return only valid JSON with this shape:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "reason": "..."}}

Scoring:
- faithfulness: 1.0 means every important claim is supported by the contexts; 0.0 means unsupported or contradicted.
- answer_relevancy: 1.0 means the answer directly addresses the user query; 0.0 means it is off-topic.

User query:
{prompt}

Answer:
{answer}

Retrieved contexts:
{contexts}
"""
        llm = ChatOpenAI(model=config["openai"]["eval_model"], temperature=0)
        parsed = _extract_json_object(llm.invoke(judge_prompt).content)
        return {
            "query": prompt,
            "faithfulness": max(0.0, min(1.0, float(parsed["faithfulness"]))),
            "answer_relevancy": max(0.0, min(1.0, float(parsed["answer_relevancy"]))),
            "method": "fallback_judge",
            "reason": parsed.get("reason", ""),
        }
    except Exception:
        return None

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "royalties" not in st.session_state:
    st.session_state["royalties"] = 0.0
if "ledger_details" not in st.session_state:
    st.session_state["ledger_details"] = []
if "gaps_log" not in st.session_state:
    st.session_state["gaps_log"] = []
if "ragas_evals" not in st.session_state:
    st.session_state["ragas_evals"] = [] 

# Sidebar Controls & Ledger
st.sidebar.markdown("### ⚙️ Controls")
openai_mode = config["system"].get("inference_mode") == "openai"
run_eval = st.sidebar.toggle(
    "🔬 Enable Live RAGAS Eval",
    value=False,
    disabled=not openai_mode,
    help=(
        "Uses OpenAI to grade the response. Adds 5-10s latency."
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
            method = message["eval"].get("method", "ragas")
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
        state = {"query": prompt}
        eval_data_to_save = None
        eval_error_to_save = None
        processed_sources = []

        with st.status("Running CiteMentor pipeline...", expanded=True) as status:
            st.write("Input guardrail: checking safety boundaries")
            guard_update = input_guard_node(state)
            state.update(guard_update)

            if state.get("is_safe") is False:
                answer = f"Request blocked: {state['guardrail_reason']}"
                status.update(label="Blocked by input guardrail", state="error")
                st.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()

            st.write("Router: classifying intent and expanding the query")
            route_update = router_node(state)
            state.update(route_update)
            st.write(f"Route selected: `{state['route']}`")

            if state.get("route") == "greeting":
                answer = "Hi! I am CiteMentor. I can offer advice based on my curated library of non-fiction books covering finance, philosophy, and relationships. How can I help you today?"
                status.update(label="Greeting handled", state="complete")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()

            if state.get("route") == "out_of_scope":
                answer = "This topic falls outside my curated non-fiction library. Please consult external resources."
                status.update(label="Out-of-scope route handled", state="complete")
                st.info("Badge: Public Domain Knowledge (Zero Charge)")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.gaps_log.append({"query": prompt})
                st.stop()

            st.write("Retriever: running semantic search, BM25, fusion, and reranking")
            retrieval_update = retriever_node(state)
            state.update(retrieval_update)
            retrieved_chunks = state.get("retrieved_chunks", [])
            st.write(f"Retriever returned `{len(retrieved_chunks)}` final source chunks")

            st.write("Synthesizer: streaming grounded answer")

        answer_placeholder = st.empty()
        streamed_answer = ""
        for token in stream_synthesis_answer(state):
            streamed_answer += token
            answer_placeholder.markdown(streamed_answer + "▌")

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

        # --- LIVE RAGAS EVALUATION ---
        if run_eval and retrieved_chunks:
            with st.status("Running RAGAS evaluation...", expanded=True) as eval_status:
                st.write("Evaluating faithfulness against retrieved contexts")
                st.write("Evaluating answer relevancy against the user query")
                eval_data_to_save, eval_error_to_save = run_live_eval(prompt, result["answer"], retrieved_chunks)
                if eval_data_to_save:
                    st.session_state.ragas_evals.append(eval_data_to_save)
                    eval_status.update(label="Evaluation complete", state="complete")
                    st.info(
                        f"🔬 **Eval Complete ({eval_data_to_save.get('method', 'ragas')})** — "
                        f"Faithfulness: {eval_data_to_save['faithfulness']:.2f} | "
                        f"Relevance: {eval_data_to_save['answer_relevancy']:.2f}"
                    )
                    if eval_error_to_save:
                        st.warning(eval_error_to_save)
                else:
                    eval_status.update(label="Evaluation failed", state="error")
                    st.warning(f"⚠️ **Eval Failed:** {eval_error_to_save}")

        # Save state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "sources": processed_sources,
            "eval": eval_data_to_save,
            "eval_error": eval_error_to_save
        })
        
        st.rerun()
