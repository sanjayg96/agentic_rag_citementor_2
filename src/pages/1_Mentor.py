import json
import yaml
import streamlit as st

from src.core.graph import app_graph
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
            st.info(f"🔬 **Eval Complete** — Faithfulness: {message['eval']['faithfulness']:.2f} | Relevance: {message['eval']['answer_relevancy']:.2f}")
        elif "eval_error" in message and message["eval_error"]:
            st.warning(f"⚠️ **Eval Failed:** {message['eval_error']}")

# Chat Input
if prompt := st.chat_input("Ask for mentorship or advice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and querying library..."):
            initial_state = {"query": prompt}
            result = app_graph.invoke(initial_state)
            
            # Guardrails & Routing Handlers
            if result.get("is_safe") is False:
                st.error(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                st.stop()
            
            if result.get("route") == "greeting":
                st.write(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                st.stop()
            
            if result.get("route") == "out_of_scope":
                st.info("Badge: Public Domain Knowledge (Zero Charge)")
                st.write(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                st.session_state.gaps_log.append({"query": prompt})
                st.stop()

            # Display Standard RAG Answer
            st.markdown(result["answer"])
            
            retrieved_chunks = result.get("retrieved_chunks", [])
            processed_sources = []
            
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
            eval_data_to_save = None
            eval_error_to_save = None
            
            if run_eval and retrieved_chunks:
                with st.spinner("🔬 Running RAGAS Evaluation..."):
                    try:
                        from datasets import Dataset
                        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                        from ragas import evaluate
                        from ragas.embeddings import LangchainEmbeddingsWrapper
                        from ragas.llms import LangchainLLMWrapper
                        from ragas.metrics import Faithfulness, AnswerRelevancy

                        contexts = [c["text"] for c in retrieved_chunks]
                        
                        # 1. Update to the strict RAGAS v0.2+ Dataset Schema
                        eval_data = {
                            "user_input": [prompt],
                            "response": [result["answer"]],
                            "retrieved_contexts": [contexts]
                        }
                        dataset = Dataset.from_dict(eval_data)
                        
                        # Initialize standard Langchain models
                        base_llm = ChatOpenAI(model=config["openai"]["eval_model"], temperature=0)
                        base_embeddings = OpenAIEmbeddings(model=config["openai"]["eval_embedding_model"])
                        
                        # Wrap them for RAGAS
                        ragas_llm = LangchainLLMWrapper(base_llm)
                        ragas_embeddings = LangchainEmbeddingsWrapper(base_embeddings)
                        
                        # Instantiate metrics
                        faithfulness_metric = Faithfulness(llm=ragas_llm)
                        relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
                        
                        # Evaluate
                        score = evaluate(
                            dataset, 
                            metrics=[faithfulness_metric, relevancy_metric]
                        )
                        
                        score_df = score.to_pandas()
                        score_dict = score_df.iloc[0].to_dict()
                        
                        eval_data_to_save = {
                            "query": prompt,
                            "faithfulness": float(score_dict.get("faithfulness", 0.0)),
                            "answer_relevancy": float(score_dict.get("answer_relevancy", 0.0))
                        }
                        st.session_state.ragas_evals.append(eval_data_to_save)
                        
                    except Exception as e:
                        # Better error logging to catch the real issue if it fails
                        eval_error_to_save = f"{type(e).__name__}: {str(e)}"

            # Save state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["answer"],
                "sources": processed_sources,
                "eval": eval_data_to_save,
                "eval_error": eval_error_to_save
            })
            
            st.rerun()
