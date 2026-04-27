import streamlit as st
from src.core.graph import app_graph
from src.core.ledger import record_transaction
import json

try:
    with open("catalog.json", "r") as f:
        catalog_data = json.load(f)
except Exception:
    catalog_data = {}

st.title("💬 CiteMentor 2.0")
st.caption("Agentic Mentorship powered by Local MLX & Hybrid RAG")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "royalties" not in st.session_state:
    st.session_state["royalties"] = 0.0
if "ledger_details" not in st.session_state:
    st.session_state["ledger_details"] = []
if "gaps_log" not in st.session_state:
    st.session_state["gaps_log"] = [] # New: Track out-of-scope queries

# Sidebar Controls & Ledger
st.sidebar.markdown("### ⚙️ Controls")
if st.sidebar.button("🗑️ Reset Session", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Micro-Royalties")
st.sidebar.metric("Accumulated Knowledge Cost", f"${st.session_state.get('royalties', 0.0):.6f}")

# Display Chat History (Now includes sources)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Render historical source cards
        if "sources" in message and message["sources"]:
            st.markdown("---")
            for chunk in message["sources"]:
                title = catalog_data.get(chunk["book_id"], {}).get("title", chunk["book_id"])
                with st.expander(f"📖 {title} | Author: {chunk['author']}"):
                    st.markdown(f"**Snippet Cost:** `${chunk.get('cost', 0.0):.6f}`")
                    st.write(chunk["text"])

# Chat Input
if prompt := st.chat_input("Ask for mentorship or advice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and querying library..."):
            initial_state = {"query": prompt}
            result = app_graph.invoke(initial_state)
            
            # 1. Handle Unsafe Inputs
            if result.get("is_safe") is False:
                st.error(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                st.stop()
            
            # 2. Handle Greetings
            if result.get("route") == "greeting":
                st.write(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                st.stop()
            
            # 3. Handle Out of Scope
            if result.get("route") == "out_of_scope":
                st.info("Badge: Public Domain Knowledge (Zero Charge)")
                st.write(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                
                # Log this explicitly for the Dashboard
                st.session_state.gaps_log.append({"query": prompt})
                st.stop()

            # 4. Handle Standard RAG Answer
            st.markdown(result["answer"])
            
            # Process Sources & Financial Ledger
            retrieved_chunks = result.get("retrieved_chunks", [])
            processed_sources = []
            
            if retrieved_chunks:
                st.markdown("---")
                st.markdown("### 📚 Source Citations & Ledger")
                
                for chunk in retrieved_chunks:
                    # Record transaction only once per fresh query
                    cost = record_transaction(st.session_state, chunk["book_id"])
                    chunk["cost"] = cost # Save cost directly into chunk for history
                    processed_sources.append(chunk)
                    
                    title = catalog_data.get(chunk["book_id"], {}).get("title", chunk["book_id"])
                    with st.expander(f"📖 {title} | Author: {chunk['author']}"):
                        st.markdown(f"**Snippet Cost:** `${cost:.6f}`")
                        st.write(chunk["text"])

            # Save the complete state to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["answer"],
                "sources": processed_sources
            })
            
            st.rerun()