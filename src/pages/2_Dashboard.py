import streamlit as st
import pandas as pd

st.set_page_config(page_title="System Observability", layout="wide")

st.title("📊 System Observability & RAGAS Metrics")
st.caption("Monitor retrieval performance and identify library gaps for future ingestion.")

# --- Dynamic RAGAS Metrics ---
st.markdown("### 🎯 Live Session Retrieval Performance")

eval_data = st.session_state.get("ragas_evals", [])

if not eval_data:
    st.info("No evaluations run yet. Enable 'Live RAGAS Eval' in the Mentor chat to generate data.")
    f_score, r_score = 0.0, 0.0
else:
    # Calculate rolling averages
    df_evals = pd.DataFrame(eval_data)
    f_score = df_evals["faithfulness"].mean()
    r_score = df_evals["answer_relevancy"].mean()

m1, m2, m3 = st.columns(3)

m1.metric(
    "Faithfulness (Live)", 
    f"{f_score:.2f}" if eval_data else "N/A", 
    help="Measures hallucinations. Is the answer grounded strictly in the retrieved snippets?"
)
m2.metric(
    "Answer Relevance (Live)", 
    f"{r_score:.2f}" if eval_data else "N/A", 
    help="Measures evasion. Does the answer directly address the user's prompt?"
)
m3.metric(
    "Context/Correctness Metrics", 
    "Requires Benchmark", 
    help="Context Precision, Recall, and Answer Correctness require a Ground Truth dataset. They cannot be calculated on live arbitrary queries."
)

if eval_data:
    with st.expander("View Query-Level Evaluation Logs"):
        st.dataframe(df_evals, use_container_width=True)

st.markdown("---")

# --- Library Gaps (Out of Scope Queries) ---
st.markdown("### 🕳️ Library Knowledge Gaps")
st.caption("These real-time queries fell outside your current library. Use this data to prioritize future ingestions.")

gaps_data = st.session_state.get("gaps_log", [])

if not gaps_data:
    st.info("No library gaps detected in the current session.")
else:
    df_gaps = pd.DataFrame(gaps_data)
    df_grouped = df_gaps.groupby('query').size().reset_index(name='frequency')
    df_grouped = df_grouped.sort_values(by='frequency', ascending=False)
    st.dataframe(df_grouped, use_container_width=True, hide_index=True)