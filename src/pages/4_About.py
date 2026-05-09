import streamlit as st

st.set_page_config(page_title="About CiteMentor 2.0", layout="wide")

st.title("🧭 About CiteMentor 2.0")
st.caption("An attribution-aware, agentic RAG portfolio project for applied AI engineering.")

st.markdown(
    """
    **CiteMentor** turns a curated non-fiction library into an interactive mentorship engine.
    Instead of asking a general-purpose model for broad advice, users ask practical life,
    career, finance, philosophy, or relationship questions and receive answers grounded in
    retrieved passages from trusted books.

    The central idea is simple: **AI answers should show their work, respect source material,
    and make attribution measurable.**
    """
)

st.markdown("### Why this project exists")

problem_cols = st.columns(3)
with problem_cols[0]:
    st.markdown(
        """
        **Reader gap**

        Books contain durable advice, but readers often need a focused answer at the moment
        they face a decision.
        """
    )

with problem_cols[1]:
    st.markdown(
        """
        **Trust gap**

        Generic LLM answers can sound confident while being vague, ungrounded, or disconnected
        from an explicit source of truth.
        """
    )

with problem_cols[2]:
    st.markdown(
        """
        **Creator gap**

        Authors rarely get transparent attribution when their ideas influence AI-generated
        outputs.
        """
    )

st.markdown("### What CiteMentor 2.0 does")

st.markdown(
    """
    - Routes each user query through a LangGraph workflow with input safety checks, domain
      classification, retrieval, reranking, and grounded response synthesis.
    - Retrieves evidence using a hybrid search pipeline that combines Chroma semantic search,
      BM25 lexical search, reciprocal rank fusion, and a local cross-encoder reranker.
    - Shows source cards for the passages used in each answer so users can inspect the evidence.
    - Tracks a micro-royalty ledger that estimates the fractional knowledge cost of each
      retrieved snippet.
    - Logs out-of-scope queries as library gaps, turning product usage into a practical roadmap
      for future ingestion.
    - Supports local inference with MLX on Apple Silicon, while keeping optional DeepEval evals
      available through OpenAI-backed evaluation models.
    """
)

st.markdown("### Version 1 to Version 2")

v1, v2 = st.columns(2)
with v1:
    st.markdown(
        """
        **Version 1**

        - Focused on proving the core RAG experience.
        - Used a small public-domain library of three books.
        - Built the first vector database externally on Colab because local hardware was limited.
        - Prioritized core retrieval and citation behavior over production concerns.
        """
    )

with v2:
    st.markdown(
        """
        **Version 2**

        - Rebuilt as a more complete portfolio system with LangGraph orchestration.
        - Creates and serves the vector database locally on a 24 GB Apple Silicon machine.
        - Adds guardrails, hybrid retrieval, reranking, session ledgering, observability, and evals.
        - Treats the product as one serious end-to-end applied AI engineering project.
        """
    )

st.markdown("### Architecture at a glance")

architecture = [
    ("Offline ingestion", "PDF parsing, chunking, contextual summaries, embeddings, Chroma persistence, and BM25 indexing."),
    ("Guardrails", "Lightweight input and output checks for sensitive data, prompt-injection patterns, and ungrounded responses."),
    ("Router", "Classifies each query and generates retrieval expansions in one structured call."),
    ("Semantic cache", "Reuses prior grounded answers when a new query is semantically similar enough."),
    ("Retriever", "Combines semantic and lexical search, deduplicates evidence, fuses rankings, and reranks candidates."),
    ("Synthesizer", "Generates an answer using only retrieved context and returns source-backed mentorship."),
    ("Observability", "Tracks DeepEval scores, latency spans, library gaps, and per-session micro-royalty transactions."),
]

for name, description in architecture:
    st.markdown(f"**{name}:** {description}")

st.markdown("### Portfolio signal")

st.markdown(
    """
    CiteMentor 2.0 is designed to demonstrate the applied AI engineering skills behind a
    production-minded RAG system: ingestion design, retrieval quality, local model serving,
    orchestration, safety boundaries, eval instrumentation, transparent attribution, and a
    user-facing product narrative.
    """
)
