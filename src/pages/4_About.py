import os
import sys

import streamlit as st

# Make `src.*` importable even when this page is opened via a direct deep link
# (e.g. /About), i.e. before the app.py entrypoint has added the project root to
# sys.path. src/pages/4_About.py → two levels up is the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import api_client

st.set_page_config(page_title="About CiteMentor 2.0", layout="wide")

# Phase 8: this same page ships on two deployments. When AWS credentials are present
# (the aws-deploy Streamlit app), the UI is a thin front end to a serverless backend;
# otherwise it runs the pipeline locally. The About copy reflects whichever it is.
USE_REMOTE = api_client.is_configured()

st.title("🧭 About CiteMentor 2.0")
st.caption("An attribution-aware, agentic RAG portfolio project for applied AI engineering.")

if USE_REMOTE:
    st.success(
        "🟢 **This instance is served by AWS Lambda.** The Streamlit UI here is a thin "
        "front end; the full RAG pipeline runs on a serverless, scale-to-zero AWS backend "
        "(see *How this instance is deployed* below)."
    )

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
    - Keeps optional DeepEval faithfulness/relevance grading available through OpenAI-backed
      evaluation models.
    """
)

st.markdown("### How this instance is deployed")

if USE_REMOTE:
    st.markdown(
        """
        This instance runs the **serverless AWS deployment**. The Streamlit app you're using is
        deployed on Streamlit Community Cloud and calls a backend running entirely on AWS:

        - **Compute:** the full pipeline is packaged as a container image and runs on **AWS Lambda**
          (ARM64/Graviton), fronted by a **Lambda Function URL** with `AWS_IAM` authentication.
        - **How the UI talks to it:** requests are **SigV4-signed** by a dedicated least-privilege
          IAM user; the UI **auto-discovers** the current Function URL (it changes each deploy).
        - **Secrets:** the OpenAI key lives in **AWS Secrets Manager** and is fetched at cold start —
          never baked into the image.
        - **Infrastructure as code:** the whole stack is **Terraform**; images are stored in **ECR**;
          deploys and teardowns run from **GitHub Actions** via **OIDC** (no long-lived cloud keys).
        - **Observability & cost:** **CloudWatch** alarms (error rate, p95 latency) notify over **SNS**,
          with a standing **AWS Budget** alert.
        - **Scale-to-zero:** when idle or torn down, AWS spend returns to roughly **\\$0**. If the backend
          is down, this UI degrades gracefully to a clear "backend offline" message.

        The pipeline logic is identical to the local build — the core is cleanly decoupled from the UI,
        so the same code runs in-process locally or behind Lambda here.
        """
    )
else:
    st.markdown(
        """
        This instance runs the pipeline **in-process**, inside the Streamlit app itself:

        - Local inference with **MLX on Apple Silicon**, or an **OpenAI** API path for fast demos and
          cloud deployment (selectable via `inference_mode`).
        - The vector database (Chroma) and BM25 index are served directly from the app process.
        - The same pipeline is also deployed as a serverless AWS backend on a separate instance — the
          core is cleanly decoupled from the UI, so identical code runs either way.
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
        - Ships a serverless AWS deployment (Lambda + Terraform + CI/CD) alongside the local build.
        """
    )

st.markdown("### Architecture at a glance")

architecture = [
    ("Offline ingestion", "PDF parsing, chunking, contextual summaries, embeddings, Chroma persistence, and BM25 indexing."),
    ("Guardrails", "Lightweight input and output checks for sensitive data, prompt-injection patterns, and ungrounded responses."),
    ("Semantic cache", "Runs right after the input guardrail and short-circuits to a prior grounded answer when a new query is semantically similar enough, skipping routing and retrieval."),
    ("Router", "Classifies each query and generates retrieval expansions in one structured call."),
    ("Retriever", "Combines semantic and lexical search, deduplicates evidence, fuses rankings, and reranks candidates."),
    ("Synthesizer", "Generates an answer using only retrieved context and returns source-backed mentorship."),
    ("Observability", "Tracks DeepEval scores, latency spans, library gaps, and per-session micro-royalty transactions."),
]

if USE_REMOTE:
    architecture.append(
        ("Serverless delivery", "The pipeline runs on AWS Lambda behind an IAM-authed Function URL; Terraform + GitHub Actions (OIDC) manage a scale-to-zero, fully-reproducible stack.")
    )

for name, description in architecture:
    st.markdown(f"**{name}:** {description}")

st.markdown("### Portfolio signal")

st.markdown(
    """
    CiteMentor 2.0 is designed to demonstrate the applied AI engineering skills behind a
    production-minded RAG system: ingestion design, retrieval quality, local model serving,
    orchestration, safety boundaries, eval instrumentation, transparent attribution, a
    user-facing product narrative, and a reproducible serverless cloud deployment.
    """
)
