# CiteMentor 2.0

**An attribution-aware, agentic RAG mentorship engine for non-fiction knowledge.**

CiteMentor transforms a curated library of non-fiction books into an interactive
AI mentor. Instead of giving generic answers, it retrieves relevant source
passages, synthesizes grounded guidance, shows citations, and tracks a
micro-royalty-style attribution ledger for the snippets used in each answer.

This is version 2 of the project. Version 1 proved the core idea with a small
three-book public-domain demo. Version 2 turns it into a fuller applied AI
engineering portfolio project with local ingestion, LangGraph orchestration,
hybrid retrieval, guardrails, evals, observability, and deployment-friendly
OpenAI inference mode.

## Why This Exists

Modern AI advice systems have three major gaps:

1. **Reader gap:** Books contain deep practical wisdom, but people often need a
   focused answer at the moment they face a decision.
2. **Trust gap:** Generic LLMs can be fluent while remaining vague, ungrounded,
   or disconnected from verifiable source material.
3. **Creator gap:** Authors rarely receive transparent attribution when their
   ideas influence AI-generated answers.

CiteMentor explores a different pattern: a grounded mentorship interface where
answers are traceable to specific retrieved passages and source usage is
measurable.

## Core Features

- **Agentic RAG workflow:** LangGraph coordinates input guardrails, routing,
  query expansion, retrieval, reranking, and grounded synthesis.
- **Hybrid retrieval:** Combines vector search, BM25 lexical search, reciprocal
  rank fusion, and reranking.
- **Source cards:** Every standard answer displays the retrieved book snippets
  used as evidence.
- **Micro-royalty ledger:** Each retrieved snippet contributes a fractional
  cost based on book price and total chunk count.
- **Observability dashboard:** Tracks live RAGAS scores and logs out-of-scope
  queries as library gaps.
- **Local-first mode:** Runs routing, synthesis, embeddings, and reranking
  locally with MLX and local rerankers.
- **OpenAI demo mode:** Uses OpenAI API models for routing, query expansion,
  reranking, synthesis, and evals to keep hosted demos fast.

## Version 1 to Version 2

| Area | Version 1 | Version 2 |
| --- | --- | --- |
| Scope | Core RAG proof of concept | End-to-end portfolio system |
| Library | Three public-domain books | Expandable curated catalog |
| Ingestion | Vector DB created externally on Colab | Local ingestion on Apple Silicon |
| Orchestration | Basic retrieval flow | LangGraph agent workflow |
| Retrieval | Core semantic search | Hybrid vector + BM25 + RRF + reranking |
| Safety | Minimal | Input guardrails |
| Observability | Minimal | Dashboard, RAGAS evals, gap logging |
| Attribution | Source display | Source display plus micro-royalty ledger |
| Deployment | Demo-focused | Local mode plus OpenAI API mode |

## Architecture

```text
User Query
   |
   v
Input Guardrail
   |
   v
Router + Query Expansion
   |
   v
Hybrid Retriever
   |-- Local mode: Chroma semantic search + BM25 + local cross-encoder
   |-- OpenAI mode: OpenAI-embedded Chroma search + BM25 + OpenAI reranking agent
   |
   v
Grounded Synthesis
   |
   v
Answer + Source Cards + Royalty Ledger
   |
   v
Observability Dashboard + Library Gap Log
```

## Inference Modes

The main switch lives in `config/retrieval.yaml`:

```yaml
system:
  inference_mode: "local" # or "openai"
```

### Local mode

Use this mode when developing on a capable Apple Silicon machine.

- Router and synthesis use the configured MLX local LLM.
- Chroma queries use the local embedding model that matches ingestion.
- Reranking uses the local cross-encoder.
- No OpenAI API call is required for core answering.
- Live RAGAS evals are disabled in the UI so local mode remains fully local.

### OpenAI mode

Use this mode for Streamlit deployment or fast demos.

- Router uses `openai.router_model`.
- Query expansion uses `openai.query_expansion_model`.
- Reranking uses `openai.reranker_model`.
- Final answer synthesis uses `openai.synthesis_model`.
- RAGAS evals use `openai.eval_model` and `openai.eval_embedding_model`.
- Local embedding and local cross-encoder models are not loaded.
- Semantic search uses the `citementor_library_openai` Chroma collection,
  which is embedded with `openai.embedding_model`.

The default OpenAI choices are optimized for demo latency:

```yaml
openai:
  router_model: "gpt-5-nano"
  query_expansion_model: "gpt-5-nano"
  reranker_model: "gpt-5-nano"
  synthesis_model: "gpt-5-mini"
  eval_model: "gpt-5-mini"
  eval_embedding_model: "text-embedding-3-small"
```

## Tech Stack

- **UI:** Streamlit
- **Workflow orchestration:** LangGraph
- **LLM framework:** LangChain
- **Vector database:** ChromaDB
- **Lexical retrieval:** BM25
- **Local inference:** MLX LM
- **Local embeddings:** `nomic-ai/nomic-embed-text-v1.5`
- **Local reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **OpenAI models:** Configurable GPT-5 family models and
  `text-embedding-3-small`
- **Evaluation:** RAGAS

## Project Structure

```text
.
├── config/
│   └── retrieval.yaml          # Retrieval and model configuration
├── src/
│   ├── app.py                  # Streamlit navigation entrypoint
│   ├── core/
│   │   ├── graph.py            # LangGraph workflow
│   │   ├── guardrails.py       # Input safety checks
│   │   ├── ledger.py           # Micro-royalty accounting
│   │   └── retriever.py        # Hybrid retrieval and reranking
│   ├── pages/
│   │   ├── 1_Mentor.py         # Chat interface
│   │   ├── 2_Dashboard.py      # RAGAS and gap observability
│   │   ├── 3_Ledger.py         # Royalty ledger page
│   │   └── 4_About.py          # Portfolio project overview
│   └── utils/
│       └── ingestion.py        # Local ingestion pipeline
├── storage/
│   ├── chroma_db/              # Persistent Chroma database
│   └── bm25/                   # Serialized BM25 index
├── catalog.json                # Book metadata and pricing inputs
├── prompts.yaml                # Router, expansion, and synthesis prompts
├── pyproject.toml              # Python dependencies
└── README.md
```

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Create a `.env` file if you plan to use OpenAI mode or live RAGAS evals:

```bash
OPENAI_API_KEY=sk-proj-...
```

Run the app:

```bash
uv run streamlit run src/app.py
```

Open the local Streamlit URL printed in the terminal, usually:

```text
http://localhost:8501
```

## Ingestion

Place PDFs in `data/books/` and ensure `catalog.json` contains matching
metadata.

For local ingestion from PDFs, run:

```bash
uv run python -m src.utils.ingestion
```

For OpenAI demo/deployment mode, rebuild the OpenAI vector collection from the
already enriched local Chroma documents:

```bash
uv run python -m src.utils.ingestion --profile openai --reset
```

This creates or refreshes `citementor_library_openai` using
`text-embedding-3-small` while preserving the contextual chunk text generated by
the local ingestion pipeline. It is much faster and cheaper than asking an LLM
to regenerate summaries for every chunk.

If you explicitly want to regenerate contextual summaries through OpenAI as
well, use:

```bash
uv run python -m src.utils.ingestion --profile openai --source pdf --reset
```

That path is slower because it calls an OpenAI chat model for contextual
summaries before embedding.

The local PDF ingestion pipeline:

1. Loads PDFs.
2. Splits documents into overlapping chunks.
3. Uses a local MLX model to generate contextual summaries.
4. Prepends contextual summaries to chunks.
5. Embeds enriched chunks into Chroma.
6. Builds a BM25 index for lexical retrieval.
7. Updates chunk counts in `catalog.json` for royalty calculation.

## Portfolio Notes

CiteMentor 2.0 is meant to demonstrate production-minded applied AI skills:

- Designing a local ingestion pipeline.
- Building retrieval beyond basic "chat with PDF."
- Using orchestration instead of a single monolithic chain.
- Separating local development mode from hosted demo mode.
- Adding safety checks and evaluation hooks.
- Making attribution visible to users.
- Turning user failures into a library expansion signal.

## Current Limitations

- OpenAI semantic search requires the `citementor_library_openai` collection.
  Run `uv run python -m src.utils.ingestion --profile openai --reset` after
  changing the local collection or adding new books.
- Guardrails are currently lightweight and input-focused.
- The royalty ledger is a prototype accounting mechanism, not a payment system.
- Live RAGAS evals are useful for demos but should be supplemented with a fixed
  benchmark set for serious regression testing.
