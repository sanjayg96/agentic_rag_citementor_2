# CiteMentor 2.0 — AWS Production Deployment

> **Resumable master plan & progress tracker.** This work is multi-session by design.
> Any fresh session should read this file first, check the **Progress Tracker**, and
> continue from the first unchecked item. Keep the checkboxes and "Session log" updated.

## Context

CiteMentor 2.0 runs as a Streamlit app deployed from `master` — that deployment stays
untouched. The goal here is **not** a new product: it's taking the existing RAG pipeline
through a full production lifecycle on AWS — containerize → deploy serverless (scale-to-zero)
→ automate (CI/CD) → observe → document → tear down and recreate on demand. It's also a
deliberate learning exercise, so every file and decision should be explainable in an interview.

All work lives on the `aws-deploy` branch (off `dev`). It is additive and isolated; it does
not change how `master`/Streamlit deploys.

**Pre-flight (done manually, outside Terraform):** AWS account created, MFA on root, dedicated
IAM user `citementor-deploy` with `AdministratorAccess` + access keys, AWS CLI configured
locally, Docker installed. Budget/billing alerts to be confirmed in Phase 7. Terraform uses
local gitignored state (solo project; remote S3+DynamoDB state is a noted team enhancement).

## Key Codebase Findings (verified)

- **Core is cleanly decoupled from Streamlit.** Pipeline logic is in `src/core/` (`graph.py`,
  `retriever.py`, `semantic_cache.py`, `guardrails.py`, `ledger.py`) — no Streamlit imports.
  UI is isolated to `main.py` / `src/pages/`.
- **The whole pipeline is one call:** `app_graph.invoke({"query": <str>, "timings": {}})`
  (`src/core/graph.py`). Returns an `AgentState` dict: `answer`, `retrieved_chunks`
  (id/text/book_id/author/cross_score), `route`, `cache_hit`, `cache_similarity`, guardrail
  flags, per-node `timings`. This is exactly what `POST /query` wraps.
- **Inference mode is `openai`** (`config/retrieval.yaml`). MLX (`local` mode) is Apple-Silicon
  only and cannot run on Lambda Graviton — AWS must run OpenAI mode.
- **Heavy deps are lazily imported and unused in OpenAI mode:**
  - `mlx_lm` → only in `get_local_llm()` / `local_generate()` (local mode only).
  - `sentence_transformers` cross-encoder → only the `cross_encoder` property; with
    `use_llm_reranker: false` (current config) the OpenAI path returns fused results without
    invoking it (`retriever.py:251-261`).
  - ⇒ The API container **drops** `torch`, `torchvision`, `sentence-transformers`, `mlx-lm`,
    `mlx-embedding-models`, `einops`, `deepeval`, `streamlit`. Image ~2GB → a few hundred MB.
- **Secrets:** only `OPENAI_API_KEY` is needed at runtime (embeddings `text-embedding-3-small`,
  router `gpt-5-nano`, synthesis `gpt-5-mini`). `HF_TOKEN` not needed in OpenAI mode. No
  Anthropic/Claude key is used despite the product name.
- **Storage to bundle:** `storage/chroma_db/` (~69MB, collection `citementor_library_openai`) +
  `storage/bm25/bm25_index.pkl` (~6MB) + `catalog.json` + `config/retrieval.yaml` +
  `prompts.yaml`. Paths are **relative**, so the container WORKDIR must contain them.
- **No Langfuse/LangSmith yet** — only DeepEval (UI) + per-node timings. Phase 7 adds Langfuse.

## Decisions Locked

1. **Response mode: synchronous JSON** via API Gateway HTTP API (no token streaming in v1).
2. **OpenAI-only slim container** — drop local-mode/eval/UI deps.
3. **ARM64/Graviton** end-to-end (local M-series build → ARM64 Lambda).
4. Commit existing `dev` changes first, branch `aws-deploy` — **done**.

---

## Progress Tracker

- [x] **Pre-step** — commit dev changes, create `aws-deploy`, commit this tracker
- [ ] **Phase 0** — Extract FastAPI service (`POST /query`, `GET /health`)
- [ ] **Phase 1** — Containerize (ARM64), run & verify locally
- [ ] **Phase 2** — Bundle ChromaDB, `/tmp` copy on cold start, verify retrieval parity
- [ ] **Phase 3** — ECR + Lambda (container, ARM64) + API Gateway, end-to-end live
- [ ] **Phase 4** — Secrets Manager for `OPENAI_API_KEY`, least-priv read at cold start
- [ ] **Phase 5** — Terraform for the whole stack; verify destroy/apply lifecycle
- [ ] **Phase 6** — GitHub Actions CI/CD (OIDC, ARM64 buildx)
- [ ] **Phase 7** — Langfuse tracing + CloudWatch alarms + Budget alert
- [ ] **Phase 8** — Streamlit optionally calls the deployed API (env-driven), graceful degradation
- [ ] **Phase 9** — `DEPLOYMENT.md` (architecture, deploy/teardown, cost, secrets, observability)
- [ ] **Post-pass** — scoped least-privilege IAM policy (replace `AdministratorAccess`), documented

---

## Phase 0 — Extract FastAPI service

New `service/` dir importing the existing core unchanged:

- `service/app.py`:
  - `GET /health` → `{"status":"ok"}` (cheap; no retriever/model init — Lambda warmup probe).
  - `POST /query` (`QueryRequest{query: str}`) → `app_graph.invoke({"query": q, "timings": {}})`,
    returns `QueryResponse{answer, sources, route, cache_hit, cache_similarity, timings}`
    (sources mapped from `retrieved_chunks`). `from src.core.graph import app_graph` — no dup logic.
  - **Mangum** adapter (`handler = Mangum(app)`) so one app runs locally (uvicorn) and on Lambda.
- `service/requirements.txt` — slim/pinned: fastapi, mangum, uvicorn (local), langchain,
  langchain-openai, langgraph, chromadb, rank-bm25, pyyaml, python-dotenv, openai, pydantic.
  Exclude torch/sentence-transformers/mlx/deepeval/streamlit.
- `service/.env.example` — document `OPENAI_API_KEY` (note `HF_TOKEN` not required).
- **Relative-path risk:** core uses `config/...`, `catalog.json`, `storage/...`. Run with repo
  root as CWD (or Docker WORKDIR holding them). Confirm during local run.

## Phase 1 — Containerize (ARM64)

- `service/Dockerfile` on `public.ecr.aws/lambda/python:3.12-arm64`. Copy `src/`, `config/`,
  `prompts.yaml`, `catalog.json`, `storage/`; install `service/requirements.txt`;
  `CMD ["service.app.handler"]`. Build `docker buildx build --platform linux/arm64`.
- Build + run locally; verify `/health` and `/query` against the container before deploying.
- Record image size + `docker inspect` arch == arm64.

## Phases 2–9 (expand when reached)

- **Phase 2:** Chroma read-only in image; cold start copies to `/tmp` and points
  `PersistentClient` there. Make `CHROMA_DIR` env-overridable. Verify retrieval parity with eval
  queries. Document static-corpus tradeoff (switch to managed vector DB when corpus grows /
  multi-tenant / incremental updates).
- **Phase 3:** ECR → push ARM64 image → Lambda (container, arm64, ~1.5–2GB mem, 30–60s timeout)
  → API Gateway HTTP API (`/query`, `/health`). Verify via public URL; note cold start. **Cost:**
  Lambda + HTTP API are pay-per-use / scale-to-zero (~$0 idle); ECR ~ pennies/mo.
- **Phase 4:** `OPENAI_API_KEY` in Secrets Manager; exec role reads only that ARN; service fetches
  at cold start via boto3 (not baked into image).
- **Phase 5 (highest leverage):** Terraform (`main.tf`/`variables.tf`/`outputs.tf`) for ECR, Lambda,
  API Gateway, IAM, Secrets Manager (reference values, no plaintext in state). Local gitignored
  state. Verify `destroy` → `apply` recreates identically. This cycle = the cost on/off switch.
- **Phase 6:** GitHub Actions on push to `aws-deploy`: buildx `--platform linux/arm64` (x86 runners
  → QEMU), push ECR, update Lambda. **OIDC role**, not long-lived keys. Verify arch match.
- **Phase 7:** Langfuse tracing in service; CloudWatch alarms (error rate, p95); AWS Budget ~$5/mo.
- **Phase 8:** Env toggle in Streamlit to call deployed API; deploy that variant to Community Cloud
  (always-on), independent of AWS up/down; graceful error if API torn down.
- **Phase 9:** `DEPLOYMENT.md` — architecture, deploy (`terraform apply` + CI/CD), teardown
  (`terraform destroy`), cost, secrets + observability wiring.
- **Post-pass:** scoped least-privilege IAM (Lambda, ECR, API Gateway, IAM-for-exec-role, Secrets
  Manager, CloudWatch, Budgets); verify stack still works; document broad + scoped policies.

---

## Session log

- **2026-06-16** — Pre-step done: committed dev changes (vector DB rebuild, book corpus, genre
  backfill util, `.gitignore` for `.DS_Store`/Terraform, untracked stray `.DS_Store`), created
  `aws-deploy` off `dev`, added this tracker. Next: Phase 0 (FastAPI extraction) + Phase 1
  (containerize), then pause for review.

## Working notes

- Correctness/clarity over cleverness — every file explainable.
- Flag non-trivial AWS cost implications **before** provisioning.
- Pause for review after each phase.
- Keep this tracker updated so any fresh session resumes cleanly.
