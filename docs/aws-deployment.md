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
- **Langfuse tracing added (Phase 7)** — optional, keyed off `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`; falls back to untraced if unset. DeepEval (UI) + per-node timings remain unchanged.

## Decisions Locked

1. **Response mode: synchronous JSON.** Originally via API Gateway HTTP API; **changed in
   Phase 3 to a Lambda Function URL (AWS_IAM auth)** — API Gateway's hard 30s integration timeout
   is too short for cold starts (~49s) and even some warm LLM syntheses (>30s). The Function URL
   honours the function's own 120s timeout. This account blocks *anonymous* function URLs, so auth
   is `AWS_IAM` and requests are SigV4-signed (helper: `scripts/demo_query.sh`). No token streaming.
2. **OpenAI-only slim container** — drop local-mode/eval/UI deps. (Now also carries a
   `langchain-aws` dep for the optional **bedrock** mode; still excludes torch/MLX/etc.)
3. **ARM64/Graviton** end-to-end (local M-series build → ARM64 Lambda).
4. Commit existing `dev` changes first, branch `aws-deploy` — **done**.
5. **Third inference mode: `bedrock`** (added post-Phase-5). A zero-OpenAI-dependency
   fallback for when the OpenAI balance runs out (Bedrock has no upfront cost, billed on
   the monthly AWS invoice). Models: **Amazon Nova Lite** (router) + **Amazon Nova Pro**
   (synthesis) + **Titan Text Embeddings V2**. *Originally Claude 3 Haiku, but Anthropic
   Claude is a third-party AWS Marketplace model whose subscription kept expiring instantly
   on this account (start==end date) → `AccessDenied` on the cloud Lambda even for admin.
   Switched to first-party Amazon Nova (no Marketplace subscription, auto-enabled, cheaper).*
   Because Chroma binds an embedder to a collection, the corpus is re-embedded with Titan
   into `citementor_library_bedrock` (cheap re-embed of already-enriched chunks, no LLM
   calls). Nova uses cross-region `apac.` inference profiles → IAM grants the profile ARNs
   + underlying model ARNs. Switch via config edit + rebuild (no env-var toggle). Full
   theory + war story in `docs/DEPLOYMENT.md`.

---

## Progress Tracker

- [x] **Pre-step** — commit dev changes, create `aws-deploy`, commit this tracker
- [x] **Phase 0** — Extract FastAPI service (`POST /query`, `GET /health`)
- [x] **Phase 1** — Containerize (ARM64), run & verify locally
- [x] **Phase 2** — Bundle ChromaDB, `/tmp` copy on cold start, verify retrieval parity
- [x] **Phase 3** — ECR + Lambda (container, ARM64) + **Lambda Function URL** (was API Gateway), end-to-end live
- [x] **Phase 4** — Secrets Manager for `OPENAI_API_KEY`, least-priv read at cold start
- [x] **Phase 5** — Terraform for the whole stack; verify destroy/apply lifecycle
- [x] **Phase 6** — GitHub Actions CI/CD (OIDC, native ARM64 runners, S3 remote state)
- [x] **Phase 7** — Langfuse tracing + CloudWatch alarms + Budget alert
- [x] **Phase 8** — Streamlit optionally calls the deployed API (env-driven), graceful degradation
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
- **Phase 8 ✅:** Env toggle in Streamlit (`src/utils/api_client.py`) — when AWS creds are present
  in `st.secrets`, the Mentor page SigV4-signs requests to the Function URL; otherwise it runs the
  pipeline locally (so `master`'s deploy is untouched). URL is auto-discovered via
  `GetFunctionUrlConfig` (it changes each deploy cycle), so the always-on Community Cloud app needs
  no edits between demos. Graceful in-chat message when the stack is torn down. A dedicated
  least-privilege IAM user (`citementor-streamlit`, created by `infra/streamlit_user.sh`, outside
  the ephemeral Terraform so its key survives teardown) is the only thing that can invoke the URL.
- **Phase 9:** `DEPLOYMENT.md` — architecture, deploy (`terraform apply` + CI/CD), teardown
  (`terraform destroy`), cost, secrets + observability wiring.
- **Post-pass:** scoped least-privilege IAM (Lambda, ECR, API Gateway, IAM-for-exec-role, Secrets
  Manager, CloudWatch, Budgets); verify stack still works; document broad + scoped policies.

---

## Session log

- **2026-06-16** — Pre-step done: committed dev changes (vector DB rebuild, book corpus, genre
  backfill util, `.gitignore` for `.DS_Store`/Terraform, untracked stray `.DS_Store`), created
  `aws-deploy` off `dev`, added this tracker.
- **2026-06-16** — **Phase 0 done.** `service/app.py` (Mangum), slim `service/requirements.txt`,
  `service/.env.example`. Verified via uvicorn: `/health` ok; `/query` grounded answer + 3 sources
  + timings; repeat → cache hit (sim ~1.0); empty query → 422.
- **2026-06-16** — **Phase 1 done.** `service/Dockerfile` (`public.ecr.aws/lambda/python:3.12-arm64`)
  + `.dockerignore`. Built `docker buildx build --platform linux/arm64 -f service/Dockerfile -t
  citementor-api:local .` → **arm64/linux**, **~1.5GB** (base image + chromadb's onnxruntime + 75MB
  corpus; later slimming candidate). Ran via Lambda RIE: `GET /health` → 200; `POST /query` → 200,
  grounded answer + 3 sources. **Cold-start data:** module INIT ~265ms (retriever is lazy); first
  `/query` ~26.4s (retriever build + Chroma/BM25 load + OpenAI embed/retrieve + gpt-5-mini synthesis
  ~12s). RIE ran at 3008MB. **Implications for Phase 3:** Lambda timeout must be ≥30s (default 3s
  fails); size memory ~2–3GB; provisioned concurrency or a warmup ping would hide cold starts.
  **Paused for review.** Next: Phase 2 (Chroma → /tmp on cold start, env-overridable `CHROMA_DIR`,
  retrieval parity check).
- **2026-06-18** — **Phase 2 done.** Lambda's image FS is read-only except `/tmp`, but Chroma's
  `PersistentClient` opens its SQLite store read-write — *and* the semantic answer cache **writes**
  into the same DB. New `src/core/storage.py::resolve_chroma_path()`: copies the bundled
  `storage/chroma_db` → `/tmp/chroma_db` once per cold start and returns that writable path;
  memoised so retriever + cache share one copy. Zero-config on Lambda (triggers on
  `AWS_LAMBDA_FUNCTION_NAME`); forceable locally via `CHROMA_COPY_TO_TMP=1`. Source dir / target
  overridable via `CHROMA_DIR` / `CHROMA_TMP_DIR`. `retriever.py` + `semantic_cache.py` now call the
  resolver; BM25 pickle stays put (read-only access is fine on Lambda). **Parity:**
  `scripts/verify_chroma_tmp_parity.py` runs 5 eval queries through the retriever in two
  subprocesses (bundled vs `/tmp` copy) — identical chunk IDs/order → **PASS**. **Container proof:**
  rebuilt ARM64 image, ran via RIE with `--read-only --tmpfs /tmp` + `AWS_LAMBDA_FUNCTION_NAME` set
  (faithful Lambda simulation): `/health` 200; `/query` grounded answer + 3 sources; confirmed
  `/var/task/storage` read-only, `/tmp/chroma_db` populated, and a repeat query → `cache_hit:true`
  (the cache writer successfully upserted to `/tmp`). **Static-corpus tradeoff:** the corpus is
  baked into the image and copied per cold start — fine for a small read-mostly library, but cache
  writes are ephemeral (lost when the execution env recycles) and the image must be rebuilt to
  update the corpus. Switch to a managed/networked vector DB (e.g. Chroma server, pgvector,
  OpenSearch) when the corpus grows large, needs incremental updates, or goes multi-tenant.
  **Paused for review.** Next: Phase 3 (ECR + Lambda + API Gateway, end-to-end live).
- **2026-06-23** — **Phase 3 done (live on AWS).** Region `ap-south-1`, account `505192030409`.
  Pushed ARM64 image to ECR (`citementor-api:latest`, `--provenance=false` to get a single
  Lambda-compatible manifest). Created Lambda `citementor-api` (Image, arm64, 3008MB, timeout 120s)
  with exec role `citementor-lambda-role` (+`AWSLambdaBasicExecutionRole`). **Cold start ~49s**
  (init ~9s + first-query retriever/Chroma-/tmp build + OpenAI calls); warm queries ~12–25s; repeat
  query → semantic-cache hit ~0.5s. **Endpoint pivot — the main Phase 3 story:** started with an
  API Gateway HTTP API but hit its **hard 30s integration timeout** — cold starts always 503'd, and
  even some *warm* gpt-5-mini syntheses exceed 30s, so API Gateway is the wrong front door for
  long synchronous LLM calls. Switched to a **Lambda Function URL** (no 30s cap; honours the 120s
  function timeout). The URL's *anonymous* (`AuthType NONE`) path returned `403 AccessDeniedException`
  despite a correct resource policy and a non-org account — proved via an `AWS_IAM`-signed request
  returning 200 that the function/plumbing are fine, so this is an **account-level block on public
  function URLs**. Settled on **`AuthType AWS_IAM`** + SigV4-signed requests (eliminates *both* the
  timeout and the anonymous block). Added `scripts/demo_query.sh` (signs requests; `--warm`,
  `--health`, or a question) so demos are one command. API Gateway (`9316p7m77a`) left in place but
  unused — can be deleted; Terraform (Phase 5) will define the canonical Function URL. **Demo
  ritual:** `scripts/demo_query.sh --warm` a few min before, then ask questions. **Cost:** Lambda +
  Function URL scale-to-zero (~$0 idle); ECR image ~$0.15/mo. **Paused for review.** Next: Phase 4
  (`OPENAI_API_KEY` → Secrets Manager; small boto3 fetch at cold start).
- **2026-06-23** — **Phase 4 done + API Gateway deleted.** Deleted the unused API Gateway
  (`9316p7m77a`) and its `apigw-invoke` permission. Stored the key in Secrets Manager
  (`citementor/openai_api_key`); granted `citementor-lambda-role` an inline least-priv policy
  (`secretsmanager:GetSecretValue` on **only** that secret ARN). New `service/secrets.py`
  (`load_openai_key_from_secrets`) called at the top of `service/app.py`: if `OPENAI_API_KEY` is
  absent but `OPENAI_SECRET_NAME` is set, fetch via boto3 at cold start and populate the env so the
  lazy core code is unchanged; no-op locally (key already in `.env`). `boto3` is provided by the
  Lambda base image (verified `boto3 1.42.97`), so it's **not** added to requirements; the lazy
  import means local runs never need it. Rebuilt/pushed image, `update-function-code`, then
  `update-function-configuration` to **drop the plaintext `OPENAI_API_KEY`** and set only
  `OPENAI_SECRET_NAME`. Verified: env now holds only the secret name; a cold query (~47s) returns a
  grounded answer, proving the key is fetched from Secrets Manager. **Cost note:** Secrets Manager
  ~$0.40/secret/mo + $0.05/10k calls (the one standing charge besides ECR). **Security follow-up
  (recommended):** the original key was passed inline to `create-function` in earlier manual steps,
  so it may live in terminal scrollback / flush to `~/.zsh_history` on shell exit — **rotate the
  OpenAI key**, `put-secret-value` the new value (no redeploy needed), revoke the old key.
  **Paused for review.** Next: Phase 5 (Terraform — the reproducible apply/destroy lifecycle).
- **2026-06-24** — **Phase 5 done (full lifecycle proven).** Installed Terraform 1.15.6.
  Authored `infra/` (`versions/providers/variables/locals/ecr/secrets/iam/lambda/outputs.tf`
  + `terraform.tfvars.example` + `README.md`) defining the *entire* stack: ECR repo, the
  ARM64 image build/push (a `null_resource` running the same `docker buildx
  --provenance=false --push`), `data.aws_ecr_image` to pin the Lambda to the immutable
  **digest** (not `:latest`, so rebuilds actually roll the function), Secrets Manager
  container, least-priv IAM role, Lambda (Image/arm64/3008MB/120s) + Function URL
  (`AWS_IAM`), and an explicit CloudWatch log group. **Secret hygiene:** Terraform manages
  only the empty secret container; the value is pushed by a `null_resource` whose shell
  reads `$OPENAI_API_KEY` from the env at apply time (`$${OPENAI_API_KEY}` heredoc → literal
  shell var), so **no plaintext ever lands in local state**. **Teardown-by-design:** ECR
  `force_delete=true`, secret `recovery_window_in_days=0`, managed log group. **Verified
  live:** deleted the Phase 0-4 manually-created resources, then `terraform apply` rebuilt
  the identical stack from nothing (10 resources, new Function URL). Ran the demo —
  `--health` → ok, `--warm` → cold start paid, a real question → grounded cited answer
  (warm ~18s: router 5.1s / retriever 3.5s / synthesis 10.0s). Then `terraform destroy` (10
  resources) and confirmed Lambda/ECR/secret/role/log-group all **not found** → spend ~$0.
  **State is local + gitignored** (`*.tfstate*`); the `.tf` files are committed.
  **Comprehensive guide written:** `docs/DEPLOYMENT.md` (story, theory, AWS-services
  glossary, request lifecycle, per-phase narrative, code walkthrough, Terraform stack, cost
  model, security model, lessons) — to be extended each remaining phase. **Paused for
  review.** Next: Phase 6 (GitHub Actions CI/CD with OIDC, ARM64 buildx).

- **2026-07-05** — **Bedrock inference mode added + cloud-tested.** New third `inference_mode:
  bedrock` for zero-OpenAI-dependency demos. Re-embedded the 2800-chunk corpus with **Titan V2**
  into `citementor_library_bedrock` via a generalized `ingestion.rebuild_from_existing` (re-embed
  only, no LLM/OpenAI calls; ~46 min as Titan is sequential). Wired `bedrock` branches into
  `graph.py` (`get_bedrock_llm` = `ChatBedrockConverse`; router structured-output via default tool
  method; `_content_to_text` normalizes streaming chunks), `retriever.py`/`semantic_cache.py`
  (Titan `AmazonBedrockEmbeddingFunction`; same no-cross-encoder path as openai). Added
  `langchain-aws==1.4.6` (kept `langchain-core==1.3.2`). **Model saga:** started on **Claude 3
  Haiku** — passed local tests, but the cloud Lambda hit `AccessDenied` on AWS **Marketplace**
  actions; Anthropic Claude is a *third-party Marketplace* model and this account's agreement kept
  **expiring instantly** (start==end), unfixable from our side (even admin invoke failed). Pivoted
  to **first-party Amazon Nova** (Lite router + Pro synthesis, `apac.` inference profiles) — no
  Marketplace subscription, auto-enabled, cheaper. Updated IAM (`bedrock:InvokeModel` on Titan +
  Nova profile ARNs + underlying regional model ARNs). **Verified full cloud lifecycle:** apply
  (11 res, image rebuilt for bedrock config) → live demo returned a grounded cited answer entirely
  via Nova (router `finance` ~0.6s / Titan retrieval ~2.2s / Nova Pro synthesis ~1.3s → **~4s warm,
  faster than openai**), repeat → `cache_hit` sim 1.0 → destroy (11 res) → **$0**. Whole run used
  an invalid OpenAI key (proves zero OpenAI use). Default `inference_mode` stays `openai`; switch
  to bedrock = edit config + `terraform apply` (rebuilds). **Next:** Phase 6 (GitHub Actions CI/CD).

- **2026-07-07** — **Phase 6 done (CI/CD live, full lifecycle proven from the Actions tab).**
  Two `workflow_dispatch` workflows: **deploy.yml** (a `bedrock`/`openai` mode dropdown →
  OIDC role assume → native **`ubuntu-24.04-arm`** runner builds the arm64 image → `terraform
  apply` → publishes the Function URL) and **teardown.yml** (`terraform destroy` → $0). Auth is
  **GitHub OIDC** (no long-lived keys): `infra/bootstrap.sh` (idempotent, one-time) created the
  IAM OIDC provider, a **scoped** role `citementor-github-actions` trusted only by
  `repo:sanjayg96/agentic_rag_citementor_2:ref:refs/heads/aws-deploy`, and the **S3 remote-state
  bucket** `citementor-tfstate-505192030409` (versioned/encrypted/private, **native S3 locking**
  via `use_lockfile` — no DynamoDB). Migrated local state → S3 (`infra/backend.tf`). Set repo
  vars `AWS_ROLE_ARN`/`AWS_REGION` + secret `OPENAI_API_KEY` via `gh`. **Made `aws-deploy` the
  repo default branch** (GitHub only dispatches workflows that live on the default branch;
  reversible; doesn't affect Streamlit). **Three gotchas fixed live:** (1) workflow-not-found →
  default-branch rule above; (2) `CreateLogGroup` 409 from a *zombie* Lambda log group (late log
  events recreate `/aws/lambda/...` after a destroy) → a pre-apply `delete-log-group` step;
  (3) `AccessDenied` on `logs:ListTagsForResource` → the provider reads log-group tags on the ARN
  *without* the `:*` suffix, so the scoped policy needed that action + both ARN forms. **Verified:**
  deploy workflow (bedrock) → live demo returned a grounded answer via Nova → teardown workflow →
  app resources gone, shared S3 state = 0 resources, standing CI infra (bucket + OIDC role) intact.
  Cost note: the S3 state bucket is the one standing resource (~$0/mo, a few KB); OIDC/IAM are free.
  **Next:** Phase 7 (observability + budget alarms).

- **2026-07-09** — **Phase 7 done (Langfuse + CloudWatch alarms + Budget), verified live end-to-end.**
  **Langfuse tracing** (optional): `service/secrets.py::load_langfuse_keys_from_secrets()` mirrors the
  OpenAI-key pattern (fetch from Secrets Manager at cold start, no-op if already set); `service/app.py`
  builds a cached `langfuse.langchain.CallbackHandler` only if both keys are present and passes it as
  `config={"callbacks": [...]}` to `app_graph.invoke(...)` — zero code changes to `graph.py` since
  LangChain/LangGraph propagate callbacks to nested LLM calls automatically. Both `LANGFUSE_HOST` and
  `LANGFUSE_BASE_URL` env vars are set (SDK v4 renamed the host var; setting both is cheap insurance).
  New Secrets Manager secret `citementor/langfuse_keys`: unlike the mandatory OpenAI secret, this one
  is intentionally optional — `null_resource.langfuse_secret_value` pushes an empty `{}` when
  `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` aren't exported at apply time, and the app just runs
  untraced. **CloudWatch alarms** (`infra/monitoring.tf`, app-level — torn down with the rest since
  they cost a few cents/month while applied): an SNS topic (`citementor-alerts`) with an email
  subscription (`var.alert_email`, defaults to the account owner's email), an `Errors` alarm (any
  error in 5 min — this function is rarely invoked, so even one matters) and a `Duration` p95 alarm
  at 80% of the Lambda timeout (an early warning before requests actually start timing out).
  **AWS Budget** (`infra/bootstrap.sh`, step 4 — deliberately standing/outside the app Terraform,
  since cost risk exists whether or not the Lambda stack is currently deployed): `$5/mo` budget with
  ACTUAL-80%/FORECASTED-100% email notifications, created idempotently alongside the existing state
  bucket/OIDC/CI-role bootstrap. The CI role's scoped policy (`perms.json`) gained `sns:*`/
  `cloudwatch:*` actions scoped to the new alarm/topic ARNs — **existing deployments must re-run
  `bash infra/bootstrap.sh` once** to pick up the updated permissions before the next CI deploy.
  **Verified live end-to-end.** Signed up for Langfuse Cloud, put the key pair in `.env`. Ran the
  service locally (uvicorn, `service/requirements.txt` installed into the project venv): `/query`
  returned a grounded answer, and the trace (full LangGraph input/output, one span per node) showed
  up in the Langfuse dashboard within seconds. Then a real `terraform apply` from this account
  (18 resources: Lambda, both secrets, SNS topic, both alarms, Function URL, etc.) — cold start via
  `demo_query.sh --warm` succeeded (proving the Langfuse secret fetch at cold start didn't break
  anything), a real query via `demo_query.sh` came back grounded, CloudWatch showed both alarms in
  `OK` state and the SNS email subscription registered, and the Langfuse public API
  (`GET /api/public/traces`) confirmed a **new cloud trace** (10 observations) landed right after the
  query — proving tracing works through the full Lambda → Secrets Manager → Langfuse path, not just
  locally. Also ran `bash infra/bootstrap.sh` again: updated the CI role's scoped policy with the new
  `sns:*`/`cloudwatch:*` actions, and created the standing AWS Budget (`citementor-monthly`, $5/mo,
  ACTUAL≥80%/FORECASTED≥100% email alerts) — confirmed via `aws budgets describe-budget` /
  `describe-notifications-for-budget`. Then `terraform destroy` (18 resources) and confirmed Lambda/
  ECR/both secrets **not found** — spend back to ~$0; the state bucket, OIDC role, and the new Budget
  intentionally survive. **Next:** Phase 8 (Streamlit → deployed API).

## Working notes

- Correctness/clarity over cleverness — every file explainable.
- Flag non-trivial AWS cost implications **before** provisioning.
- Pause for review after each phase.
- Keep this tracker updated so any fresh session resumes cleanly.
