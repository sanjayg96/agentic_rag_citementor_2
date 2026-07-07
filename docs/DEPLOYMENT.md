# CiteMentor 2.0 on AWS — The Complete Deployment Guide

> **What this document is.** The full story of taking the CiteMentor RAG pipeline
> from a local Streamlit app to a serverless, scale-to-zero, fully-reproducible
> AWS deployment. It is written to do three jobs at once:
>
> 1. **Memory refresh** — come back in six months and re-load the whole mental model fast.
> 2. **Learning guide** — explain *why* each AWS service and each line of config exists, from first principles.
> 3. **Interview reference** — every decision here is defensible; this is the script for talking through it.
>
> It is a living document. It currently covers **Phases 0–5** (extract → containerize →
> bundle data → deploy → secrets → Terraform). Phases 6–9 (CI/CD, observability,
> Streamlit integration, hardening) will be appended as they land.
>
> Companion files: [`aws-deployment.md`](aws-deployment.md) is the terse resumable
> tracker (checkboxes + session log). [`../infra/README.md`](../infra/README.md) is
> the operational quick-start. **This** file is the narrative + theory.

---

## Table of contents

1. [TL;DR — the elevator pitch](#1-tldr--the-elevator-pitch)
2. [The product and the goal of this exercise](#2-the-product-and-the-goal-of-this-exercise)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [AWS services used — what they are and why](#4-aws-services-used--what-they-are-and-why)
5. [The request lifecycle](#5-the-request-lifecycle)
6. [The story, phase by phase](#6-the-story-phase-by-phase)
7. [Code walkthrough — how the app is wired](#7-code-walkthrough--how-the-app-is-wired)
8. [The Terraform stack, resource by resource](#8-the-terraform-stack-resource-by-resource)
9. [Operations runbook](#9-operations-runbook)
10. [Cost model](#10-cost-model)
11. [Security model](#11-security-model)
12. [Challenges and lessons (consolidated)](#12-challenges-and-lessons-consolidated)
13. [Glossary](#13-glossary)
14. [What's next (Phases 6–9)](#14-whats-next-phases-69)

---

## 1. TL;DR — the elevator pitch

CiteMentor is an **agentic RAG** (Retrieval-Augmented Generation) service: ask a
question about a small library of philosophy/self-improvement books and it routes
the query, retrieves grounded passages with a **hybrid retriever** (dense vectors +
BM25), and synthesizes a cited answer with guardrails and a semantic answer cache.

The deployment turns that pipeline into a **serverless container on AWS Lambda**:

- The FastAPI app is packaged as an **ARM64 (Graviton) container image**, stored in
  **ECR**, and run by **Lambda** behind a **Lambda Function URL** (HTTPS endpoint).
- It is **scale-to-zero**: idle cost is ~\$0. You only pay per request (plus a few
  cents/month for the stored image and the secret).
- The OpenAI key lives in **Secrets Manager**, read at cold start by a least-
  privilege execution role — never baked into the image or the Terraform state.
- The **entire stack is defined in Terraform**. `terraform apply` builds the image
  and creates everything from nothing; `terraform destroy` returns spend to **\$0**.
  This apply/destroy cycle is the project's cost on/off switch.

**The single most interesting engineering decision:** the synchronous front door is a
**Lambda Function URL with `AWS_IAM` auth**, *not* API Gateway. API Gateway caps
integration time at 30 seconds; our cold start is ~49s and even some warm LLM
syntheses exceed 30s, so API Gateway would time out. The Function URL honours the
function's own 120s timeout. Because the account blocks anonymous Function URLs,
requests are **SigV4-signed**.

---

## 2. The product and the goal of this exercise

**CiteMentor 2.0** already exists and is deployed as a Streamlit app from the
`master` branch. That deployment stays untouched. This work is *not* a new product —
it is taking the **existing RAG pipeline** through a full production lifecycle on AWS
as a deliberate learning exercise. Every file and decision is meant to be explainable
in an interview.

All of this lives on the **`aws-deploy`** branch, is additive and isolated, and does
not change how the Streamlit/`master` deployment works.

**Why the core was easy to lift out.** The pipeline logic is cleanly decoupled from
the UI:

- Business logic lives in `src/core/` (`graph.py`, `retriever.py`,
  `semantic_cache.py`, `guardrails.py`, `ledger.py`) — **no Streamlit imports**.
- The UI is isolated to `main.py` / `src/pages/`.
- The whole pipeline is a **single call**:
  `app_graph.invoke({"query": <str>, "timings": {}})`.

That single entry point is exactly what the API wraps. No business logic was
duplicated or rewritten for AWS.

**Inference mode.** The app supports a local Apple-Silicon MLX path and an OpenAI
path. Lambda runs **OpenAI mode** (`config/retrieval.yaml`), because MLX is Apple-
Silicon only and can't run on Graviton. In OpenAI mode the heavy local-inference
deps (`torch`, `sentence-transformers`, `mlx-*`, `deepeval`, `streamlit`) are never
imported at runtime, which is what makes the slim container possible.

**Pre-flight (done manually, before any of this).** AWS account created, MFA on root,
a dedicated IAM user `citementor-deploy` with `AdministratorAccess` + access keys,
AWS CLI configured locally, Docker installed. (A scoped least-privilege policy that
replaces `AdministratorAccess` is a documented post-pass.)

---

## 3. Architecture at a glance

```
                         ┌─────────────────────────────────────────────────────┐
                         │                      AWS  (ap-south-1)               │
                         │                                                       │
  developer / demo       │   ┌──────────────┐        ┌────────────────────────┐ │
  ───────────────────    │   │   Lambda     │        │  Secrets Manager        │ │
  scripts/demo_query.sh  │   │  Function    │        │  citementor/            │ │
        │                │   │  URL         │        │  openai_api_key         │ │
        │  HTTPS +       │   │ (AWS_IAM)    │        └───────────▲────────────┘ │
        │  SigV4 sign    │   └──────┬───────┘                    │ GetSecretValue│
        ▼                │          │ invokes                    │ (cold start)  │
  ┌───────────┐  POST    │   ┌──────▼─────────────────────────┐  │              │
  │ AWS creds │ ───────► │   │  Lambda function: citementor-api│──┘              │
  │ (IAM user)│  /query  │   │  • container image (ARM64)      │                 │
  └───────────┘          │   │  • 3008 MB, 120s timeout        │                 │
                         │   │  • exec role: logs + read secret│                 │
                         │   │                                 │                 │
                         │   │  ┌───────────────────────────┐  │   ┌──────────┐  │
                         │   │  │ FastAPI app (Mangum)      │  │   │CloudWatch│  │
                         │   │  │  GET /health              │  │──►│  Logs    │  │
                         │   │  │  POST /query → app_graph  │  │   └──────────┘  │
                         │   │  │  /var/task/storage (RO)   │  │                 │
                         │   │  │      └─copy→ /tmp/chroma_db│  │                 │
                         │   │  └───────────────────────────┘  │                 │
                         │   └─────────────────────────────────┘                 │
                         │          ▲ image pulled from                          │
                         │   ┌──────┴───────┐                                     │
                         │   │     ECR      │  citementor-api:latest (by digest)  │
                         │   │ (registry)   │                                     │
                         │   └──────────────┘                                     │
                         └─────────────────────────────────────────────────────┘
                                    ▲ docker buildx --platform linux/arm64 --push
                                    │
                         ┌──────────┴───────────┐
                         │  Terraform (local)   │  terraform apply / destroy
                         │  infra/*.tf          │  = the whole stack, on demand
                         └──────────────────────┘
```

**The wiring in one breath:** Terraform builds the ARM64 image and pushes it to ECR,
then creates a Lambda that runs that image behind a Function URL. A request is
SigV4-signed by the caller's AWS credentials, hits the Function URL, and invokes the
Lambda. On cold start the function reads the OpenAI key from Secrets Manager and
copies the bundled Chroma DB into `/tmp`; then the FastAPI app runs the RAG graph and
returns JSON. Logs flow to CloudWatch. `terraform destroy` removes all of it.

---

## 4. AWS services used — what they are and why

| Service | One-line definition | Why it's here |
|---|---|---|
| **IAM** (Identity & Access Management) | Who can do what in AWS. Users, roles, policies. | The `citementor-deploy` *user* deploys and invokes; the Lambda *role* (`citementor-lambda-role`) grants the function only logs + read-one-secret. Principle of least privilege. |
| **ECR** (Elastic Container Registry) | A private Docker image registry. | Lambda container images must be pulled from ECR. Stores `citementor-api`. |
| **Lambda** | Run code without managing servers; scales to zero; pay per ms. | Hosts the FastAPI app as a container. Scale-to-zero = ~\$0 idle, which is the whole point. |
| **Lambda Function URL** | A built-in HTTPS endpoint for a single function. | The synchronous front door. Chosen over API Gateway because it honours the 120s function timeout (no 30s cap). |
| **Secrets Manager** | Encrypted store for secrets with fine-grained IAM access + rotation. | Holds `OPENAI_API_KEY`. The key is fetched at cold start, never baked into the image or Terraform state. |
| **CloudWatch Logs** | Centralized logs (and metrics/alarms). | Lambda's stdout/stderr land here. Managed explicitly so retention is bounded and `destroy` cleans them up. |
| **STS / SigV4** | Temporary creds + the AWS request-signing algorithm. | `AWS_IAM` auth on the Function URL means every request is SigV4-signed; the caller's IAM identity authorizes it. |
| **Bedrock** *(optional mode)* | Managed access to foundation models (Amazon Nova, Anthropic, etc.) via one API, pay-per-token, no upfront cost. | Powers `inference_mode: bedrock` (Amazon Nova Lite+Pro + Titan V2) — an OpenAI-independent fallback billed on the AWS invoice. See the Bedrock add-on in §6. |
| *(considered, rejected)* **API Gateway** | Managed API front door (routing, auth, throttling). | Rejected: hard 30s integration timeout < our cold start. Left as a documented trade-off. |

**Region:** `ap-south-1` (Mumbai). **Account:** `505192030409`.

### Why serverless containers (and not a plain Lambda zip, or a VM, or ECS)?

- **Zip Lambda** has a 250 MB unzipped limit. Our deps + bundled Chroma corpus blow
  past that. **Container images** allow up to 10 GB, so the corpus ships *inside* the
  image. That's the deciding factor.
- **A VM (EC2)** or **always-on ECS/Fargate** would cost money 24/7 even when idle.
  This is a demo/portfolio service queried rarely — scale-to-zero is ideal.
- **Lambda** gives scale-to-zero *and* containers *and* a free HTTPS endpoint
  (Function URL). It is the cheapest correct option for "rarely queried, must be
  reproducible, must cost \$0 at rest."

---

## 5. The request lifecycle

There are three distinct paths, and their latencies are very different. Knowing this
cold is essential for demos and for explaining cold starts in interviews.

**1. Cold start (~49s).** The first request after the function has been idle/recycled:
1. Lambda pulls/initializes the container, starts the Python process.
2. `service/app.py` runs at import time: `load_openai_key_from_secrets()` calls
   Secrets Manager (one boto3 call), then `from src.core.graph import app_graph`
   loads catalog/config/prompts.
3. First `/query` lazily builds the retriever: copies the bundled `storage/chroma_db`
   → `/tmp/chroma_db` (read-only image FS forces this), loads Chroma + the BM25
   pickle, then runs the OpenAI embedding + retrieval + `gpt-5-mini` synthesis.
   The retriever/cache are **lazy**, so `/health` never pays this.

**2. Warm query (~12–25s).** The execution environment is reused: process is up,
`/tmp/chroma_db` already exists, retriever cached in memory. Cost is dominated by the
LLM calls. A measured warm turn from the live demo:

```
router      5.1s   (gpt-5-nano classifies the query)
retriever   3.5s   (OpenAI embed + Chroma dense search + BM25 + fusion)
synthesis  10.0s   (gpt-5-mini writes the grounded answer)
total      ~18s
```

**3. Semantic cache hit (~0.5s).** If a semantically-similar question was asked
before, the **semantic answer cache** (a Chroma collection keyed by query embedding,
threshold 0.88) returns the stored answer without routing, retrieval, or synthesis.

**Demo ritual:** run `scripts/demo_query.sh --warm` a few minutes before showing it,
so the audience sees warm (~18s) rather than cold (~49s) latency.

**Why not hide cold starts?** Provisioned concurrency or a scheduled warmer ping would
keep an instance hot — but both cost money at rest and defeat scale-to-zero. For a
rarely-queried demo, paying the cold start occasionally is the right trade.

---

## 6. The story, phase by phase

### Phase 0 — Extract the FastAPI service

**Goal:** put an HTTP API in front of the existing pipeline without touching the core.

Created `service/`:
- `service/app.py` — a deliberately thin FastAPI app:
  - `GET /health` → `{"status":"ok"}`. No retriever/model init, so it stays fast even
    on a cold Lambda (used as a warmup/liveness probe).
  - `POST /query` (`{query: str}`) → `app_graph.invoke(...)` → returns
    `{answer, sources, route, cache_hit, cache_similarity, timings}`.
  - **Mangum** wraps the app: `handler = Mangum(app)`. One app object runs locally
    under uvicorn *and* on Lambda — Mangum translates the Lambda event ↔ ASGI.
- `service/requirements.txt` — slim, **OpenAI-only**, versions pinned to the project's
  `uv.lock` for parity. Deliberately excludes `torch`, `sentence-transformers`,
  `mlx-*`, `deepeval`, `streamlit`.
- `service/.env.example` — documents `OPENAI_API_KEY` (and notes `HF_TOKEN` is not
  needed in OpenAI mode).

**Key risk handled:** the core uses **relative paths** (`config/`, `catalog.json`,
`storage/`). The process must run with the repo root as its working directory — later
enforced by the Docker `WORKDIR`.

**Verified:** uvicorn locally — `/health` ok; `/query` returns a grounded answer + 3
sources + timings; repeat → cache hit (~1.0 similarity); empty query → 422.

### Phase 1 — Containerize for ARM64 / Graviton

**Goal:** a Lambda-compatible container image, built and verified locally.

`service/Dockerfile` on `public.ecr.aws/lambda/python:3.12-arm64` (the AWS Lambda base
image, which bundles the Runtime Interface Client and a local Runtime Interface
Emulator). It copies `src/`, `service/`, `config/`, `storage/`, `catalog.json`,
`prompts.yaml` under `${LAMBDA_TASK_ROOT}` (= `/var/task`, the runtime working dir),
installs the slim requirements, and sets `CMD ["service.app.handler"]`.

Build from the **repo root** so the relative COPY paths resolve:

```bash
docker buildx build --platform linux/arm64 -f service/Dockerfile -t citementor-api:local .
```

**Why ARM64 end-to-end:** the dev machine is Apple Silicon (ARM), and Graviton Lambda
is ARM — building natively for ARM avoids slow x86→ARM QEMU emulation and is cheaper to
run. Architecture must match the Lambda's `--architectures arm64`.

**Verified via the Lambda Runtime Interface Emulator (RIE):** `/health` → 200;
`/query` → 200 with a grounded answer + 3 sources. Image was ~1.5 GB locally (base +
chromadb's onnxruntime + ~75 MB corpus). **Cold-start data:** module init ~265ms
(retriever is lazy); first `/query` ~26s. **Implication:** the Lambda timeout must be
≥30s — the default 3s would always fail — and memory wants ~2–3 GB.

### Phase 2 — Bundle the vector DB, relocate to `/tmp` on cold start

**The problem:** Lambda's container filesystem is **read-only except `/tmp`**. But
ChromaDB's `PersistentClient` opens its SQLite store **read-write** (locks, WAL/journal
files) — *and* the semantic answer cache **writes** into that same database. Opening the
bundled DB in place would fail on Lambda.

**The fix:** `src/core/storage.py::resolve_chroma_path()`:
- Copies the bundled `storage/chroma_db` → `/tmp/chroma_db` once per cold start and
  returns that writable path.
- **Memoised** at module level, so the retriever and the answer cache share *one*
  copy (they must — the cache collection lives in the same SQLite DB as the corpus).
- **Zero-config on Lambda:** triggers on the `AWS_LAMBDA_FUNCTION_NAME` env var that
  every Lambda runtime sets. Forceable locally via `CHROMA_COPY_TO_TMP=1`. Source/target
  overridable via `CHROMA_DIR` / `CHROMA_TMP_DIR`.
- Idempotent across warm invocations: only copies if `/tmp/chroma_db` is absent, so
  only the cold start pays the copy cost.

`retriever.py` and `semantic_cache.py` now call the resolver instead of a hardcoded
path. The BM25 pickle stays in place (read-only access is fine on Lambda).

**Parity proof:** `scripts/verify_chroma_tmp_parity.py` runs 5 eval queries through the
retriever in two subprocesses (bundled path vs `/tmp` copy) — identical chunk IDs and
order → PASS. Also verified inside the container with `--read-only --tmpfs /tmp` and
`AWS_LAMBDA_FUNCTION_NAME` set (a faithful Lambda simulation): `/var/task/storage`
read-only, `/tmp/chroma_db` populated, and a repeat query → `cache_hit: true` (proving
the cache writer succeeded against `/tmp`).

**Static-corpus trade-off (worth saying out loud in an interview):** baking the corpus
into the image and copying per cold start is great for a small, read-mostly library,
but (a) cache writes are **ephemeral** — lost when the execution environment recycles —
and (b) updating the corpus means **rebuilding the image**. Switch to a managed/networked
vector DB (Chroma server, pgvector, OpenSearch) when the corpus grows large, needs
incremental updates, or goes multi-tenant.

### Phase 3 — ECR + Lambda + Function URL (live on AWS)

**Goal:** the first real cloud deployment.

Pushed the ARM64 image to ECR and created the Lambda (Image, arm64, 3008 MB, 120s) with
the exec role. **This phase contains the project's headline war story.**

**War story 1 — the multi-arch manifest.** A default `docker buildx` push produces a
multi-arch **manifest list**, which Lambda rejects ("source image ... is not valid").
Fix: build with **`--provenance=false`** to emit a single-platform manifest. (This is
now encoded in both the manual workflow and Terraform.)

**War story 2 — API Gateway's 30s wall.** Started with an API Gateway HTTP API. Cold
starts (~49s) always returned 503, and even some *warm* `gpt-5-mini` syntheses exceed
30s. API Gateway has a **hard 30-second integration timeout** — it is simply the wrong
front door for long synchronous LLM calls. **Switched to a Lambda Function URL**, which
has no 30s cap and honours the function's own 120s timeout.

**War story 3 — the anonymous-URL block.** A Function URL with `AuthType NONE`
(anonymous/public) returned `403 AccessDeniedException` despite a correct resource
policy on a non-org account. An `AWS_IAM`-signed request to the same URL returned 200 —
proving the function and plumbing were fine and the block was **account-level** (public
Function URLs disabled). **Settled on `AuthType AWS_IAM`**: requests are SigV4-signed,
which eliminates *both* the timeout problem and the anonymous block. Added
`scripts/demo_query.sh` to sign requests so a demo is one command.

**Numbers:** cold start ~49s, warm ~12–25s, cache hit ~0.5s. The unused API Gateway was
later deleted in Phase 4.

### Phase 4 — Secrets Manager for the OpenAI key

**Goal:** stop setting the API key as a plaintext env var; fetch it securely at
runtime.

- Stored the key in Secrets Manager as `citementor/openai_api_key`.
- Granted `citementor-lambda-role` an **inline least-privilege** policy:
  `secretsmanager:GetSecretValue` on **only** that secret's ARN.
- New `service/secrets.py::load_openai_key_from_secrets()`, called at the top of
  `service/app.py`: if `OPENAI_API_KEY` is unset *and* `OPENAI_SECRET_NAME` is set,
  fetch the value via boto3 at cold start and populate `os.environ`, so the lazy core
  code is **unchanged**. No-op locally (key already in `.env`).
- `boto3` is provided by the Lambda base image and imported lazily, so it is **not** in
  `requirements.txt` and local runs never touch it.
- Reconfigured the live function to drop the plaintext `OPENAI_API_KEY` and set only
  `OPENAI_SECRET_NAME`. Verified a cold query (~47s) still returns a grounded answer —
  proving the key is sourced from Secrets Manager.

Also deleted the now-unused API Gateway from Phase 3.

**Security follow-up (recommended):** the key was at one point passed inline on a CLI,
so it may live in shell history/scrollback — **rotate the OpenAI key**, `put-secret-value`
the new one (no redeploy needed), and revoke the old.

### Phase 5 — Terraform: the whole stack, on demand

**Goal:** replace every manual `aws ...` step with declarative infrastructure-as-code so
the stack can be created from nothing and destroyed to \$0, repeatably. This is the
highest-leverage phase — it *is* the cost on/off switch.

Everything from Phases 1–4 is now defined under `infra/` (see
[§8](#8-the-terraform-stack-resource-by-resource)). The manually-created resources were
deleted, then `terraform apply` recreated the **identical** stack from scratch.

**Three things that make this Terraform interesting:**

1. **It builds the image itself.** A `null_resource` with a `local-exec` provisioner runs
   the same `docker buildx --platform linux/arm64 --provenance=false --push` command, so
   `apply` truly creates everything from nothing — no separate manual build step. A
   `data "aws_ecr_image"` then reads the pushed image's **digest**, and the Lambda is
   pinned to that immutable digest (not the mutable `:latest` tag) so every rebuild rolls
   the function forward.

2. **The secret value never enters Terraform state.** Terraform manages only the empty
   secret *container*. A second `null_resource` pushes the value with a shell command that
   reads `$OPENAI_API_KEY` **from the environment at apply time** (`$${OPENAI_API_KEY}` in
   the heredoc renders to a literal shell variable, not a Terraform interpolation). So the
   plaintext flows env → AWS CLI only; the local state file never sees it.

3. **It's built to be torn down cleanly.** `aws_ecr_repository` has `force_delete = true`
   (destroy works even with images present) and the secret has
   `recovery_window_in_days = 0` (immediate delete, so a later `apply` doesn't collide with
   a name "scheduled for deletion"). The CloudWatch log group is managed explicitly so
   `destroy` removes the logs too.

**Verified the full lifecycle live:** `terraform apply` (10 resources) → `demo_query.sh
--health/--warm/"<question>"` returned a grounded, cited answer → `terraform destroy` (10
resources) → confirmed Lambda, ECR, secret, role, and log group all **not found** → AWS
spend back to ~\$0.

### Add-on — Bedrock mode (an OpenAI-independent fallback)

**The problem it solves.** The stack runs `inference_mode: openai`, which needs a funded
OpenAI account. When that balance hits \$0 you must top it up (min \$5) just to run a
one-off demo. **AWS Bedrock has no upfront cost** — usage is billed on the same monthly
AWS invoice, and at this project's rare-demo volume it's negligible. So we added a third
`inference_mode: "bedrock"` that runs the **entire pipeline on Bedrock with zero OpenAI
dependency**.

**Models (in `ap-south-1`), tiered like the openai path (cheap router, premium synthesis):**
- **Router:** **Amazon Nova Lite** (`apac.amazon.nova-lite-v1:0`) — cheap, fast, reliable at
  the structured-output classification the router needs. ~\$0.06/\$0.24 per 1M tokens.
- **Synthesis:** **Amazon Nova Pro** (`apac.amazon.nova-pro-v1:0`) — the higher-quality tier
  where the answer actually gets written. ~\$0.80/\$3.20 per 1M tokens.
- **Embeddings:** **Amazon Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).

> **Why Nova and not Claude? (a real war story worth telling.)** We first wired Bedrock
> mode to **Anthropic Claude 3 Haiku** — it passed every *local* test. But on the *cloud*
> Lambda it returned `AccessDeniedException: ... not authorized to perform the required AWS
> Marketplace actions (aws-marketplace:Subscribe) to enable access to this model`. The
> catch: **Anthropic models on Bedrock are third-party AWS Marketplace products**, and this
> account's Marketplace *agreement* for Claude kept being created and **expiring in the same
> instant** (start date == end date) — a broken subscription loop that even the admin user
> couldn't invoke through. **Amazon Nova models are first-party**, so they need *no*
> Marketplace subscription and are auto-enabled on first invoke. Switching router+synthesis
> to Nova made the cloud path work immediately — and Nova is cheaper too. *Lesson: on
> Bedrock, first-party (Amazon) vs. Marketplace (third-party) models have very different
> enablement paths; the third-party subscription is an account-level dependency your IAM
> can't satisfy.* The Nova models are also **cross-region inference-profile** models (the
> `apac.` prefix), which is why the IAM (below) grants both the profile ARN and the
> underlying regional model ARNs.

**The one hard constraint — embedding spaces don't mix.** ChromaDB binds an embedding
model to a collection at creation time. The corpus lives in `citementor_library_openai`,
embedded with OpenAI `text-embedding-3-small` (1536-dim). A Titan query vector (different
model, different space) **cannot** search that collection — it returns garbage. So Bedrock
mode needs its **own** collection, `citementor_library_bedrock`, embedded with Titan.

**Building that collection is cheap.** The enriched chunk texts (with their contextual
summaries) already live in the local collection, so we don't re-run any LLM
summarization — we just **re-embed** the existing 2 800 chunks with Titan. `ingestion.py`
already had `rebuild_openai_from_existing()`; it was generalized to
`rebuild_from_existing(target_profile, ...)`, and `--profile=bedrock` routes through it.
Cost: 2 800 Titan embed calls ≈ a fraction of a cent, **no OpenAI calls**.

**How the code branches (mirrors the existing openai/local pattern):**
- `config/retrieval.yaml` — a `bedrock:` block (region + model ids) and
  `vector_stores.bedrock_collection`.
- `src/core/graph.py` — a `get_bedrock_llm()` factory (`langchain_aws.ChatBedrockConverse`)
  plus `bedrock` branches in the router (structured output via the default tool method,
  not OpenAI's `json_schema`) and synthesis (streaming; a small `_content_to_text()`
  normalizes Bedrock's content-block chunks to plain text).
- `src/core/retriever.py` / `semantic_cache.py` — a `bedrock` branch selecting the Titan
  `AmazonBedrockEmbeddingFunction` and the bedrock collection / cache
  (`citementor_answer_cache_bedrock`). Bedrock takes the same no-cross-encoder path as
  openai (the slim image has no sentence-transformers).
- `service/requirements.txt` — adds `langchain-aws==1.4.6` (compatible with the pinned
  `langchain-core==1.3.2`; pulls boto3 transitively). Imported lazily, inert in openai mode.
- `infra/iam.tf` — a least-privilege inline policy granting `bedrock:InvokeModel[WithResponseStream]`
  on **only**: the Titan model ARN, the two Nova **inference-profile** ARNs, and the Nova
  underlying foundation-model ARNs (region-wildcard, since a cross-region profile may route
  to any of its APAC regions — still scoped to the specific model). Harmless in openai mode.
- The guardrails are pure regex (no LLM), so nothing else touches a provider.

**Two operational notes:**
1. **Model enablement.** Amazon Nova + Titan are first-party and auto-enable on first invoke —
   no manual step, no Marketplace subscription. (Third-party models like Anthropic Claude
   would require a one-time Marketplace subscription completed by a marketplace-capable
   principal — the enablement failure that made us switch to Nova; see the war story above.)
2. **Switching modes = config edit + rebuild.** Set `inference_mode: bedrock` in
   `config/retrieval.yaml`, then `terraform apply` (the config change bumps the build hash,
   so the image is rebuilt+pushed and the Lambda rolls forward). Revert + apply to go back.
   No env-var toggle — one source of truth, at the cost of a ~3-minute rebuild per switch.

**Verified live end-to-end:** `terraform apply` in bedrock mode → `demo_query.sh` returned a
grounded, cited answer routed entirely through Nova (router `finance` ~0.6s, Titan retrieval
~2.2s, Nova Pro synthesis ~1.3s — **~4s warm, faster than the OpenAI path**), repeat →
`cache_hit: true` (sim 1.0) → `terraform destroy` → \$0. The whole run used an *invalid*
OpenAI key, proving zero OpenAI dependency.

**Credentials.** Bedrock mode uses no API key: locally the boto3 default chain reads your
`aws configure` profile; on Lambda it's the execution role. This is *why* it's the perfect
"OpenAI ran out" escape hatch — nothing to fund, nothing to inject.

### Phase 6 — CI/CD: deploy & tear down from the GitHub Actions tab

**Goal.** Turn the whole apply/destroy lifecycle into two buttons in the **Actions** tab —
so a demo is: click **Deploy** (pick `bedrock`/`openai`), wait, demo, click **Teardown** →
back to \$0 — with **no long-lived AWS keys** anywhere.

**The pieces:**
- **GitHub OIDC, not access keys.** Each workflow run mints a short-lived OIDC token; AWS
  trusts GitHub's OIDC provider and lets the run assume an IAM role for ~an hour. Nothing
  secret is stored in GitHub except the (unused-in-bedrock) OpenAI key. The role
  (`citementor-github-actions`) is trusted **only** by this repo's `aws-deploy` branch
  (`sub = repo:<owner>/<repo>:ref:refs/heads/aws-deploy`) and carries a **scoped** policy —
  exactly the ECR/Lambda/IAM/Secrets/Logs/S3 actions Terraform touches.
- **Remote state in S3.** The Deploy run and a later Teardown run are separate ephemeral
  machines, so state can't be local — it lives in an S3 bucket with **native S3 locking**
  (`use_lockfile`, no DynamoDB). See `infra/backend.tf`.
- **Native ARM runners.** The repo is public, so `runs-on: ubuntu-24.04-arm` is free — the
  arm64 image builds natively (no slow QEMU emulation).
- **Mode picked at click-time.** `deploy.yml` has a `workflow_dispatch` **choice input**
  (`inference_mode`); a step rewrites `config/retrieval.yaml` in the checkout before build,
  so no commit/push is needed to switch engines. (The build hash covers the config, so a
  different mode triggers a fresh image automatically.)

**One-time bootstrap (`infra/bootstrap.sh`).** The state bucket, OIDC provider, and CI role
must exist *before* Terraform can run in CI (chicken-and-egg), so they're created once,
outside the main Terraform, by an idempotent script. They're cheap standing resources (the
S3 state bucket ≈ \$0/mo; OIDC + IAM are free) that **intentionally survive teardown** — the
state must outlive the app.

**What the automation did — and the exact manual equivalent (for learning):**

| Step | What the script/CLI did | If you did it by hand |
|---|---|---|
| State bucket | `aws s3api create-bucket` + versioning + AES256 + block-public-access | S3 console → Create bucket → enable Versioning, Default encryption, Block all public access |
| OIDC provider | `aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com` | IAM console → Identity providers → Add provider → OpenID Connect → that URL, audience `sts.amazonaws.com` |
| CI role | `aws iam create-role` with a trust policy scoped to `repo:…:ref:refs/heads/aws-deploy`, `+ put-role-policy` (scoped perms) | IAM → Roles → Create role → Web identity → GitHub provider → add the repo/branch condition → attach the policy |
| State migration | `terraform init -migrate-state` (moves the empty local state into S3) | same command locally after adding `backend.tf` |
| Repo config | `gh variable set AWS_ROLE_ARN/AWS_REGION` and `gh secret set OPENAI_API_KEY` | Repo → Settings → Secrets and variables → Actions → add the two **Variables** and the one **Secret** |
| Default branch | `gh api -X PATCH … -f default_branch=aws-deploy` | Repo → Settings → General → Default branch → switch to `aws-deploy` |

**Why `aws-deploy` had to become the default branch.** GitHub only lets you *dispatch* a
`workflow_dispatch` workflow that exists on the **default** branch. The workflows (and all of
`infra/`) live on `aws-deploy`, so it became the default. It's reversible and doesn't affect
the Streamlit deploy (Streamlit Cloud is pinned to its own configured branch).

**Three gotchas hit while wiring it up (all now handled):**
1. **"workflow not found on the default branch"** — the default-branch rule above.
2. **`CreateLogGroup` 409 (ResourceAlreadyExists)** — Lambda can recreate its
   `/aws/lambda/<fn>` log group from *late-arriving log events after a destroy*, leaving a
   "zombie" not in Terraform state. `deploy.yml` deletes any orphan log group before apply.
3. **`AccessDenied` on `logs:ListTagsForResource`** — the AWS provider reads a log group's
   tags on the ARN **without** the `:*` suffix; the scoped CI policy needed that action and
   *both* ARN forms. *Lesson: scoping IAM to exact ARNs is real least-privilege but you
   discover the provider's precise action/ARN calls by iterating.*

**Verified live:** Deploy workflow (bedrock) → 1m10s → live Function URL → `demo_query.sh`
returned a grounded answer via Nova → Teardown workflow → app resources gone, shared S3 state
back to 0 resources, standing bucket + OIDC role intact.

---

## 7. Code walkthrough — how the app is wired

```
service/
  app.py              FastAPI app + Mangum handler (the only HTTP layer)
  secrets.py          cold-start fetch of OPENAI_API_KEY from Secrets Manager
  requirements.txt    slim, OpenAI-only, pinned deps
  Dockerfile          ARM64 Lambda image
  .env.example        documents env vars
src/core/
  graph.py            the compiled LangGraph pipeline (app_graph) — unchanged by AWS
  retriever.py        hybrid dense+BM25 retriever — now resolves Chroma via storage.py
  semantic_cache.py   semantic answer cache — now resolves Chroma via storage.py
  storage.py          resolve_chroma_path(): copy bundled DB → /tmp on Lambda
scripts/
  demo_query.sh       SigV4-signed client for the Function URL
  verify_chroma_tmp_parity.py   Phase 2 parity test
infra/                Terraform (see §8)
```

**`service/app.py` — the HTTP layer (thin by design).**
- Line 1 of work at import time is `load_openai_key_from_secrets()` — *before* anything
  reads the key.
- Then `from src.core.graph import app_graph` runs graph.py's module-level setup
  (catalog/config/prompts). The retriever and cache stay **lazy** (built on first
  `/query`), so `/health` is cheap on a cold Lambda.
- `POST /query` is a 1:1 wrapper: `app_graph.invoke({"query": q, "timings": {}})` →
  map the resulting `AgentState` dict into `QueryResponse`. Every field is read
  defensively (`.get(...)`) because not every graph path fills every key (a cache hit
  or greeting skips routing/retrieval).
- `handler = Mangum(app)` is the Lambda entry point named by the container `CMD`.

**`service/secrets.py` — secure key loading.**
- If `OPENAI_API_KEY` is already set (local/dev), return immediately — no AWS touched.
- Else, if `OPENAI_SECRET_NAME` is set, lazily import boto3 and
  `get_secret_value(...)`, accept either a raw string or a `{"OPENAI_API_KEY": ...}`
  JSON blob, and set `os.environ["OPENAI_API_KEY"]`. The lazy import is why boto3
  isn't a dependency locally.

**`src/core/storage.py` — the read-only-filesystem fix.**
- `resolve_chroma_path()` returns the bundled path locally and the `/tmp` copy on
  Lambda, memoised so the copy runs once and every client shares it. (See Phase 2.)

**`scripts/demo_query.sh` — the signed client.**
- Discovers the Function URL via `aws lambda get-function-url-config`, then uses
  `curl --aws-sigv4 "aws:amz:<region>:lambda" --user "<AK>:<SK>"` to SigV4-sign the
  request (with `x-amz-security-token` support for temporary creds). `--warm`, `--health`,
  or a question argument.

---

## 8. The Terraform stack, resource by resource

Files under `infra/` (one concern per file for readability):

| File | Contents |
|---|---|
| `versions.tf` | Terraform ≥ 1.6, AWS provider `~> 5.60`, null provider. Local (gitignored) state. |
| `providers.tf` | AWS provider (region var, default tags), `aws_caller_identity`/`aws_region` data sources. |
| `variables.tf` | `region`, `project_name`, `lambda_memory_mb` (3008), `lambda_timeout_s` (120), `log_retention_days` (14), `image_tag` (latest). |
| `locals.tf` | Resource name, repo URL, and `source_hash` (md5 over code/config files → triggers rebuilds). |
| `ecr.tf` | ECR repo (`force_delete`), `null_resource.image_build_push` (buildx + push), `data.aws_ecr_image` (digest). |
| `secrets.tf` | Secret container (`recovery_window_in_days = 0`) + `null_resource.openai_secret_value` (push value from env). |
| `iam.tf` | Exec role + assume policy, `AWSLambdaBasicExecutionRole` attachment, inline read-one-secret policy. |
| `lambda.tf` | Log group, `aws_lambda_function` (image by digest, arm64, mem/timeout, `OPENAI_SECRET_NAME` env), `aws_lambda_function_url` (`AWS_IAM`). |
| `outputs.tf` | `function_url`, `function_name`, `ecr_repository_url`, `image_digest`, `secret_name`, `demo_hint`. |
| `terraform.tfvars.example` | Optional overrides; documents that the key comes from the env, not here. |

**The dependency chain Terraform resolves automatically:**

```
aws_ecr_repository.api
        │
        ▼  (triggers on source_hash)
null_resource.image_build_push  ──build+push──►  ECR
        │
        ▼  (depends_on)
data.aws_ecr_image.api  ──reads digest──►  image_uri = repo@sha256:...
        │
aws_secretsmanager_secret.openai ──► null_resource.openai_secret_value (puts value)
        │                                    │
        └──────────────► aws_iam_role.lambda + inline GetSecretValue policy
                                             │
                                             ▼
                          aws_lambda_function.api (uses image digest, role, secret name)
                                             │
                                             ▼
                          aws_lambda_function_url.api  ──►  https://...lambda-url...aws/
```

**Why pin the digest, not the tag?** Lambda caches by `image_uri`. If you reference
`:latest`, pushing a new image with the same tag would *not* change `image_uri`, so
Terraform would see no change and the function would keep running the old image. Pinning
`repo@sha256:<digest>` means each rebuild changes `image_uri` → Terraform updates the
function. Correctness over convenience.

---

## 9. Operations runbook

### Preferred: from GitHub Actions (Phase 6)

No laptop, Docker, or AWS keys needed:
1. **Actions → Deploy → Run workflow** → choose `bedrock` or `openai`. The run assumes the
   AWS role via OIDC, builds the image on an ARM runner, `terraform apply`s, and prints the
   Function URL in its summary.
2. Demo (warm first): `scripts/demo_query.sh --warm` then `scripts/demo_query.sh "<question>"`.
3. **Actions → Teardown → Run workflow** → `terraform destroy` → **\$0**.

*(First-time-only setup was `infra/bootstrap.sh` + a few `gh` commands — see §6.)*

### Alternative: locally

**Prerequisites:** Terraform ≥ 1.6, Docker running (with buildx), AWS CLI configured,
and the OpenAI key exported in the shell. (Local runs use the same S3 remote state as CI.)

```bash
# 0. Provide the OpenAI key to the environment (never stored in state)
set -a; source .env; set +a            # or: export OPENAI_API_KEY=sk-...

# 1. Create everything on demand
cd infra
terraform init        # first run only
terraform apply        # builds ARM64 image, pushes to ECR, creates the stack (~few min)

# 2. Run the demo (from repo root)
cd ..
scripts/demo_query.sh --health                       # {"status":"ok"}
scripts/demo_query.sh --warm                          # pays the ~40-50s cold start once
scripts/demo_query.sh "What does the Gita say about duty?"

# 3. Tear everything down → $0
cd infra
terraform destroy

# 4. (optional) confirm nothing is left
aws lambda get-function --function-name citementor-api --region ap-south-1   # → not found
```

**Updating the corpus.** The build hash covers code/config, not `storage/`. After
changing the corpus, force a rebuild:
`terraform taint null_resource.image_build_push && terraform apply`.

**Rotating the OpenAI key.** No Terraform change needed —
`aws secretsmanager put-secret-value --secret-id citementor/openai_api_key --secret-string "$NEW_KEY"`.
The next cold start picks it up.

---

## 10. Cost model

The design target is **\$0 at rest**.

| Resource | Idle cost | Per-use cost |
|---|---|---|
| Lambda + Function URL | \$0 (scale-to-zero) | ~\$0.0000... per ms × 3 GB; pennies even for a heavy demo day |
| ECR image (~340 MB stored) | ~\$0.03–0.15 / month | — |
| Secrets Manager (1 secret) | ~\$0.40 / month + \$0.05 / 10k API calls | per cold-start fetch (negligible) |
| CloudWatch Logs | ~\$0 at this volume (14-day retention) | log ingestion (negligible) |
| OpenAI API *(openai mode)* | — | per query (embeddings + `gpt-5-nano` router + `gpt-5-mini` synthesis) |
| Bedrock *(bedrock mode)* | \$0 (no upfront, no reservation) | per query (Titan V2 embeddings + Nova Lite router + Nova Pro synthesis); on the monthly AWS invoice |

**The two standing charges while applied are ECR and Secrets Manager — together well
under \$1/month.** `terraform destroy` removes even those, taking AWS spend to ~\$0. The
only ongoing variable cost is the LLM provider, per-query. **Bedrock mode's whole appeal
is that its per-query cost has _no upfront minimum_** — unlike OpenAI's \$5 top-up — so a
one-off demo when the OpenAI balance is empty costs literal cents on the AWS bill.

---

## 11. Security model

- **No long-lived secrets in the image or in Terraform state.** The OpenAI key lives in
  Secrets Manager; the image has only the secret's *name*; the state file never sees the
  value (pushed from the shell env at apply time).
- **Least-privilege execution role.** `citementor-lambda-role` can do exactly two things:
  write CloudWatch logs (`AWSLambdaBasicExecutionRole`) and `GetSecretValue` on the *one*
  OpenAI secret ARN. It cannot read other secrets, touch S3, etc.
- **Authenticated endpoint.** The Function URL uses `AWS_IAM`, so only callers with valid
  signed AWS credentials (and permission to invoke) can reach `/query`. There is no
  anonymous access — partly by choice, partly because the account blocks public Function
  URLs.
- **Known follow-ups:** (1) rotate the OpenAI key that briefly touched shell history
  (Phase 4 note). (2) Replace the deployer's `AdministratorAccess` with a scoped policy
  (documented post-pass). (3) For a public demo without AWS creds, you'd front the Lambda
  with something that adds its own auth (e.g. API Gateway + a usage key, accepting its
  timeout limits, or a Cloudflare Worker proxy).

---

## 12. Challenges and lessons (consolidated)

The interview-ready highlight reel:

1. **API Gateway's 30s timeout vs. LLM latency.** The classic mismatch: managed API
   front doors assume sub-second backends; synchronous LLM synthesis is tens of seconds.
   → Lambda Function URL (120s) instead. *Lesson: match the front door's timeout budget to
   the real backend latency before picking it.*
2. **Anonymous Function URLs blocked at the account level.** Diagnosed by proving an
   IAM-signed request succeeded where an anonymous one 403'd. → `AWS_IAM` + SigV4.
   *Lesson: isolate "is it my config or the platform?" with a controlled A/B request.*
3. **Lambda rejects multi-arch manifests.** `--provenance=false` to get a single-platform
   image. *Lesson: container "images" can secretly be manifest lists; platforms differ in
   what they accept.*
4. **Read-only container filesystem vs. a read-write embedded DB.** Chroma + the answer
   cache both write SQLite; only `/tmp` is writable. → copy-to-`/tmp` on cold start,
   memoised and shared. *Lesson: know exactly which paths your dependencies write to.*
5. **Cold starts are real and large (~49s).** Driven by container init + lazy retriever
   build + LLM calls. → generous timeout, lazy `/health`, a `--warm` demo ritual.
   *Lesson: separate liveness from readiness; warm before you demo.*
6. **Keeping secrets out of state.** IaC tools love to slurp secrets into state. → manage
   only the container in Terraform, push the value from the env via a provisioner.
   *Lesson: the state file is a secret-leak vector; treat it like one.*
7. **Designing for teardown, not just standup.** `force_delete`, `recovery_window = 0`,
   explicit log group. → `destroy` actually reaches \$0 and `apply` doesn't collide on
   re-create. *Lesson: a reproducible stack must destroy as cleanly as it creates.*
8. **Slimming the image by understanding the code paths.** Because OpenAI mode never
   imports torch/MLX/etc., the image dropped from ~2 GB to a few hundred MB.
   *Lesson: dependency hygiene is a deployment concern, not just a dev-time one.*

---

## 13. Glossary

- **RAG (Retrieval-Augmented Generation):** retrieve relevant text, then have an LLM
  answer grounded in it (with citations), instead of answering from parametric memory.
- **Hybrid retrieval:** combine dense (embedding/vector) search with sparse (BM25
  keyword) search and fuse the results — catches both semantic and exact-term matches.
- **BM25:** a classic keyword-ranking function (TF-IDF family). Here, a prebuilt pickle.
- **Embedding:** a vector representation of text; similar meanings → nearby vectors.
- **ChromaDB:** an embedded vector database (SQLite-backed here) storing the corpus
  embeddings and the semantic answer cache.
- **Semantic answer cache:** cache keyed by query *embedding* similarity (threshold 0.88),
  so paraphrases hit the cache, not just exact repeats.
- **LangGraph:** a library for building LLM pipelines as a graph of nodes
  (router → retriever → synthesis → guards). `app_graph` is the compiled graph.
- **Mangum:** an adapter that lets an ASGI app (FastAPI) run on Lambda by translating the
  Lambda event ↔ ASGI interface.
- **Graviton / ARM64:** AWS's ARM CPUs; cheaper per compute. We build the image for ARM64.
- **Cold start:** the latency when Lambda must initialize a fresh execution environment
  before handling a request.
- **Function URL:** a dedicated HTTPS endpoint for one Lambda function.
- **SigV4:** AWS's request-signing algorithm; proves the caller's identity for `AWS_IAM`
  auth.
- **ECR:** AWS's private Docker registry.
- **IaC (Infrastructure as Code):** declaring infrastructure in files (Terraform) so it's
  versioned, reviewable, and reproducible.
- **`null_resource` / `local-exec`:** Terraform escape hatch to run a local shell command
  as part of apply (used here to build/push the image and push the secret value).
- **OIDC (OpenID Connect) for CI:** GitHub Actions presents a short-lived signed token that
  AWS trusts (via an IAM identity provider) to grant temporary role credentials — so no
  long-lived AWS access keys are stored in GitHub.
- **Remote state backend:** Terraform state kept in shared storage (here an S3 bucket with
  native locking) instead of on one machine, so separate CI runs (and your laptop) operate
  on one authoritative state.
- **`workflow_dispatch`:** a GitHub Actions trigger that adds a manual "Run workflow" button
  (with optional typed inputs, e.g. our `inference_mode` dropdown). Only dispatchable from
  the repo's default branch.

---

## 14. What's next (Phases 7–9)

This document grows with the project. Planned additions:

- **Phase 7 — Observability + guardrails on cost.** Langfuse tracing in the service;
  CloudWatch alarms (error rate, p95 latency); an AWS Budget (~\$5/mo) alert.
- **Phase 8 — Streamlit integration.** An env toggle so the Streamlit app can call the
  deployed API, with graceful degradation when the stack is torn down.
- **Phase 9 — Final hardening + this doc made exhaustive.** Scoped least-privilege IAM to
  replace `AdministratorAccess`; complete the architecture/runbook/cost/observability
  reference.

### The demo flow — run everything from GitHub (Phase 6 ✅, Phase 8 pending)

CI/CD is live, so giving a demo is already down to:

1. GitHub → **Actions** → **Deploy** → *Run workflow* → pick `bedrock` or `openai` → it
   assumes the AWS role via OIDC and runs `terraform apply` (the run summary prints the
   Function URL).
2. Warm up and demo. **Today** that's `scripts/demo_query.sh` (SigV4-signed). **After
   Phase 8** it'll be the Streamlit app calling the deployed API.
3. GitHub → **Actions** → **Teardown** → *Run workflow* → `terraform destroy` → **\$0**.

**Still to wire for Phase 8 — Streamlit → Function URL auth.** The URL is `AWS_IAM`, so the
Streamlit app must SigV4-sign its requests using AWS credentials stored as Streamlit secrets
(or we add a separate public-auth front door). It's why the endpoint isn't just a plain
public URL.

> _Last updated: 2026-07-07 — covers Phases 0–6 + the Bedrock inference mode._
