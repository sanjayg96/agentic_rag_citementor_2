# CiteMentor AWS infra (Terraform)

Infrastructure-as-code for the whole serverless stack: ECR repo + image
build/push, Lambda (container, ARM64) + Function URL, IAM exec role, Secrets
Manager secrets (OpenAI key, optional Langfuse tracing keys), CloudWatch log
group + alarms (error rate, p95 latency) with an SNS email topic. One `apply`
creates everything from nothing; one `destroy` returns AWS spend to ~$0.

See the full narrative + theory in [`../docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).

## Prerequisites

- Terraform ≥ 1.6, Docker (running, with buildx), AWS CLI configured
  (`citementor-deploy` user).
- The OpenAI key exported in your shell — Terraform reads it from the
  environment when pushing the secret value, so it never lands in state:

  ```bash
  set -a; source ../.env; set +a      # or: export OPENAI_API_KEY=sk-...
  ```

## Create everything (on demand)

```bash
cd infra
terraform init        # first time only
terraform apply       # builds the ARM64 image, pushes to ECR, creates the stack
```

`apply` prints the `function_url`. The first build takes a few minutes (Docker
buildx + push); subsequent applies only rebuild when source files change.

## Run the demo

```bash
cd ..
scripts/demo_query.sh --health                       # {"status":"ok"}
scripts/demo_query.sh --warm                          # pays the ~40-50s cold start
scripts/demo_query.sh "What does the Gita say about duty?"
```

`demo_query.sh` discovers the Function URL and SigV4-signs each request (the URL
uses `AWS_IAM` auth).

## Tear everything down (→ $0)

```bash
cd infra
terraform destroy
```

Removes the Lambda, Function URL, ECR repo (incl. images, via `force_delete`),
secret (immediate, `recovery_window_in_days = 0`), role, and log group. Verify
with `aws lambda get-function --function-name citementor-api` → not found.

## CI/CD

Preferred path is the **Actions** tab — **Deploy** / **Teardown** workflows (GitHub
OIDC, no keys). One-time setup lives in [`bootstrap.sh`](bootstrap.sh) (state bucket +
OIDC provider + scoped role). See [`../docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) §6.

## Streamlit UI on the deployed Lambda (Phase 8)

[`streamlit_user.sh`](streamlit_user.sh) creates a dedicated least-privilege IAM user
(`citementor-streamlit`) that can only discover + invoke the Function URL, and prints an
`[aws]` secrets block to paste into a Streamlit Community Cloud app (branch `aws-deploy`,
main file `src/app.py`). Like `bootstrap.sh`, it lives outside the app Terraform so its
access key survives `destroy` → the Streamlit secret never needs re-editing across
deploy/teardown cycles. See [`../docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) §15.

```bash
bash infra/streamlit_user.sh            # create user + policy, mint a key
bash infra/streamlit_user.sh --rotate   # replace the key
```

## Notes

- **State is remote** in S3 with native locking (`backend.tf`) so CI and local share it.
  Created once by `bootstrap.sh`; the bucket survives `destroy` (state must outlive the app).
- **Corpus changes** under `storage/` don't auto-trigger an image rebuild (the
  build hash covers code/config only). To ship a new corpus:
  `terraform taint null_resource.image_build_push && terraform apply`.
- **Secret value** is set by a `null_resource` reading `$OPENAI_API_KEY` at apply
  time; rotating the key needs no Terraform change — just
  `aws secretsmanager put-secret-value`.
- **Observability (Phase 7).** CloudWatch alarms (`monitoring.tf`) notify `var.alert_email`
  via SNS — AWS emails a one-time confirmation link on first apply that must be clicked.
  LLM tracing via Langfuse is optional: export `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
  before apply to enable it, or leave unset to run untraced. A standing `$5/mo` AWS Budget
  alert lives outside this Terraform, in `bootstrap.sh` (survives `destroy` on purpose).
