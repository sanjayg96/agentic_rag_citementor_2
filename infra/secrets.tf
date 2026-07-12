# --- Secrets Manager: OpenAI API key ---
#
# Terraform manages only the empty secret *container*. The plaintext key is never
# a Terraform variable or resource attribute, so it never lands in the (local)
# state file. The value is pushed by the null_resource below, whose shell command
# reads it from the OPENAI_API_KEY environment variable at apply time.

resource "aws_secretsmanager_secret" "openai" {
  name        = "${var.project_name}/openai_api_key"
  description = "OpenAI API key for the CiteMentor Lambda. Read-only by the exec role."

  # 0 = delete immediately on destroy (no 7-30 day recovery window). Essential for
  # the destroy -> apply cycle: otherwise the name stays reserved and re-apply
  # fails with "already scheduled for deletion".
  recovery_window_in_days = 0
}

resource "null_resource" "openai_secret_value" {
  # Re-push the value whenever a brand-new secret is created.
  triggers = {
    secret_id = aws_secretsmanager_secret.openai.id
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-c"]
    # $${OPENAI_API_KEY} -> rendered as ${OPENAI_API_KEY}, expanded by bash at run
    # time. The key therefore flows env -> aws CLI only; Terraform never sees it.
    command = <<-EOT
      set -euo pipefail
      if [ -z "$${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: export OPENAI_API_KEY before 'terraform apply'." >&2
        echo "       (Terraform never stores the key in state; it is read from the env here.)" >&2
        exit 1
      fi
      aws secretsmanager put-secret-value --region ${var.region} \
        --secret-id ${aws_secretsmanager_secret.openai.id} \
        --secret-string "$${OPENAI_API_KEY}" >/dev/null
      echo ">> OpenAI key written to ${aws_secretsmanager_secret.openai.name}"
    EOT
  }
}

# --- Secrets Manager: Langfuse tracing keys (Phase 7, optional) ---
#
# Same pattern as the OpenAI secret above, with one difference: tracing is
# optional, so a missing LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY does not fail
# the apply. Instead the secret is left holding an empty JSON object, and
# service/secrets.py treats that as "tracing disabled" — the app runs
# untraced rather than erroring.

resource "aws_secretsmanager_secret" "langfuse" {
  name        = "${var.project_name}/langfuse_keys"
  description = "Langfuse public/secret API keys for LLM tracing. Read-only by the exec role. Optional — app runs untraced if empty."

  recovery_window_in_days = 0
}

resource "null_resource" "langfuse_secret_value" {
  triggers = {
    secret_id = aws_secretsmanager_secret.langfuse.id
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -euo pipefail
      if [ -z "$${LANGFUSE_PUBLIC_KEY:-}" ] || [ -z "$${LANGFUSE_SECRET_KEY:-}" ]; then
        echo ">> LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - leaving ${aws_secretsmanager_secret.langfuse.name} empty (app runs untraced)." >&2
        aws secretsmanager put-secret-value --region ${var.region} \
          --secret-id ${aws_secretsmanager_secret.langfuse.id} \
          --secret-string '{}' >/dev/null
        exit 0
      fi
      SECRET_JSON=$(python3 -c 'import json,os,sys; json.dump({"public_key": os.environ["LANGFUSE_PUBLIC_KEY"], "secret_key": os.environ["LANGFUSE_SECRET_KEY"]}, sys.stdout)')
      aws secretsmanager put-secret-value --region ${var.region} \
        --secret-id ${aws_secretsmanager_secret.langfuse.id} \
        --secret-string "$${SECRET_JSON}" >/dev/null
      echo ">> Langfuse keys written to ${aws_secretsmanager_secret.langfuse.name}"
    EOT
  }
}
