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
