# --- CloudWatch log group ---
# Created explicitly (instead of letting Lambda auto-create it) so retention is
# bounded and `terraform destroy` cleans the logs up too. Name must match the
# convention Lambda writes to: /aws/lambda/<function-name>.
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = var.log_retention_days
}

# --- The function itself: a container image on ARM64/Graviton ---
resource "aws_lambda_function" "api" {
  function_name = local.name
  role          = aws_iam_role.lambda.arn

  package_type  = "Image"
  image_uri     = "${local.repo_url}@${data.aws_ecr_image.api.image_digest}"
  architectures = ["arm64"]

  memory_size = var.lambda_memory_mb
  timeout     = var.lambda_timeout_s

  environment {
    variables = {
      # No plaintext key here — only the name of the secret to fetch at cold start
      # (service/secrets.py reads OPENAI_SECRET_NAME, calls Secrets Manager).
      OPENAI_SECRET_NAME = aws_secretsmanager_secret.openai.name

      # Phase 7: Langfuse tracing. Secret name only (may hold empty '{}' -> the
      # app runs untraced). LANGFUSE_HOST is not a secret; both env var names are
      # set because the SDK version in use determines which one it reads.
      LANGFUSE_SECRET_NAME = aws_secretsmanager_secret.langfuse.name
      LANGFUSE_HOST        = var.langfuse_host
      LANGFUSE_BASE_URL    = var.langfuse_host
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.basic_execution,
    aws_cloudwatch_log_group.lambda,
  ]
}

# --- Function URL (AWS_IAM auth) ---
# The synchronous front door. Chosen over API Gateway because API Gateway caps
# integration time at 30s and our cold start (~49s) plus some warm LLM syntheses
# exceed that; a Function URL honours the function's own 120s timeout. AuthType is
# AWS_IAM because this account blocks anonymous (public) function URLs — callers
# SigV4-sign requests (scripts/demo_query.sh). The caller's IAM identity
# (the citementor-deploy user) authorises invocation, so no resource policy is
# needed.
resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "AWS_IAM"
}
