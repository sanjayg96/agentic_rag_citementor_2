# --- IAM execution role for the Lambda ---

# Trust policy: only the Lambda service may assume this role.
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.project_name}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# AWS-managed policy granting CloudWatch Logs create/put — the minimum a Lambda
# needs to emit logs.
resource "aws_iam_role_policy_attachment" "basic_execution" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least-privilege inline policy: read GetSecretValue on ONLY the OpenAI secret.
data "aws_iam_policy_document" "read_openai_secret" {
  statement {
    sid       = "ReadOpenAIKeyOnly"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.openai.arn]
  }
}

resource "aws_iam_role_policy" "read_openai_secret" {
  name   = "${var.project_name}-read-openai-secret"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.read_openai_secret.json
}

# Least-privilege inline policy for the Bedrock inference mode. Allows InvokeModel
# on exactly: the Titan embedding model (on-demand, region-local), the Nova
# inference-profile ARNs, and the Nova underlying foundation models (region
# wildcard, since a cross-region profile may route to any of its APAC regions —
# still scoped to the specific model). Harmless when inference_mode=openai
# (nothing calls Bedrock), so it is always attached.
data "aws_iam_policy_document" "invoke_bedrock" {
  statement {
    sid    = "InvokeBedrockModels"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = concat(
      # Titan embeddings (on-demand, this region only)
      ["arn:aws:bedrock:${var.region}::foundation-model/${var.bedrock_embedding_model_id}"],
      # Nova inference profiles (account-scoped, this region)
      [for profile_id, _ in var.bedrock_inference_profiles :
      "arn:aws:bedrock:${var.region}:${local.account_id}:inference-profile/${profile_id}"],
      # Nova underlying foundation models (any region the profile routes to)
      [for _, model_id in var.bedrock_inference_profiles :
      "arn:aws:bedrock:*::foundation-model/${model_id}"],
    )
  }
}

resource "aws_iam_role_policy" "invoke_bedrock" {
  name   = "${var.project_name}-invoke-bedrock"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.invoke_bedrock.json
}
