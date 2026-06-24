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
