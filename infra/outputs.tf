output "function_url" {
  description = "Lambda Function URL (AWS_IAM auth; sign requests with scripts/demo_query.sh)."
  value       = aws_lambda_function_url.api.function_url
}

output "function_name" {
  description = "Lambda function name (used by scripts/demo_query.sh via LAMBDA_FUNCTION)."
  value       = aws_lambda_function.api.function_name
}

output "ecr_repository_url" {
  description = "ECR repository the image is pushed to."
  value       = aws_ecr_repository.api.repository_url
}

output "image_digest" {
  description = "Digest the Lambda is currently pinned to."
  value       = data.aws_ecr_image.api.image_digest
}

output "secret_name" {
  description = "Secrets Manager secret holding the OpenAI key."
  value       = aws_secretsmanager_secret.openai.name
}

output "demo_hint" {
  description = "How to run the demo once applied."
  value       = "Warm up:  scripts/demo_query.sh --warm   |   Ask:  scripts/demo_query.sh \"What is virtue?\""
}
