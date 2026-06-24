# Input variables. Defaults match the manually-built Phase 0-4 stack so a bare
# `terraform apply` reproduces it exactly. Override via terraform.tfvars or
# -var flags.

variable "region" {
  description = "AWS region for all resources."
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Base name for resources (ECR repo, Lambda, role, secret prefix)."
  type        = string
  default     = "citementor"
}

variable "lambda_memory_mb" {
  description = "Lambda memory (MB). CPU scales with memory; ~3GB keeps the RAG cold start tolerable."
  type        = number
  default     = 3008
}

variable "lambda_timeout_s" {
  description = "Lambda timeout (s). Must cover the ~49s cold start + LLM synthesis; Function URL honours this (API Gateway's 30s cap could not)."
  type        = number
  default     = 120
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the Lambda log group."
  type        = number
  default     = 14
}

variable "image_tag" {
  description = "ECR image tag to build/push. The Lambda actually pins the immutable digest of whatever this tag resolves to after the build."
  type        = string
  default     = "latest"
}

# The OpenAI key is intentionally NOT a Terraform variable that gets written
# into a resource — that would persist plaintext in the local state file. Instead
# the secret VALUE is pushed by a null_resource whose shell command reads the key
# from the OPENAI_API_KEY environment variable at apply time (see secrets.tf).
# Terraform only ever manages the empty secret *container*.
