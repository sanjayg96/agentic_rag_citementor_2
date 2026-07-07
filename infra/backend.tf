# Remote Terraform state (added in Phase 6 for CI/CD).
#
# Why remote now: the "Deploy" and "Teardown" GitHub Actions run in SEPARATE,
# ephemeral runners. Local state can't be shared between them, so state lives in
# S3. Locking uses S3's native conditional-write lock (`use_lockfile`, Terraform
# >= 1.11) — no DynamoDB table needed.
#
# The bucket is created ONCE by infra/bootstrap.sh (it must exist before
# `terraform init`). Backend blocks can't use variables, so values are literal;
# edit them together with bootstrap.sh if you fork/rename.
#
# Local runs and CI both use this same backend, so `terraform apply` on your
# laptop and from Actions operate on one shared state.
terraform {
  backend "s3" {
    bucket       = "citementor-tfstate-505192030409"
    key          = "citementor/terraform.tfstate"
    region       = "ap-south-1"
    encrypt      = true
    use_lockfile = true
  }
}
