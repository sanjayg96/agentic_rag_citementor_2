# Terraform + provider version pins.
#
# State is LOCAL and gitignored (see repo .gitignore: *.tfstate*). This is a
# solo project, so a local backend is fine. For a team you would move state to a
# shared backend (S3 + a DynamoDB lock table) — noted as a future enhancement.

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}
