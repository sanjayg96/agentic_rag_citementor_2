# AWS provider. Region comes from var.region (defaults to ap-south-1, where the
# stack was first built). Credentials are read from the standard AWS CLI chain
# (the `citementor-deploy` IAM user configured via `aws configure`).

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "citementor"
      ManagedBy = "terraform"
    }
  }
}

# Used to discover the current account id and region without hardcoding them,
# so the config is portable across accounts.
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
