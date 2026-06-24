# --- Elastic Container Registry: the private repo holding the Lambda image ---

resource "aws_ecr_repository" "api" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  # Let `terraform destroy` remove the repo even though it still holds images.
  # Without this, destroy errors out on a non-empty repository.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

# Build the ARM64 image locally and push it to ECR. This runs the same buildx
# command we used by hand in Phase 1/3, so `terraform apply` reproduces the image
# from nothing. Requires Docker running locally (and `aws` CLI for the login).
#
# --provenance=false forces a single-manifest image; the default buildx output is
# a multi-arch manifest *list* that Lambda rejects ("source image ... is not
# valid"). Learned the hard way in Phase 3.
resource "null_resource" "image_build_push" {
  triggers = {
    source_hash = local.source_hash
    image_tag   = var.image_tag
    repo_url    = aws_ecr_repository.api.repository_url
  }

  provisioner "local-exec" {
    working_dir = "${path.module}/.."
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -euo pipefail
      echo ">> Logging in to ECR..."
      aws ecr get-login-password --region ${var.region} \
        | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.region}.amazonaws.com
      echo ">> Building + pushing ARM64 image (${var.image_tag})..."
      docker buildx build --platform linux/arm64 --provenance=false \
        -f service/Dockerfile \
        -t ${aws_ecr_repository.api.repository_url}:${var.image_tag} \
        --push .
    EOT
  }
}

# Read back the digest of the image we just pushed. Pinning the Lambda to an
# immutable digest (rather than the mutable :latest tag) means every rebuild
# changes image_uri, so `terraform apply` actually rolls the function forward.
data "aws_ecr_image" "api" {
  repository_name = aws_ecr_repository.api.name
  image_tag       = var.image_tag

  # Deferred to apply time, after the build/push provisioner has run.
  depends_on = [null_resource.image_build_push]
}
