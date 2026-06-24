locals {
  account_id = data.aws_caller_identity.current.account_id
  name       = "${var.project_name}-api" # citementor-api

  repo_url = aws_ecr_repository.api.repository_url

  # Source files baked into the image. Changing any of them changes source_hash,
  # which retriggers the build/push (and so produces a new image digest the
  # Lambda then pins). The bundled corpus under storage/ is deliberately EXCLUDED
  # (large, rarely-changing binaries) — to ship a new corpus, bump var.image_tag
  # or `terraform taint null_resource.image_build_push`.
  source_files = sort(concat(
    [for f in fileset("${path.module}/..", "src/**/*.py") : f],
    [for f in fileset("${path.module}/..", "service/**/*.py") : f],
    ["service/Dockerfile", "service/requirements.txt", "config/retrieval.yaml", "prompts.yaml", "catalog.json"],
  ))
  source_hash = md5(join("", [for f in local.source_files : filemd5("${path.module}/../${f}")]))
}
