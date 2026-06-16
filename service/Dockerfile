# CiteMentor API — AWS Lambda container image (ARM64 / Graviton).
#
# Built from the AWS Lambda Python base image, which already bundles the Lambda
# Runtime Interface Client (and a local Runtime Interface Emulator for testing).
# Build from the REPO ROOT so the COPY paths below resolve:
#   docker buildx build --platform linux/arm64 -f service/Dockerfile -t citementor-api .
#
# The core pipeline reads relative paths (config/, catalog.json, prompts.yaml,
# storage/), so everything is copied under LAMBDA_TASK_ROOT (=/var/task), which
# is the function's working directory at runtime.

FROM public.ecr.aws/lambda/python:3.12-arm64

# 1. Dependencies first (better layer caching). Slim, OpenAI-only set.
COPY service/requirements.txt ${LAMBDA_TASK_ROOT}/service/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/service/requirements.txt

# 2. Application code.
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY service/ ${LAMBDA_TASK_ROOT}/service/

# 3. Config + bundled corpus (read-only). storage/ holds the Chroma DB and BM25
#    index; in Phase 2 the app copies the Chroma dir to /tmp on cold start.
COPY config/ ${LAMBDA_TASK_ROOT}/config/
COPY storage/ ${LAMBDA_TASK_ROOT}/storage/
COPY catalog.json prompts.yaml ${LAMBDA_TASK_ROOT}/

# Lambda entry point: FastAPI app wrapped by Mangum (service/app.py::handler).
CMD ["service.app.handler"]
