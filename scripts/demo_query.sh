#!/usr/bin/env bash
#
# Demo helper for the CiteMentor Lambda Function URL.
#
# The function URL uses AWS_IAM auth (this account blocks anonymous/public
# function URLs), so every request must be SigV4-signed. This script does the
# signing for you with your configured AWS credentials.
#
#   Warm it up before a demo:   scripts/demo_query.sh --warm
#   Ask a question:             scripts/demo_query.sh "What is virtue?"
#   Health check:               scripts/demo_query.sh --health
#
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
FUNCTION="${LAMBDA_FUNCTION:-citementor-api}"

URL=$(aws lambda get-function-url-config --function-name "$FUNCTION" \
        --query FunctionUrl --output text)
AK=$(aws configure get aws_access_key_id)
SK=$(aws configure get aws_secret_access_key)
ST=$(aws configure get aws_session_token || true)

# Temporary-credential support (no-op for long-lived keys).
EXTRA=()
[ -n "${ST:-}" ] && EXTRA=(-H "x-amz-security-token: ${ST}")

# Note: ${EXTRA[@]+...} guards against "unbound variable" for an empty array
# under `set -u` on macOS's bash 3.2.
sign=( curl -s --aws-sigv4 "aws:amz:${REGION}:lambda" --user "${AK}:${SK}" ${EXTRA[@]+"${EXTRA[@]}"} )

case "${1:-}" in
  --health)
    "${sign[@]}" "${URL}health"; echo
    ;;
  --warm)
    echo "Warming (first call pays the cold start, ~40-50s)..." >&2
    "${sign[@]}" -XPOST "${URL}query" -H 'content-type: application/json' \
      -d '{"query":"warmup"}' >/dev/null && echo "warm ✅" >&2
    ;;
  "")
    echo "usage: $0 \"<question>\" | --warm | --health" >&2; exit 1
    ;;
  *)
    Q="$1"
    BODY=$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$Q")
    "${sign[@]}" -XPOST "${URL}query" -H 'content-type: application/json' \
      -d "$BODY" | python3 -m json.tool
    ;;
esac
