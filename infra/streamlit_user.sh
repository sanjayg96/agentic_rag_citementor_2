#!/usr/bin/env bash
#
# One-time setup for Phase 8 (Streamlit → deployed Lambda).
#
# Creates a DEDICATED, least-privilege IAM user whose only power is to discover
# and invoke the CiteMentor Function URL. Its long-lived access key goes into the
# Streamlit Community Cloud app's secrets so the UI can SigV4-sign requests to the
# AWS_IAM-auth Function URL (Streamlit Cloud can't use the OIDC path the CI role
# uses).
#
# Why this lives OUTSIDE the app Terraform (like bootstrap.sh): the app stack is
# torn down to $0 between demos. If these credentials were part of that stack they
# would be destroyed and regenerated on every deploy, breaking the Streamlit secret
# each cycle. The user is scoped to the function *by name* (citementor-api, stable
# across deploys), so it keeps working across teardown→redeploy without any change
# to Streamlit. The Function URL itself changes each cycle, which is why the app
# discovers it via lambda:GetFunctionUrlConfig instead of hard-coding it.
#
# Usage:
#   bash infra/streamlit_user.sh            # create user + policy, mint a key if none
#   bash infra/streamlit_user.sh --rotate   # delete existing keys, mint a fresh one
#
# Run locally as an admin (the citementor-deploy user).
set -euo pipefail

REGION="ap-south-1"
USER_NAME="citementor-streamlit"
FUNCTION="citementor-api"
POLICY_NAME="citementor-invoke-url"
ROTATE="${1:-}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION}"

echo ">> Account ${ACCOUNT_ID} | region ${REGION} | user ${USER_NAME} | function ${FUNCTION}"

# --- 1. The user ------------------------------------------------------------
if aws iam get-user --user-name "${USER_NAME}" >/dev/null 2>&1; then
  echo ">> [1/3] User ${USER_NAME} already exists."
else
  echo ">> [1/3] Creating user ${USER_NAME}..."
  aws iam create-user --user-name "${USER_NAME}" \
    --tags Key=project,Value=citementor Key=purpose,Value=streamlit-frontend >/dev/null
fi

# --- 2. Least-privilege inline policy ---------------------------------------
# GetFunctionUrlConfig: discover the current URL (it changes each deploy).
# InvokeFunctionUrl: call it. Scoped to this one function's ARN — that is the whole
# authorization boundary. We deliberately do NOT add a
# `lambda:FunctionUrlAuthType == AWS_IAM` condition: the real Function URL invoke
# request does not reliably populate that condition key, so requiring it caused the
# invoke to be denied (403) even though discovery was allowed. The condition would be
# redundant anyway — the URL's auth type is hard-pinned to AWS_IAM in lambda.tf and
# this account blocks public URLs.
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
cat > "${TMP}/policy.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DiscoverFunctionUrl",
      "Effect": "Allow",
      "Action": ["lambda:GetFunctionUrlConfig"],
      "Resource": "${FUNCTION_ARN}"
    },
    {
      "Sid": "InvokeFunctionUrl",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunctionUrl"],
      "Resource": "${FUNCTION_ARN}"
    }
  ]
}
JSON
aws iam put-user-policy --user-name "${USER_NAME}" \
  --policy-name "${POLICY_NAME}" \
  --policy-document "file://${TMP}/policy.json"
echo ">> [2/3] Inline policy ${POLICY_NAME} applied (GetFunctionUrlConfig + InvokeFunctionUrl)."

# --- 3. Access key ----------------------------------------------------------
EXISTING="$(aws iam list-access-keys --user-name "${USER_NAME}" \
             --query 'AccessKeyMetadata[].AccessKeyId' --output text)"

if [ "${ROTATE}" = "--rotate" ] && [ -n "${EXISTING}" ]; then
  echo ">> [3/3] Rotating: deleting existing key(s) ${EXISTING}..."
  for kid in ${EXISTING}; do
    aws iam delete-access-key --user-name "${USER_NAME}" --access-key-id "${kid}"
  done
  EXISTING=""
fi

if [ -n "${EXISTING}" ]; then
  echo ">> [3/3] User already has an access key (${EXISTING})."
  echo "   AWS never re-reveals the secret. To mint a fresh pair, re-run with --rotate."
  echo "   (An IAM user may hold at most 2 keys.)"
  exit 0
fi

echo ">> [3/3] Creating a new access key..."
CREDS_JSON="$(aws iam create-access-key --user-name "${USER_NAME}" --output json)"
AK="$(printf '%s' "${CREDS_JSON}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["AccessKey"]["AccessKeyId"])')"
SK="$(printf '%s' "${CREDS_JSON}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["AccessKey"]["SecretAccessKey"])')"

cat <<EOF

============================================================================
 DONE. Paste this into your Streamlit app's Secrets
 (Streamlit Cloud → your app → ⋮ → Settings → Secrets). Shown ONCE — AWS will
 not reveal the secret again; re-run with --rotate if you lose it.
----------------------------------------------------------------------------
[aws]
access_key_id     = "${AK}"
secret_access_key = "${SK}"
region            = "${REGION}"
function_name     = "${FUNCTION}"
============================================================================
EOF
