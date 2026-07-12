#!/usr/bin/env bash
#
# One-time bootstrap for the CI/CD (Phase 6) prerequisites that CANNOT live in the
# main Terraform, because they must exist *before* Terraform can run in CI:
#
#   1. An S3 bucket to hold Terraform *remote state* (so the "Deploy" workflow run
#      and a later "Teardown" run share one state). Uses S3-native locking.
#   2. A GitHub OIDC identity provider in IAM, so GitHub Actions can get short-lived
#      AWS credentials by exchanging its OIDC token — NO long-lived access keys.
#   3. A scoped IAM role the workflows assume (trusted ONLY by this repo).
#   4. (Phase 7) An AWS Budget with an email alert — deliberately OUTSIDE the app
#      Terraform, because the cost-runaway risk it guards against (Bedrock/OpenAI/
#      Lambda usage) exists independent of whether the app stack is currently
#      applied or torn down.
#
# These are cheap, standing resources (the S3 state bucket is a few KB → ~$0/mo;
# IAM/OIDC/Budgets are free). They intentionally survive `terraform destroy` — the
# state (and the budget alert) must outlive the app. Re-running this script is
# safe (idempotent).
#
# Run once, locally, as an admin (the citementor-deploy user):  bash infra/bootstrap.sh
set -euo pipefail

# --- Config (edit here if you fork/rename) -----------------------------------
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGION="ap-south-1"
GH_OWNER="sanjayg96"
GH_REPO="agentic_rag_citementor_2"
GH_BRANCH="aws-deploy"                       # workflows run on this branch
STATE_BUCKET="citementor-tfstate-${ACCOUNT_ID}"
ROLE_NAME="citementor-github-actions"
OIDC_HOST="token.actions.githubusercontent.com"
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_HOST}"
BUDGET_NAME="citementor-monthly"
BUDGET_LIMIT_USD="5"
ALERT_EMAIL="sanjaybg96@gmail.com"   # edit if you want budget alerts elsewhere

echo ">> Account ${ACCOUNT_ID} | region ${REGION} | repo ${GH_OWNER}/${GH_REPO}"

# --- 1. S3 state bucket ------------------------------------------------------
if aws s3api head-bucket --bucket "${STATE_BUCKET}" 2>/dev/null; then
  echo ">> [1/4] State bucket ${STATE_BUCKET} already exists."
else
  echo ">> [1/4] Creating state bucket ${STATE_BUCKET}..."
  aws s3api create-bucket --bucket "${STATE_BUCKET}" --region "${REGION}" \
    --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
fi
# Versioning (recover a clobbered/corrupt state), encryption, and block all public access.
aws s3api put-bucket-versioning --bucket "${STATE_BUCKET}" \
  --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption --bucket "${STATE_BUCKET}" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
aws s3api put-public-access-block --bucket "${STATE_BUCKET}" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
echo "   bucket ready (versioned + encrypted + private)."

# --- 2. GitHub OIDC provider -------------------------------------------------
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "${OIDC_ARN}" >/dev/null 2>&1; then
  echo ">> [2/4] OIDC provider already exists."
else
  echo ">> [2/4] Creating GitHub OIDC provider..."
  # Thumbprints are no longer security-critical for GitHub (AWS validates against a
  # trusted CA library) but the API still requires the field; these are GitHub's.
  aws iam create-open-id-connect-provider \
    --url "https://${OIDC_HOST}" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 1c58a3a8518e8759bf075b76b750d4f2df264fce >/dev/null
fi

# --- 3. CI role (trusted only by this repo's aws-deploy branch) ---------------
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT

cat > "${TMP}/trust.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "${OIDC_ARN}" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "${OIDC_HOST}:aud": "sts.amazonaws.com" },
      "StringLike":   { "${OIDC_HOST}:sub": "repo:${GH_OWNER}/${GH_REPO}:ref:refs/heads/${GH_BRANCH}" }
    }
  }]
}
JSON

# Scoped permissions: exactly what `terraform apply/destroy` for THIS stack touches.
cat > "${TMP}/perms.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "TfState", "Effect": "Allow",
      "Action": ["s3:ListBucket","s3:GetObject","s3:PutObject","s3:DeleteObject"],
      "Resource": ["arn:aws:s3:::${STATE_BUCKET}","arn:aws:s3:::${STATE_BUCKET}/*"] },
    { "Sid": "Sts", "Effect": "Allow", "Action": ["sts:GetCallerIdentity"], "Resource": "*" },
    { "Sid": "EcrAuth", "Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": "*" },
    { "Sid": "EcrRepo", "Effect": "Allow",
      "Action": ["ecr:CreateRepository","ecr:DeleteRepository","ecr:DescribeRepositories",
                 "ecr:DescribeImages","ecr:ListImages","ecr:BatchDeleteImage",
                 "ecr:TagResource","ecr:UntagResource","ecr:ListTagsForResource",
                 "ecr:PutImageTagMutability","ecr:GetRepositoryPolicy","ecr:SetRepositoryPolicy",
                 "ecr:GetLifecyclePolicy","ecr:PutLifecyclePolicy","ecr:DeleteLifecyclePolicy",
                 "ecr:BatchCheckLayerAvailability","ecr:GetDownloadUrlForLayer","ecr:BatchGetImage",
                 "ecr:InitiateLayerUpload","ecr:UploadLayerPart","ecr:CompleteLayerUpload","ecr:PutImage"],
      "Resource": "arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/citementor-api" },
    { "Sid": "Lambda", "Effect": "Allow", "Action": ["lambda:*"],
      "Resource": ["arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:citementor-api"] },
    { "Sid": "IamExecRole", "Effect": "Allow",
      "Action": ["iam:CreateRole","iam:DeleteRole","iam:GetRole","iam:PassRole","iam:TagRole","iam:UntagRole",
                 "iam:PutRolePolicy","iam:DeleteRolePolicy","iam:GetRolePolicy","iam:ListRolePolicies",
                 "iam:AttachRolePolicy","iam:DetachRolePolicy","iam:ListAttachedRolePolicies",
                 "iam:ListInstanceProfilesForRole"],
      "Resource": "arn:aws:iam::${ACCOUNT_ID}:role/citementor-lambda-role" },
    { "Sid": "Secrets", "Effect": "Allow",
      "Action": ["secretsmanager:CreateSecret","secretsmanager:DeleteSecret","secretsmanager:DescribeSecret",
                 "secretsmanager:PutSecretValue","secretsmanager:GetSecretValue","secretsmanager:TagResource",
                 "secretsmanager:UntagResource","secretsmanager:GetResourcePolicy","secretsmanager:ListSecretVersionIds"],
      "Resource": "arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:citementor/*" },
    { "Sid": "LogsGroup", "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:DeleteLogGroup","logs:PutRetentionPolicy",
                 "logs:TagResource","logs:UntagResource","logs:ListTagsForResource","logs:ListTagsLogGroup"],
      "Resource": ["arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/lambda/citementor-api",
                   "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/lambda/citementor-api:*"] },
    { "Sid": "LogsDescribe", "Effect": "Allow", "Action": ["logs:DescribeLogGroups"], "Resource": "*" },
    { "Sid": "Sns", "Effect": "Allow",
      "Action": ["sns:CreateTopic","sns:DeleteTopic","sns:GetTopicAttributes","sns:SetTopicAttributes",
                 "sns:TagResource","sns:UntagResource","sns:ListTagsForResource",
                 "sns:Subscribe","sns:Unsubscribe","sns:ListSubscriptionsByTopic",
                 "sns:GetSubscriptionAttributes","sns:SetSubscriptionAttributes"],
      "Resource": ["arn:aws:sns:${REGION}:${ACCOUNT_ID}:citementor-alerts",
                   "arn:aws:sns:${REGION}:${ACCOUNT_ID}:citementor-alerts:*"] },
    { "Sid": "CloudwatchAlarms", "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricAlarm","cloudwatch:DeleteAlarms","cloudwatch:DescribeAlarms",
                 "cloudwatch:TagResource","cloudwatch:UntagResource","cloudwatch:ListTagsForResource"],
      "Resource": ["arn:aws:cloudwatch:${REGION}:${ACCOUNT_ID}:alarm:citementor-api-errors",
                   "arn:aws:cloudwatch:${REGION}:${ACCOUNT_ID}:alarm:citementor-api-latency-p95"] }
  ]
}
JSON

if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo ">> [3/4] Role ${ROLE_NAME} exists — updating trust + permissions..."
  aws iam update-assume-role-policy --role-name "${ROLE_NAME}" \
    --policy-document "file://${TMP}/trust.json" >/dev/null
else
  echo ">> [3/4] Creating role ${ROLE_NAME}..."
  aws iam create-role --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "file://${TMP}/trust.json" \
    --description "GitHub Actions OIDC role for the citementor CI/CD workflows" >/dev/null
fi
aws iam put-role-policy --role-name "${ROLE_NAME}" \
  --policy-name "citementor-cicd" \
  --policy-document "file://${TMP}/perms.json"

# --- 4. AWS Budget with an email alert (Phase 7, standing) --------------------
# Deliberately NOT in the app Terraform: it must keep watching spend even while
# the app stack is torn down (Bedrock/OpenAI calls, or someone hammering the
# Function URL, cost money regardless of Lambda's own on/off state). No IAM
# change needed here — this runs once, manually, as the admin user.
cat > "${TMP}/budget.json" <<JSON
{
  "BudgetName": "${BUDGET_NAME}",
  "BudgetLimit": { "Amount": "${BUDGET_LIMIT_USD}", "Unit": "USD" },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
JSON
cat > "${TMP}/notifications.json" <<JSON
[
  { "Notification": { "NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 80, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "${ALERT_EMAIL}" }] },
  { "Notification": { "NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN", "Threshold": 100, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "${ALERT_EMAIL}" }] }
]
JSON

if aws budgets describe-budget --account-id "${ACCOUNT_ID}" --budget-name "${BUDGET_NAME}" >/dev/null 2>&1; then
  echo ">> [4/4] Budget ${BUDGET_NAME} already exists — updating limit..."
  aws budgets update-budget --account-id "${ACCOUNT_ID}" --new-budget "file://${TMP}/budget.json" >/dev/null
else
  echo ">> [4/4] Creating budget ${BUDGET_NAME} (\$${BUDGET_LIMIT_USD}/mo, alerts to ${ALERT_EMAIL})..."
  aws budgets create-budget --account-id "${ACCOUNT_ID}" \
    --budget "file://${TMP}/budget.json" \
    --notifications-with-subscribers "file://${TMP}/notifications.json" >/dev/null
fi
echo "   AWS Budgets emails directly — no confirmation-link step (unlike the SNS alarm topic above)."

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo ""
echo ">> DONE. Set these in GitHub (repo → Settings → Secrets and variables → Actions):"
echo "     Variable  AWS_ROLE_ARN = ${ROLE_ARN}"
echo "     Variable  AWS_REGION   = ${REGION}"
echo "     Secret    OPENAI_API_KEY = <your key>"
echo "     Secret    LANGFUSE_PUBLIC_KEY = <optional; from cloud.langfuse.com>"
echo "     Secret    LANGFUSE_SECRET_KEY = <optional; from cloud.langfuse.com>"
echo ""
echo "   State backend (already referenced in infra/backend.tf):"
echo "     bucket=${STATE_BUCKET}  key=citementor/terraform.tfstate  region=${REGION}"
