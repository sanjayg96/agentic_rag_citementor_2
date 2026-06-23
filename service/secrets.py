"""Load OPENAI_API_KEY from AWS Secrets Manager at cold start.

The container image ships with **no** API key baked in. On Lambda, the key lives
in Secrets Manager and the function's execution role is granted read access to
*only* that secret's ARN. At cold start we fetch it once and populate
``os.environ["OPENAI_API_KEY"]``, so the existing core code (retriever / semantic
cache read that env var lazily on the first query) works unchanged.

Locally and in dev, ``OPENAI_API_KEY`` is normally already set (via ``.env``), so
this is a no-op and boto3 / AWS are never touched — boto3 is imported lazily for
exactly that reason. ``boto3`` is provided by the AWS Lambda Python base image,
so it is not listed in ``service/requirements.txt``.
"""

import os


def load_openai_key_from_secrets() -> None:
    """If no OPENAI_API_KEY is set but a secret is configured, fetch and set it.

    Controlled by the ``OPENAI_SECRET_NAME`` env var (the secret's name or ARN).
    Idempotent and cheap: returns immediately when the key is already present.
    """
    # Already provided (local dev, or an explicit override) → nothing to do.
    if os.getenv("OPENAI_API_KEY"):
        return

    secret_id = os.getenv("OPENAI_SECRET_NAME")
    if not secret_id:
        # No secret configured; leave the env untouched. The core raises a clear
        # error on first query if the key is genuinely missing.
        return

    import json
    import boto3  # Lazy: only needed on Lambda, where the base image provides it.

    secret = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)["SecretString"]

    # Accept either a raw string secret ("sk-...") or a JSON blob
    # ({"OPENAI_API_KEY": "sk-..."}); we store the raw string form.
    try:
        key = json.loads(secret).get("OPENAI_API_KEY", secret)
    except (json.JSONDecodeError, AttributeError):
        key = secret

    os.environ["OPENAI_API_KEY"] = key
