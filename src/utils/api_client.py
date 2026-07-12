"""Remote client for the CiteMentor Lambda Function URL (Phase 8).

This lets the Streamlit UI act as a *thin front end* to the deployed AWS stack
instead of running the RAG pipeline locally. It is the programmatic equivalent
of `scripts/demo_query.sh`:

  * The Function URL uses ``AWS_IAM`` auth, so every request is SigV4-signed with
    long-lived credentials belonging to a dedicated least-privilege IAM user
    (``citementor-streamlit`` — see ``infra/streamlit_user.sh``).
  * The URL itself changes on every teardown→redeploy cycle, so it is
    *discovered* at call time via ``lambda:GetFunctionUrlConfig`` rather than
    hard-coded. Set ``function_url`` explicitly in secrets only to override.

Configuration comes from ``st.secrets`` (Streamlit Community Cloud) with an
environment-variable fallback for local testing. The presence of AWS credentials
is the toggle: when they are absent (e.g. the ``master`` deployment), the app
runs the local pipeline and this module is never used.

    # .streamlit/secrets.toml
    [aws]
    access_key_id     = "AKIA..."
    secret_access_key = "..."
    region            = "ap-south-1"      # optional, defaults below
    function_name     = "citementor-api"  # optional, defaults below
    # function_url    = "https://xxxx.lambda-url.ap-south-1.on.aws/"  # optional override
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

DEFAULT_REGION = "ap-south-1"
DEFAULT_FUNCTION_NAME = "citementor-api"

# The demo Lambda pays a ~40-50s cold start on the first call after a deploy, so
# the request timeout has to comfortably exceed that. The function's own ceiling
# is 120s; we allow a little headroom for network + signing.
REQUEST_TIMEOUT_S = 130


class BackendOfflineError(RuntimeError):
    """The AWS stack is not currently deployed (scaled to $0), or the Function URL
    could not be discovered. Callers render a friendly 'deploy the stack' hint."""


class BackendError(RuntimeError):
    """The stack is up but the request failed (auth, timeout, 5xx, bad JSON)."""


@dataclass(frozen=True)
class RemoteConfig:
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str]
    region: str
    function_name: str
    function_url: Optional[str]  # explicit override; discovered when None


def _read_secret_section() -> Dict[str, Any]:
    """Merge ``st.secrets['aws']`` (if present) with matching env vars.

    Accessing ``st.secrets`` when no secrets file exists raises, so every read is
    guarded — a missing/broken secrets store simply means 'not configured'.
    """
    values: Dict[str, Any] = {}

    try:
        import streamlit as st

        aws = st.secrets.get("aws") if hasattr(st, "secrets") else None
        if aws:
            values.update(dict(aws))
    except Exception:
        # No secrets file, not running under Streamlit, or malformed TOML — fall
        # back to the environment.
        pass

    # Environment fallback (also lets `demo_query.sh`-style creds drive local runs).
    env_map = {
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "session_token": os.getenv("AWS_SESSION_TOKEN"),
        "region": os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        "function_name": os.getenv("CITEMENTOR_FUNCTION_NAME"),
        "function_url": os.getenv("CITEMENTOR_FUNCTION_URL"),
    }
    for key, val in env_map.items():
        if val and not values.get(key):
            values[key] = val

    return values


def load_config() -> Optional[RemoteConfig]:
    """Return a RemoteConfig if AWS credentials are configured, else None.

    None is the 'run the local pipeline' signal — it is not an error.
    """
    values = _read_secret_section()
    access_key = values.get("access_key_id")
    secret_key = values.get("secret_access_key")
    if not (access_key and secret_key):
        return None

    return RemoteConfig(
        access_key_id=str(access_key),
        secret_access_key=str(secret_key),
        session_token=(str(values["session_token"]) if values.get("session_token") else None),
        region=str(values.get("region") or DEFAULT_REGION),
        function_name=str(values.get("function_name") or DEFAULT_FUNCTION_NAME),
        function_url=(str(values["function_url"]) if values.get("function_url") else None),
    )


def is_configured() -> bool:
    """True when the UI should call the deployed API instead of running locally."""
    return load_config() is not None


class CiteMentorRemoteAPI:
    """Thin, SigV4-signing HTTP client for the CiteMentor Function URL."""

    def __init__(self, config: RemoteConfig):
        self._config = config
        self._resolved_url: Optional[str] = None

    # -- URL discovery --------------------------------------------------------
    def _base_url(self) -> str:
        """Discover (and cache) the Function URL, or use the explicit override.

        Raises BackendOfflineError when the function does not exist (torn down).
        """
        if self._config.function_url:
            return self._config.function_url.rstrip("/") + "/"
        if self._resolved_url:
            return self._resolved_url

        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        client = boto3.client(
            "lambda",
            region_name=self._config.region,
            aws_access_key_id=self._config.access_key_id,
            aws_secret_access_key=self._config.secret_access_key,
            aws_session_token=self._config.session_token,
        )
        try:
            resp = client.get_function_url_config(FunctionName=self._config.function_name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("ResourceNotFoundException", "AccessDeniedException"):
                # Not deployed (or the URL config was destroyed) — treat as offline.
                raise BackendOfflineError(
                    f"Function '{self._config.function_name}' has no URL config "
                    f"(the stack is likely torn down): {code}"
                ) from exc
            raise BackendError(f"Could not look up the Function URL: {code}") from exc
        except BotoCoreError as exc:
            raise BackendError(f"AWS SDK error resolving the Function URL: {exc}") from exc

        self._resolved_url = resp["FunctionUrl"].rstrip("/") + "/"
        return self._resolved_url

    # -- Signed requests ------------------------------------------------------
    def _signed_request(self, method: str, path: str, body: Optional[bytes]) -> "requests.Response":
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        url = self._base_url() + path.lstrip("/")
        headers = {"content-type": "application/json"} if body is not None else {}

        aws_request = AWSRequest(method=method, url=url, data=body, headers=headers)
        credentials = Credentials(
            self._config.access_key_id,
            self._config.secret_access_key,
            self._config.session_token,
        )
        # Service name for a Lambda Function URL is "lambda" (mirrors demo_query.sh's
        # `--aws-sigv4 aws:amz:<region>:lambda`).
        SigV4Auth(credentials, "lambda", self._config.region).add_auth(aws_request)

        try:
            return requests.request(
                method,
                url,
                data=body,
                headers=dict(aws_request.headers),
                timeout=REQUEST_TIMEOUT_S,
            )
        except requests.Timeout as exc:
            raise BackendError(
                "The request timed out. The first call after a deploy pays a cold "
                "start (~40-50s) — try again."
            ) from exc
        except requests.RequestException as exc:
            raise BackendError(f"Network error calling the API: {exc}") from exc

    # -- Public API -----------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        resp = self._signed_request("GET", "health", None)
        if resp.status_code != 200:
            raise BackendError(f"Health check returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def query(self, prompt: str) -> Dict[str, Any]:
        """Run one RAG turn on the deployed Lambda and return the parsed response.

        The shape matches ``service/app.py::QueryResponse``:
        ``{answer, route, sources, cache_hit, cache_similarity, timings}``.
        """
        import json

        body = json.dumps({"query": prompt}).encode("utf-8")
        resp = self._signed_request("POST", "query", body)

        if resp.status_code == 403:
            raise BackendError(
                "Request was not authorized (HTTP 403). Check the AWS access key / "
                "secret in Streamlit secrets and that the IAM user may invoke this "
                "Function URL."
            )
        if resp.status_code >= 500:
            raise BackendError(f"The API returned HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code != 200:
            raise BackendError(f"Unexpected HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            return resp.json()
        except ValueError as exc:
            raise BackendError(f"API returned non-JSON body: {resp.text[:300]}") from exc
