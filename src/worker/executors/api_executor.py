import email.utils
import logging
import math
import os
import threading
from datetime import UTC, datetime
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Any, ClassVar

import httpx

from shared.schemas.result import APIResult
from shared.tasks.specs import ApiSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.redact import is_credential_key, redact_url

from ..utils.redaction import redact_urls
from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    RunSignals,
    TaskCancelledError,
)

logger = logging.getLogger(__name__)

# Cache key: (base_url, timeout_seconds, verify_tls, follow_redirects)
_ClientKey = tuple[str, float, bool, bool]

_ROUTING_HEADERS = frozenset(
    {
        "host",
        "forwarded",
        "x-host",
        "x-original-host",
        "x-original-url",
        "x-rewrite-url",
    }
)
_ROUTING_HEADER_PREFIX = "x-forwarded-"

# Base delay between retry attempts, doubled each retry.
_RETRY_BACKOFF_SEC = 1.0
# Upper bound on any single retry wait.
_RETRY_BACKOFF_MAX_SEC = 60.0
# Upper bound on spec.api.retries.
_MAX_RETRIES = 10
# Failures before the request left the worker, so sending it again cannot repeat its
# effect.
_UNSENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def _is_routing_header(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in _ROUTING_HEADERS or lowered.startswith(_ROUTING_HEADER_PREFIX)


def _is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status is transient and worth retrying."""
    return status_code >= 500 or status_code in (408, 429)


class APIExecutor(Executor):
    """Performs a single HTTP request defined by task YAML.

    Without ``spec.api.url`` it calls ``NEBULA_API_BASE_URL`` with ``NEBULA_API_TOKEN``,
    unless ``spec.api.headers`` carries a credential header of its own. A request
    carrying the Nebula token drops any author header an ingress may route on and
    always verifies TLS, so neither routes nor exposes the token past the configured
    Nebula host. A ``spec.api.url`` is called with its own headers alone; the Nebula
    token is never sent to it.
    """

    name = "api"
    supported_task_types = frozenset({TaskType.API})

    # ---- Class-level connection pool (shared across all instances) ----
    _clients: ClassVar[dict[_ClientKey, httpx.Client]] = {}
    _clients_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._signals = RunSignals()

    def cancel(self, task_id: str) -> None:
        """Signal the executor to abort the current request and any retries."""
        self._signals.cancel(task_id)

    @classmethod
    def _base_url(cls, url: str) -> str:
        """Extract scheme + host + port from a URL for pool keying."""
        parsed = httpx.URL(url)
        # httpx.URL exposes .scheme, .host, .port; rebuild origin string
        port = parsed.port
        if port is None:
            return f"{parsed.scheme}://{parsed.host}"
        return f"{parsed.scheme}://{parsed.host}:{port}"

    @classmethod
    def _get_client(
        cls,
        base_url: str,
        timeout: httpx.Timeout,
        verify_tls: bool,
        follow_redirects: bool,
    ) -> httpx.Client:
        """Return a cached client or create a new one for the given parameters."""
        timeout_sec = timeout.connect  # all four fields are set to same value
        if timeout_sec is None:
            timeout_sec = 0.0
        key: _ClientKey = (base_url, float(timeout_sec), verify_tls, follow_redirects)
        with cls._clients_lock:
            client = cls._clients.get(key)
            if client is not None and not client.is_closed:
                return client
            # Create a new client for this combination
            # A pooled client serves many tasks and tenants; a stored cookie would ride
            # into the next task's request to the same host.
            client = httpx.Client(
                timeout=timeout,
                verify=verify_tls,
                follow_redirects=follow_redirects,
                cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
            )
            cls._clients[key] = client
            logger.debug(
                "Created new HTTP client for %s (verify=%s, timeout=%s)",
                base_url,
                verify_tls,
                timeout_sec,
            )
            return client

    def _request_with_retries(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        headers: dict[str, Any],
        params: dict[str, Any] | None,
        request_kwargs: dict[str, Any],
        retries: int,
    ) -> httpx.Response:
        """Issue the request, retrying transient failures up to ``retries`` times.

        A retryable failure is a connection error, where the request never left, or a
        transient HTTP status (5xx, 408, 429). Any other failure and a cancelled task
        stop the loop immediately. The final attempt's failure propagates to the
        caller.
        """
        attempt = 0
        while True:
            self._signals.raise_if_cancelled()
            try:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    **request_kwargs,
                )
            except _UNSENT_ERRORS as exc:
                if attempt >= retries:
                    raise
                attempt += 1
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "API request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    retries,
                    redact_urls(str(exc), url),
                    delay,
                )
                self._wait_for_backoff(delay)
                continue
            if resp.is_error and _is_retryable_status(resp.status_code):
                if attempt >= retries:
                    return resp
                attempt += 1
                delay = self._backoff_delay(attempt, resp)
                logger.warning(
                    "API request returned %s (attempt %d/%d); retrying in %.1fs",
                    resp.status_code,
                    attempt,
                    retries,
                    delay,
                )
                self._wait_for_backoff(delay)
                continue
            return resp

    def _backoff_delay(self, attempt: int, resp: httpx.Response | None = None) -> float:
        """Return the wait before the next attempt, honouring Retry-After."""
        if resp is not None:
            retry_after = self._retry_after_seconds(resp)
            if retry_after is not None:
                return min(retry_after, _RETRY_BACKOFF_MAX_SEC)
        return min(_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), _RETRY_BACKOFF_MAX_SEC)

    @staticmethod
    def _retry_after_seconds(resp: httpx.Response) -> float | None:
        """Parse the Retry-After header as seconds or an HTTP date."""
        value = resp.headers.get("Retry-After")
        if value is None:
            return None
        try:
            seconds = float(value)
        except ValueError:
            pass
        else:
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())

    def _wait_for_backoff(self, delay: float) -> None:
        """Wait out the retry backoff, aborting early if the task is cancelled."""
        if self._signals.wait_cancelled(delay):
            raise TaskCancelledError("API request cancelled")

    @classmethod
    def close_all_clients(cls) -> None:
        """Close and discard all cached HTTP clients."""
        with cls._clients_lock:
            for key, client in cls._clients.items():
                try:
                    client.close()
                except Exception:
                    logger.debug("Error closing HTTP client for %s", key[0])
            cls._clients.clear()
            logger.debug("All cached HTTP clients closed")

    def cleanup_after_run(self) -> None:
        """Close the connection pool when the runner deactivates this executor."""
        self.close_all_clients()

    def run(self, task: ExecutorTask, out_dir: Path) -> APIResult:
        with self._signals.running(task.task_id):
            return self._run(task, out_dir)

    def _run(self, task: ExecutorTask, out_dir: Path) -> APIResult:
        spec = self.require_spec(task, ApiSpecStrict)
        api_cfg = spec.api or {}
        if not isinstance(api_cfg, dict):
            raise ExecutionError("spec.api must be a mapping")

        url = api_cfg.get("url")
        method = str(api_cfg.get("method", "POST")).upper()
        headers = api_cfg.get("headers", {})
        if not isinstance(headers, dict):
            raise ExecutionError("spec.api.headers must be a mapping")

        carries_deployment_token = False
        if url is None:
            url = os.getenv("NEBULA_API_BASE_URL")
            if not url:
                raise ExecutionError("spec.api.url or NEBULA_API_BASE_URL is required")
            url = url.rstrip("/") + "/v1/chat/completions"

            if not any(is_credential_key(k) for k in headers):
                token = os.getenv("NEBULA_API_TOKEN")
                if not token:
                    raise ExecutionError(
                        "no credential configured: set a credential header or "
                        "NEBULA_API_TOKEN"
                    )
                headers = {
                    name: value
                    for name, value in headers.items()
                    if not _is_routing_header(str(name))
                }
                headers["Authorization"] = f"Bearer {token}"
                carries_deployment_token = True

        params = api_cfg.get("params")
        if params is not None and not isinstance(params, dict):
            raise ExecutionError("spec.api.params must be a mapping")

        timeout_sec = api_cfg.get("timeout_sec", 60)
        if not isinstance(timeout_sec, (int, float)):
            raise ExecutionError("spec.api.timeout_sec must be a number")
        timeout = httpx.Timeout(timeout_sec)

        verify_tls = carries_deployment_token or api_cfg.get("verify_tls", True)
        follow_redirects = api_cfg.get("follow_redirects", True)

        body = api_cfg.get("body")
        json_payload = api_cfg.get("json")
        data_payload = api_cfg.get("data")

        if json_payload is not None and body is not None:
            raise ExecutionError(
                "spec.api.json and spec.api.body are mutually exclusive"
            )

        request_kwargs: dict[str, Any] = {}
        if json_payload is not None:
            request_kwargs["json"] = json_payload
        elif body is not None:
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = body
            else:
                request_kwargs["content"] = body
        elif data_payload is not None:
            request_kwargs["data"] = data_payload

        response_cfg = api_cfg.get("response") or {}
        if response_cfg and not isinstance(response_cfg, dict):
            raise ExecutionError("spec.api.response must be a mapping")

        include_headers = bool(response_cfg.get("include_headers", False))
        # return_body is a JSON backdoor: keep raw text when JSON isn't usable.
        return_body = bool(response_cfg.get("return_body", True))
        parse_json = bool(response_cfg.get("parse_json", True))
        raise_for_status = bool(response_cfg.get("raise_for_status", True))
        max_body_bytes = int(response_cfg.get("max_body_bytes", 200000))

        retries = api_cfg.get("retries", 0)
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ExecutionError("spec.api.retries must be a non-negative integer")
        if retries > _MAX_RETRIES:
            raise ExecutionError(f"spec.api.retries must be at most {_MAX_RETRIES}")

        try:
            base = self._base_url(str(url))
            client = self._get_client(base, timeout, verify_tls, follow_redirects)
            resp = self._request_with_retries(
                client,
                method,
                str(url),
                headers,
                params,
                request_kwargs,
                retries,
            )
        except httpx.RequestError as exc:
            raise ExecutionError(
                redact_urls(f"API request failed: {exc}", str(url)), retryable=True
            ) from exc

        body_bytes = resp.content
        truncated = False
        if max_body_bytes is not None and len(body_bytes) > max_body_bytes:
            body_bytes = body_bytes[:max_body_bytes]
            truncated = True

        result = APIResult(
            ok=resp.is_success,
            executor=self.name,
            method=method,
            url=redact_url(str(resp.url)),
            status_code=resp.status_code,
            truncated=truncated,
        )

        if include_headers:
            result.headers = dict(resp.headers)

        body_text: str | None = None
        if return_body:
            encoding = resp.encoding or "utf-8"
            body_text = body_bytes.decode(encoding, errors="replace")

        if raise_for_status and resp.is_error:
            message = f"API request returned status {resp.status_code}"
            if body_text:
                message = f"{message}: {body_text[:200]}"
            retryable = _is_retryable_status(resp.status_code)
            raise ExecutionError(message, retryable=retryable)

        if parse_json:
            try:
                result.response_json = resp.json()
            except (ValueError, RecursionError) as exc:
                raise ExecutionError("Response is not a valid JSON mapping") from exc
            if not isinstance(result.response_json, dict):
                raise ExecutionError("Response is not a valid JSON mapping")
            usage = result.response_json.get("usage")
            if not isinstance(usage, dict):
                raise ExecutionError(
                    "spec.api.response.parse_json is true but response JSON "
                    f"does not contain usage info: {result.response_json}"
                )
            result.usage = usage
            try:
                result.text = result.response_json["choices"][0]["message"]["content"]
            except Exception as exc:
                raise ExecutionError(
                    "spec.api.response.parse_json is true but response JSON "
                    f"does not contain message.content: {result.response_json}"
                ) from exc
        elif return_body:
            result.text = body_text

        return result
