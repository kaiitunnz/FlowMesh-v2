"""Locality-neutral engine-invocation adapter for the inference family.

The adapter consumes a claim-bound admission handoff and delivers the request to the
selected replica incarnation, returning the completion. It is the compatibility path
over a stock OpenAI-compatible engine — a vLLM serve replica or the GPU-free
``dev_model`` stand-in — so the server relays the request. The handoff is locality-
neutral: its descriptor is data an adapter consumes, not a server-owned client, so where
the bytes run is not fixed by this contract.
"""

import json
from typing import Any, Protocol

import httpx

from .state import AdmissionHandoff


class AdapterError(RuntimeError):
    """The engine adapter could not deliver a claim-bound request to its replica.

    ``pre_acceptance`` marks a failure before an engine enqueue acknowledgement — a
    connection or refusal that releases the credit as an enqueue failure — apart from a
    loss after the request was received, which reconciles rather than releasing.
    ``connection_failure`` narrows that further to an unreachable replica (a refused or
    dropped connection), distinct from a transient HTTP status a live replica returned,
    so the caller can invalidate a dead incarnation without nuking a healthy one.
    """

    def __init__(
        self,
        message: str,
        *,
        pre_acceptance: bool,
        connection_failure: bool = False,
    ) -> None:
        super().__init__(message)
        self.pre_acceptance = pre_acceptance
        self.connection_failure = connection_failure


class EngineInvocationAdapter(Protocol):
    """The seam an admission handoff is executed through."""

    async def issue(
        self, handoff: AdmissionHandoff, request_payload: str | None
    ) -> str: ...


def _chat_body(request_payload: str | None, model: str) -> dict[str, Any]:
    """Build the OpenAI chat request from a boundary payload.

    A payload that is already a chat request (a JSON object carrying ``messages``) is
    forwarded faithfully with the replica's model pinned, preserving its system and
    multi-turn messages and sampling parameters; a bare prompt is wrapped as one user
    message. A payload naming a single ``prompt``/``input``/``content`` field is treated
    as that prompt.
    """
    parsed: Any = None
    if request_payload:
        try:
            parsed = json.loads(request_payload)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
        return {**parsed, "model": model}
    if isinstance(parsed, dict):
        for key in ("prompt", "input", "content"):
            if isinstance(value := parsed.get(key), str):
                return {
                    "model": model,
                    "messages": [{"role": "user", "content": value}],
                }
    prompt = request_payload or ""
    return {"model": model, "messages": [{"role": "user", "content": prompt}]}


class HttpInferenceAdapter:
    """Delivers a request to an OpenAI-compatible replica over the server relay."""

    def __init__(
        self,
        *,
        timeout_sec: float = 60.0,
        forward_api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_sec
        self._forward_api_key = forward_api_key
        self._transport = transport

    async def issue(
        self, handoff: AdmissionHandoff, request_payload: str | None
    ) -> str:
        endpoint = handoff.endpoint
        body = _chat_body(request_payload, endpoint.model)
        headers = {"Content-Type": "application/json"}
        # A replica's own key when it reports one; else the deployment forward key the
        # adapter holds out-of-band, so a keyless stand-in can reach a keyed upstream.
        if api_key := (endpoint.api_key or self._forward_api_key):
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise AdapterError(
                str(exc), pre_acceptance=True, connection_failure=True
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise AdapterError(str(exc), pre_acceptance=True) from exc
        except httpx.HTTPError as exc:
            raise AdapterError(str(exc), pre_acceptance=False) from exc
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError(
                f"malformed engine response: {exc}", pre_acceptance=False
            ) from exc
