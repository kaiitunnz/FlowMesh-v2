"""Locality-neutral engine-invocation adapter for the inference family.

The adapter consumes a claim-bound admission handoff and delivers the request to the
selected replica incarnation, returning the completion. It is the compatibility path
over a stock OpenAI-compatible engine — a vLLM serve replica or the GPU-free
``dev_model`` stand-in — so the server relays the request. The same handoff descriptor
could be consumed by an authenticated worker-side deputy without changing this contract;
a tighter engine-native pre-admission handshake is a later refinement.
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
    """

    def __init__(self, message: str, *, pre_acceptance: bool) -> None:
        super().__init__(message)
        self.pre_acceptance = pre_acceptance


class EngineInvocationAdapter(Protocol):
    """The seam an admission handoff is executed through."""

    async def issue(
        self, handoff: AdmissionHandoff, request_payload: str | None
    ) -> str: ...


def _extract_prompt(payload: str | None) -> str:
    if not payload:
        return ""
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload
    if isinstance(parsed, dict):
        for key in ("prompt", "input", "content"):
            if isinstance(value := parsed.get(key), str):
                return value
    return payload


class HttpInferenceAdapter:
    """Delivers a request to an OpenAI-compatible replica over the server relay."""

    def __init__(
        self,
        *,
        timeout_sec: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_sec
        self._transport = transport

    async def issue(
        self, handoff: AdmissionHandoff, request_payload: str | None
    ) -> str:
        endpoint = handoff.endpoint
        body: dict[str, Any] = {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": _extract_prompt(request_payload)}],
        }
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise AdapterError(str(exc), pre_acceptance=True) from exc
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
