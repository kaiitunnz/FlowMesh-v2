"""In-server engine-invocation adapter for the inference family.

The adapter delivers a claim-gated request to a selected replica endpoint and returns
the completion. It is the compatibility path over a stock OpenAI-compatible engine — a
vLLM serve replica or the GPU-free ``dev_model`` stand-in — used when the native fabric
path is not in effect, so the server relays the request to the replica's endpoint (read
from the replica directory, never carried on the handoff).
"""

from typing import Protocol

import httpx

from shared.resident.contracts import ReplicaEndpoint
from shared.resident.engine_request import chat_body

__all__ = [
    "AdapterError",
    "EngineInvocationAdapter",
    "HttpInferenceAdapter",
    "chat_body",
]


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
    """The seam a claim-gated request is delivered to a replica endpoint through."""

    async def issue(
        self, endpoint: ReplicaEndpoint, request_payload: str | None
    ) -> str: ...


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
        self, endpoint: ReplicaEndpoint, request_payload: str | None
    ) -> str:
        body = chat_body(request_payload, endpoint.model)
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
