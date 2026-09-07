"""The replica worker's call to its co-located engine.

The replica sidecar reaches the serve task running on the same worker over loopback. The
call is non-streaming — it fits a stock vLLM replica and the GPU-free ``dev_model``
stand-in alike — but the content is emitted in bounded pieces so the windowed relay
session flow-controls a large completion rather than framing it whole.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import httpx

from shared.resident.contracts import ReplicaEndpoint
from shared.resident.engine_request import chat_body


@dataclass
class EngineResponse:
    """An opened engine request: its acknowledgement is implied, its body streams."""

    chunks: AsyncIterator[str]
    aclose: Callable[[], Awaitable[None]]


# Opens the engine request against the replica endpoint and returns once the engine has
# acknowledged it, so the response body can stream under the post-acceptance fence.
EngineOpen = Callable[[ReplicaEndpoint, str | None], Awaitable[EngineResponse]]


class HttpEngineDelivery:
    """Delivers a completion from the co-located OpenAI-compatible engine."""

    def __init__(self, *, timeout_sec: float = 300.0, chunk_chars: int = 8192) -> None:
        self._timeout = timeout_sec
        self._chunk_chars = max(1, chunk_chars)

    async def __call__(
        self, endpoint: ReplicaEndpoint, request_payload: str | None
    ) -> EngineResponse:
        body = chat_body(request_payload, endpoint.model)
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
        content = str(data["choices"][0]["message"]["content"])
        size = self._chunk_chars

        async def chunks() -> AsyncIterator[str]:
            for start in range(0, len(content), size):
                yield content[start : start + size]

        async def aclose() -> None:
            return None

        return EngineResponse(chunks=chunks(), aclose=aclose)
