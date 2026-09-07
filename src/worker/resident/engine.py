"""The replica worker's call to its co-located engine.

The replica sidecar reaches the serve task running on the same worker over loopback. The
call is non-streaming — it fits a stock vLLM replica and the GPU-free ``dev_model``
stand-in alike — but the content is emitted in bounded pieces so the windowed relay
session flow-controls a large completion rather than framing it whole. A chat replica
returns the assistant message text; an embedding replica returns the ``data`` array of
vectors serialized as JSON — both ride the content path as opaque bytes. An
adapter-bound invocation loads its adapter into a replica slot and selects it as the
request model before the call.
"""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import httpx

from shared.resident.contracts import ReplicaEndpoint
from shared.resident.engine_request import chat_body, embeddings_body

# The already-loaded shapes an engine reports for an idempotent adapter re-load; matched
# narrowly so a precise load error is not swallowed.
_ADAPTER_ALREADY_LOADED = ("already loaded", "already been loaded", "already exists")

# The not-loaded shapes an engine reports for an idempotent unload of an adapter already
# gone; matched narrowly for the same reason.
_ADAPTER_NOT_LOADED = ("not found", "not loaded", "does not exist", "no adapter")


@dataclass
class EngineResponse:
    """An opened engine request: its acknowledgement is implied, its body streams."""

    chunks: AsyncIterator[str]
    aclose: Callable[[], Awaitable[None]]


# Opens the engine request against the replica endpoint and returns once the engine has
# acknowledged it, so the response body can stream under the post-acceptance fence. An
# adapter-bound invocation carries the adapter name and its loadable source.
EngineOpen = Callable[
    [ReplicaEndpoint, str | None, str | None, str | None], Awaitable[EngineResponse]
]

# Unloads a LoRA adapter from a replica slot when its last credit-bearing claim
# releases, so the engine's adapter registry frees a slot with the server accounting.
EngineUnload = Callable[[ReplicaEndpoint, str], Awaitable[None]]


async def unload_adapter(
    endpoint: ReplicaEndpoint, name: str, *, timeout_sec: float = 30.0
) -> None:
    """Unload a LoRA adapter from the co-located engine's slot.

    Called only after the last credit-bearing claim for the adapter on the replica has
    released, so it never unloads an adapter a peer still holds. Idempotent: an adapter
    already gone is not an error.
    """
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    base = endpoint.base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        response = await client.post(
            f"{base}/unload_lora_adapter",
            json={"lora_name": name},
            headers=headers,
        )
        if response.status_code < 400:
            return
        body = response.text.lower()
        if any(phrase in body for phrase in _ADAPTER_NOT_LOADED):
            return
        response.raise_for_status()


class HttpEngineDelivery:
    """Delivers a completion from the co-located OpenAI-compatible engine."""

    def __init__(self, *, timeout_sec: float = 300.0, chunk_chars: int = 8192) -> None:
        self._timeout = timeout_sec
        self._chunk_chars = max(1, chunk_chars)

    async def __call__(
        self,
        endpoint: ReplicaEndpoint,
        request_payload: str | None,
        adapter_name: str | None = None,
        adapter_source: str | None = None,
    ) -> EngineResponse:
        model = adapter_name or endpoint.model
        embedding = endpoint.interface == "embedding"
        if embedding:
            body = embeddings_body(request_payload, model)
            path = "/embeddings"
        else:
            body = chat_body(request_payload, model)
            path = "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        base = endpoint.base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            if adapter_name is not None and adapter_source is not None:
                await self._ensure_adapter(
                    client, base, headers, adapter_name, adapter_source
                )
            response = await client.post(f"{base}{path}", json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
        if embedding:
            content = json.dumps(data["data"])
        else:
            content = str(data["choices"][0]["message"]["content"])
        size = self._chunk_chars

        async def chunks() -> AsyncIterator[str]:
            for start in range(0, len(content), size):
                yield content[start : start + size]

        async def aclose() -> None:
            return None

        return EngineResponse(chunks=chunks(), aclose=aclose)

    @staticmethod
    async def _ensure_adapter(
        client: httpx.AsyncClient,
        base: str,
        headers: dict[str, str],
        name: str,
        source: str,
    ) -> None:
        """Load the LoRA adapter into a replica slot before selecting it on the request.

        Loading is idempotent: an adapter already resident (or one a stand-in pre-loads)
        is not an error. Any other failure raises, so an unloadable adapter fails the
        invocation loudly rather than silently serving the base model.
        """
        response = await client.post(
            f"{base}/load_lora_adapter",
            json={"lora_name": name, "lora_path": source},
            headers=headers,
        )
        if response.status_code < 400:
            return
        body = response.text.lower()
        # _ADAPTER_ALREADY_LOADED is a narrow phrase list coupled to vLLM's own
        # already-loaded error text: matching the specific phrasing rather than a broad
        # "already" swallow keeps a genuine load error (a wrong path, an OOM) loud.
        if any(phrase in body for phrase in _ADAPTER_ALREADY_LOADED):
            return
        response.raise_for_status()
