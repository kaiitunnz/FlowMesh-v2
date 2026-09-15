"""The replica worker's call to its co-located engine.

The replica sidecar reaches the serve task running on the same worker over loopback. A
workflow consumer receives the extracted completion content in bounded pieces so the
windowed relay session flow-controls a large completion rather than framing it whole: a
chat replica returns the assistant message text and an embedding replica the ``data``
array of vectors serialized as JSON, both riding the content path as opaque bytes. A
consumer that carries several conversations on one boundary has each issued as its own
concurrent engine request and receives their texts as a JSON array, so the whole batch
settles as the one invocation that admitted it.

A task-addressed serve request is instead reverse-proxied verbatim in both directions:
the client's method, path, query, end-to-end headers, and raw body reach the engine
unchanged, and its own status, headers, and body bytes come back unchanged, so an
OpenAI-compatible client drives any endpoint the engine serves and a ``stream: true``
body passes through unbuffered. FlowMesh supplies only the upstream host, which no
client header can redirect, and the engine credential. An adapter-bound workflow
invocation loads its adapter into a replica slot and selects it as the request model
before the call.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from shared.resident.contracts import ReplicaEndpoint
from shared.resident.engine_request import (
    batch_chat_bodies,
    chat_body,
    embeddings_body,
)
from shared.resident.envelope import ServeRequestEnvelope, filter_response_headers


def _completion(data: dict[str, Any]) -> str:
    """The assistant message text of one chat response."""
    return str(data["choices"][0]["message"]["content"])


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


@dataclass
class RawEngineResponse:
    """An opened engine request whose raw response is reverse-proxied verbatim.

    ``status`` and ``headers`` are the engine response's own non-hop-by-hop metadata,
    relayed ahead of the body so the client sees the engine's envelope; ``chunks``
    streams the body bytes as they arrive, unbuffered, so a ``stream: true`` response
    passes straight through.
    """

    status: int
    headers: tuple[tuple[str, str], ...]
    chunks: AsyncIterator[bytes]
    aclose: Callable[[], Awaitable[None]]


# Opens the engine request for a task-addressed serve invocation and returns its raw
# streaming response, whose sidecar replays the frozen client envelope verbatim rather
# than constructing a request and extracting completion content.
RawEngineOpen = Callable[
    [ReplicaEndpoint, ServeRequestEnvelope], Awaitable[RawEngineResponse]
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
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _shared_client(self) -> httpx.AsyncClient:
        """The client every invocation shares, so its connections stay warm.

        A replica is called repeatedly over loopback for the life of the lane, and a
        client per call would hand each invocation a cold pool. One client keeps the
        engine connections alive across invocations, and is safe to drive concurrently,
        so the conversations of a batch share it.
        """
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Release the shared client's connections when the lane is reaped."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def __call__(
        self,
        endpoint: ReplicaEndpoint,
        request_payload: str | None,
        adapter_name: str | None = None,
        adapter_source: str | None = None,
    ) -> EngineResponse:
        model = adapter_name or endpoint.model
        embedding = endpoint.interface == "embedding"
        batch = None if embedding else batch_chat_bodies(request_payload, model)
        if embedding:
            bodies = [embeddings_body(request_payload, model)]
            path = "/embeddings"
        else:
            bodies = batch or [chat_body(request_payload, model)]
            path = "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        base = endpoint.base_url.rstrip("/")
        client = await self._shared_client()
        if adapter_name is not None and adapter_source is not None:
            await self._ensure_adapter(
                client, base, headers, adapter_name, adapter_source
            )
        # The conversations of a batch are issued together and concurrently: each is
        # its own engine request, so the engine's continuous batching combines them
        # as it does requests from any other source.
        responses = await self._post_all(client, f"{base}{path}", bodies, headers)
        if embedding:
            content = json.dumps(responses[0]["data"])
        elif batch is not None:
            content = json.dumps([_completion(data) for data in responses])
        else:
            content = _completion(responses[0])
        size = self._chunk_chars

        async def chunks() -> AsyncIterator[str]:
            for start in range(0, len(content), size):
                yield content[start : start + size]

        async def aclose() -> None:
            return None

        return EngineResponse(chunks=chunks(), aclose=aclose)

    @classmethod
    async def _post_all(
        cls,
        client: httpx.AsyncClient,
        url: str,
        bodies: list[dict[str, Any]],
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Issue every conversation concurrently and settle them together.

        The invocation carrying them settles whole, so one refused conversation fails it
        rather than leaving the rest to be abandoned mid-flight. Results keep the order
        the conversations were declared in, whatever order the engine finishes them.
        """
        settled = await asyncio.gather(
            *(cls._post(client, url, body, headers) for body in bodies),
            return_exceptions=True,
        )
        for outcome in settled:
            if isinstance(outcome, BaseException):
                raise outcome
        return cast(list[dict[str, Any]], settled)

    @staticmethod
    async def _post(
        client: httpx.AsyncClient,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

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


class RawHttpEngineDelivery:
    """Reverse-proxies the co-located engine for a task-addressed serve request.

    The frozen client envelope is replayed against the engine unchanged — method, path,
    query, end-to-end headers, and raw body — and the engine's status, headers, and body
    bytes relay back unchanged. The response is never parsed, any status is carried
    through, and a ``stream: true`` body passes unbuffered. FlowMesh supplies only the
    upstream host, which no client header can redirect, and the engine credential.
    """

    def __init__(self, *, timeout_sec: float = 300.0) -> None:
        self._timeout = timeout_sec

    async def __call__(
        self, endpoint: ReplicaEndpoint, envelope: ServeRequestEnvelope
    ) -> RawEngineResponse:
        headers = list(envelope.headers)
        if endpoint.api_key:
            headers.append(("Authorization", f"Bearer {endpoint.api_key}"))
        client = httpx.AsyncClient(timeout=self._timeout)
        try:
            request = client.build_request(
                envelope.method,
                _engine_origin(endpoint.base_url) + envelope.target,
                content=envelope.body,
                headers=headers,
            )
            _strip_injected_headers(request, envelope)
            response = await client.send(request, stream=True)
        except BaseException:
            await client.aclose()
            raise

        async def chunks() -> AsyncIterator[bytes]:
            # The raw stream is not content-decoded, so the body bytes stay consistent
            # with the Content-Encoding and Content-Length the engine reported.
            async for data in response.aiter_raw():
                if data:
                    yield data

        async def aclose() -> None:
            with contextlib.suppress(Exception):
                await response.aclose()
            await client.aclose()

        return RawEngineResponse(
            status=response.status_code,
            headers=filter_response_headers(
                (name.decode("latin-1"), value.decode("latin-1"))
                for name, value in response.headers.raw
            ),
            chunks=chunks(),
            aclose=aclose,
        )


def _strip_injected_headers(
    request: httpx.Request, envelope: ServeRequestEnvelope
) -> None:
    """Drop the conveniences the HTTP client adds that the caller never sent.

    Only ``Host`` and the body framing are FlowMesh's to supply; a header the client did
    not send must not appear upstream just because a client library defaults it.
    """
    sent = {name.lower() for name, _ in envelope.headers}
    for name in ("accept", "accept-encoding", "user-agent", "connection"):
        if name not in sent and name in request.headers:
            del request.headers[name]


def _engine_origin(base_url: str) -> str:
    """The engine's scheme and authority, without the interface path a client carries.

    A serve client addresses the engine's own path (``/v1/models``,
    ``/v1/chat/completions``), so the upstream target is the endpoint's origin joined
    with that path rather than the interface-suffixed base URL a workflow consumer posts
    to.
    """
    split = urlsplit(base_url)
    return f"{split.scheme}://{split.netloc}"
