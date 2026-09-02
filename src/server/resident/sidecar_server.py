"""The resident-facing sidecar listener: the enforced data-plane boundary.

It fronts a resident allocation's engine and admits traffic only through the claim gate.
A connection carries one invocation: a bootstrap frame opens a session under a validated
handoff and initiates the engine request for its enqueue acknowledgement; then, only
under a route authorization the gate accepts, the engine response streams back frame by
frame with the connection's own flow control as backpressure. A cancel frame under a
valid authorization stops the engine stream. A rejected fence is refused before any
response body is forwarded, and the raw engine endpoint is reached only here, never
named on the wire to the deputy.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from . import wire
from .adapter import chat_body
from .sidecar import LoadEvidence, SidecarClaimGate
from .state import AdmissionHandoff, ReplicaEndpoint, RouteAuthorization


@dataclass
class EngineResponse:
    """An opened engine request: its acknowledgement is implied, its body streams."""

    chunks: AsyncIterator[str]
    aclose: Callable[[], Awaitable[None]]


# Opens the engine request against the replica endpoint and returns once the engine has
# acknowledged it, so the response body can stream under the post-acceptance fence.
EngineOpen = Callable[[ReplicaEndpoint, str | None], Awaitable[EngineResponse]]
LoadSink = Callable[[LoadEvidence], None]


class HttpEngineDelivery:
    """Delivers a completion from an OpenAI-compatible engine.

    The engine call is non-streaming — it fits a stock vLLM replica and the GPU-free
    ``dev_model`` stand-in alike — but its content is emitted in bounded pieces so the
    two-phase carriage between the origin deputy and the sidecar flow-controls a large
    completion over the windowed relay instead of framing it as one oversized frame.
    """

    def __init__(
        self,
        *,
        timeout_sec: float = 300.0,
        forward_api_key: str | None = None,
        chunk_chars: int = 8192,
    ) -> None:
        self._timeout = timeout_sec
        self._forward_api_key = forward_api_key
        self._chunk_chars = max(1, chunk_chars)

    async def __call__(
        self, endpoint: ReplicaEndpoint, request_payload: str | None
    ) -> EngineResponse:
        body = chat_body(request_payload, endpoint.model)
        headers = {"Content-Type": "application/json"}
        if api_key := (endpoint.api_key or self._forward_api_key):
            headers["Authorization"] = f"Bearer {api_key}"
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


class ResidentSidecarServer:
    """Serves one resident replica incarnation's claim-gated data-plane connections."""

    def __init__(
        self,
        *,
        gate: SidecarClaimGate,
        endpoint: ReplicaEndpoint,
        engine_open: EngineOpen,
        on_load: LoadSink | None = None,
        stream_deadline_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._gate = gate
        self._endpoint = endpoint
        self._engine_open = engine_open
        self._on_load = on_load or (lambda _ev: None)
        self._stream_deadline = stream_deadline_sec
        self._logger = logger
        # The live serve task per invocation, so a fresh-session re-drive supersedes its
        # prior attempt and exactly one engine runs per invocation under one credit.
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    def _supersede(self, invocation_id: str) -> None:
        current = asyncio.current_task()
        prior = self._inflight.get(invocation_id)
        if prior is not None and prior is not current and not prior.done():
            # A prior attempt for this invocation is still live (its engine running as
            # it awaits a stream that never comes); cancel it so no two engines run.
            prior.cancel()
        if current is not None:
            self._inflight[invocation_id] = current

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        engine: EngineResponse | None = None
        try:
            engine = await self._serve(reader, writer)
        except (
            TimeoutError,
            ValidationError,
            ValueError,
            ConnectionError,
            asyncio.IncompleteReadError,
            OSError,
            httpx.HTTPError,
        ):
            # An engine that refuses or drops the request closes the connection without
            # an ack; the origin deputy reads the close and settles the boundary. An
            # engine transport error is not the caller's fence failure.
            pass
        finally:
            if engine is not None:
                await _quiet(engine.aclose())
            await _close(writer)

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> EngineResponse | None:
        opening = await wire.read_msg(reader)
        if opening["kind"] != wire.KIND_BOOTSTRAP:
            return None
        handoff = AdmissionHandoff.model_validate(opening["handoff"])
        decision = self._gate.check_bootstrap(handoff)
        if not decision.admitted:
            await wire.write_msg(
                writer, wire.KIND_REJECT, reason=str(decision.rejection)
            )
            return None
        session = self._gate.session_for(handoff)
        self._supersede(handoff.invocation_id)
        self._on_load(self._gate.load_evidence(handoff, "request"))
        # Dispatch the engine request and acknowledge immediately: the ack marks engine
        # receipt, not completion, so the origin deputy is not blocked on inference
        # before the control plane can authorize the response stream.
        engine_task: asyncio.Task[EngineResponse] = asyncio.ensure_future(
            self._engine_open(self._endpoint, opening.get("request"))
        )
        await wire.write_msg(writer, wire.KIND_ACK)
        engine: EngineResponse | None = None
        try:
            # Bound the post-ack wait: an acknowledged session whose stream or cancel
            # never arrives is reaped rather than holding the engine open indefinitely.
            follow = await asyncio.wait_for(
                wire.read_msg(reader), timeout=self._stream_deadline
            )
            auth = RouteAuthorization.model_validate(follow["auth"])
            gate = self._gate.check_stream(auth, session)
            if not gate.admitted:
                await wire.write_msg(
                    writer, wire.KIND_REJECT, reason=str(gate.rejection)
                )
                return None
            if follow["kind"] != wire.KIND_STREAM:
                return None
            self._on_load(self._gate.load_evidence(auth, "stream"))
            # Race the engine against a cancel: the origin deputy cancels by closing the
            # connection, so an EOF while the engine still runs aborts the request.
            cancel_watch = asyncio.ensure_future(reader.read(1))
            try:
                await asyncio.wait(
                    {engine_task, cancel_watch},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_watch.done():
                    return None  # the connection closed: the finally aborts the engine
            finally:
                cancel_watch.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await cancel_watch
            try:
                engine = await engine_task
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # A 4xx request error (bar 429) is a definite refusal that held no slot,
                # so the origin deputy releases the credit. A 429 rate-limit or any 5xx
                # is a transient engine condition, carried as uncertain so the boundary
                # holds the credit and re-drives rather than failing fast.
                definite = 400 <= status < 500 and status != 429
                await wire.write_msg(
                    writer,
                    wire.KIND_FAILED,
                    definite=definite,
                    reason=f"engine {status}",
                )
                return None
            async for chunk in engine.chunks:
                await wire.write_msg(writer, wire.KIND_CHUNK, data=chunk)
            await wire.write_msg(writer, wire.KIND_DONE)
            return engine
        finally:
            # Deregister unless a newer attempt has already superseded this one.
            if self._inflight.get(handoff.invocation_id) is asyncio.current_task():
                self._inflight.pop(handoff.invocation_id, None)
            if engine is None:
                engine_task.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await engine_task


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError):
        pass


async def _quiet(coro: Awaitable[None]) -> None:
    try:
        await coro
    except Exception:  # noqa: BLE001 - engine teardown is best-effort
        pass


class ResidentSidecarListener:
    """A per-replica TCP listener bound on the sidecar route."""

    def __init__(self, server: ResidentSidecarServer, *, route: str) -> None:
        self._server = server
        self._host, self._port = wire.split_host_port(route)
        self._tcp: asyncio.Server | None = None
        self._conns: set[asyncio.Task[None]] = set()

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._conns.add(task)
        try:
            await self._server.handle(reader, writer)
        finally:
            if task is not None:
                self._conns.discard(task)

    async def start(self) -> tuple[str, int]:
        self._tcp = await asyncio.start_server(
            self._on_connection, self._host, self._port
        )
        host, port = self._tcp.sockets[0].getsockname()[:2]
        return host, port

    async def stop(self) -> None:
        if self._tcp is None:
            return
        self._tcp.close()
        # Drop any connection still held after its ack so shutdown never blocks on an
        # abandoned session waiting for a stream or cancel that will not come.
        for task in list(self._conns):
            task.cancel()
        if self._conns:
            await asyncio.gather(*self._conns, return_exceptions=True)
        try:
            await self._tcp.wait_closed()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass
        self._tcp = None


__all__ = [
    "EngineResponse",
    "EngineOpen",
    "HttpEngineDelivery",
    "LoadSink",
    "ResidentSidecarListener",
    "ResidentSidecarServer",
]
