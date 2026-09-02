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
    ``dev_model`` stand-in alike — and its content is handed to the sidecar as one
    chunk; the two-phase carriage between the origin deputy and the sidecar frames it
    over the authorized channel regardless.
    """

    def __init__(
        self, *, timeout_sec: float = 300.0, forward_api_key: str | None = None
    ) -> None:
        self._timeout = timeout_sec
        self._forward_api_key = forward_api_key

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

        async def chunks() -> AsyncIterator[str]:
            yield content

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
            if follow["kind"] == wire.KIND_CANCEL:
                self._on_load(self._gate.load_evidence(auth, "cancel"))
                await wire.write_msg(writer, wire.KIND_DONE, cancelled=True)
                return None
            if follow["kind"] != wire.KIND_STREAM:
                return None
            self._on_load(self._gate.load_evidence(auth, "stream"))
            try:
                engine = await engine_task
            except httpx.HTTPStatusError as exc:
                # The engine refused with a definite status: no slot is held, so the
                # origin deputy may release the credit rather than hold it.
                await wire.write_msg(
                    writer,
                    wire.KIND_FAILED,
                    definite=True,
                    reason=f"engine {exc.response.status_code}",
                )
                return None
            async for chunk in engine.chunks:
                await wire.write_msg(writer, wire.KIND_CHUNK, data=chunk)
            await wire.write_msg(writer, wire.KIND_DONE)
            return engine
        finally:
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
