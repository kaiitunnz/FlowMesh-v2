"""The resident endpoint that bridges the reverse-relay attachment to local I/O.

On each node the attachment hands every down frame to this endpoint, which routes it by
session role: a response toward an invocation this node originated wakes the waiting
origin channel, and a request toward a replica this node targets is bridged to the
co-located sidecar over loopback, its responses published back to the up stream. The
origin side drives the two-phase ``control_relay`` exchange — bootstrap, then the
authorized stream — over the rr-session rather than a dialed socket, carrying the exact
resident wire messages as opaque frame payloads so the sidecar and its claim gate serve
them unchanged.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from ..network import wire as netwire
from ..network.reverse_relay import (
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
    RelaySessionStore,
    RelayStreamStore,
)
from ..network.state import ResolvedRoute, Transport
from . import wire
from .deputy import BootstrapResult, StreamResult
from .state import AdmissionHandoff, RouteAuthorization

# Opens a loopback connection to the co-located sidecar's advertised route.
_Conn = tuple[asyncio.StreamReader, asyncio.StreamWriter]
SidecarConnect = Callable[[str], Awaitable[_Conn]]


async def _default_connect(
    route: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    host, port = wire.split_host_port(route)
    return await asyncio.open_connection(host, port)


class _TargetBridge:
    """Holds a session's loopback connection to the co-located sidecar and pumps its
    responses back to the up stream as target-to-origin frames."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        publish: Callable[[RelayFrame], Awaitable[None]],
        frame: RelayFrame,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._publish = publish
        self._frame = frame
        self._seq = 0
        self._pump: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._pump = asyncio.ensure_future(self._drain())

    async def to_sidecar(self, payload: bytes) -> None:
        await netwire.write_frame(self._writer, payload)

    async def _drain(self) -> None:
        try:
            while True:
                payload = await netwire.read_frame(self._reader)
                self._seq += 1
                await self._publish(self._response(payload))
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError):
            return
        except asyncio.CancelledError:
            return

    def _response(self, payload: bytes) -> RelayFrame:
        return RelayFrame(
            kind=RelayFrameKind.DATA,
            session_id=self._frame.session_id,
            invocation_id=self._frame.invocation_id,
            idm=self._frame.idm,
            direction=RelayDirection.TARGET_TO_ORIGIN,
            seq=self._seq,
            payload=payload,
        )

    async def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
        with contextlib.suppress(OSError):
            self._writer.close()
            await self._writer.wait_closed()


class ResidentRelayEndpoint:
    """Per-node local delivery and the origin-side control_relay driver."""

    def __init__(
        self,
        redis,  # noqa: ANN001 - a network.reverse_relay.BinaryRedis
        node_id: str,
        *,
        connect: SidecarConnect = _default_connect,
        recv_budget_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._node_id = node_id
        self._streams = RelayStreamStore(redis)
        self._sessions = RelaySessionStore(redis)
        self._connect = connect
        self._budget = recv_budget_sec
        self._logger = logger or logging.getLogger("resident-relay-endpoint")
        self._origin: dict[str, asyncio.Queue[RelayFrame]] = {}
        self._targets: dict[str, _TargetBridge] = {}

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one down frame by this node's role in the session."""
        record = await self._sessions.load(frame.session_id)
        if not record:
            return
        target_here = record.get("target_node") == self._node_id
        origin_here = record.get("origin_node") == self._node_id
        if frame.direction is RelayDirection.ORIGIN_TO_TARGET and target_here:
            if frame.kind is RelayFrameKind.CANCEL:
                # A cancel closes the sidecar connection so its end-of-stream watch
                # aborts the co-located engine promptly.
                await self.reap_target(frame.session_id)
            else:
                await self._to_sidecar(frame, record.get("sidecar_route", ""))
        elif frame.direction is RelayDirection.TARGET_TO_ORIGIN and origin_here:
            queue = self._origin.get(frame.session_id)
            if queue is not None:
                queue.put_nowait(frame)

    async def _to_sidecar(self, frame: RelayFrame, route: str) -> None:
        bridge = self._targets.get(frame.session_id)
        if bridge is None:
            if not route:
                return
            reader, writer = await self._connect(route)
            bridge = _TargetBridge(reader, writer, self._publish_up, frame)
            self._targets[frame.session_id] = bridge
            bridge.start()
        await bridge.to_sidecar(frame.payload)

    async def _publish_up(self, frame: RelayFrame) -> None:
        await self._streams.publish_up(self._node_id, frame)

    # ---- origin control_relay driver ----

    async def open_origin(
        self,
        session_id: str,
        *,
        invocation_id: str,
        idm: str,
        origin_node: str,
        target_node: str,
        sidecar_route: str,
    ) -> None:
        await self._sessions.update(
            session_id,
            invocation_id=invocation_id,
            idm=idm,
            origin_node=origin_node,
            target_node=target_node,
            sidecar_route=sidecar_route,
        )
        self._origin.setdefault(session_id, asyncio.Queue())

    async def _send(
        self, session_id: str, invocation_id: str, idm: str, seq: int, payload: bytes
    ) -> None:
        await self._streams.publish_up(
            self._node_id,
            RelayFrame(
                kind=RelayFrameKind.DATA,
                session_id=session_id,
                invocation_id=invocation_id,
                idm=idm,
                direction=RelayDirection.ORIGIN_TO_TARGET,
                seq=seq,
                payload=payload,
            ),
        )

    async def _recv(self, session_id: str, timeout: float) -> bytes | None:
        queue = self._origin.get(session_id)
        if queue is None:
            return None
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            return None
        return frame.payload

    async def bootstrap(
        self,
        session_id: str,
        *,
        route: ResolvedRoute,
        handoff: AdmissionHandoff,
        request_payload: str | None,
    ) -> BootstrapResult:
        """Deliver the bootstrap over the rr-session and await the enqueue ack."""
        candidate = route.candidates[0]
        origin_hop, target_hop = candidate.hops
        idm = handoff.idempotency_key or ""
        await self.open_origin(
            session_id,
            invocation_id=handoff.invocation_id,
            idm=idm,
            origin_node=origin_hop.node_id or "",
            target_node=target_hop.node_id or "",
            sidecar_route=target_hop.endpoint,
        )
        await self._send(
            session_id,
            handoff.invocation_id,
            idm,
            1,
            wire.encode_msg(
                wire.KIND_BOOTSTRAP,
                handoff=handoff.model_dump(mode="json"),
                request=request_payload,
            ),
        )
        raw = await self._recv(session_id, self._budget)
        if raw is None:
            return BootstrapResult(False, Transport.CONTROL_RELAY, None, True, [])
        msg = wire.decode_msg(raw)
        if msg["kind"] == wire.KIND_ACK:
            return BootstrapResult(True, Transport.CONTROL_RELAY, None, False, [])
        if msg["kind"] == wire.KIND_REJECT:
            return BootstrapResult(
                False, Transport.CONTROL_RELAY, msg.get("reason"), False, []
            )
        return BootstrapResult(False, Transport.CONTROL_RELAY, "protocol", False, [])

    async def stream(self, session_id: str, auth: RouteAuthorization) -> StreamResult:
        """Present the authorization and assemble the streamed completion over rr."""
        await self._send(
            session_id,
            auth.invocation_id,
            auth.idempotency_key or "",
            2,
            wire.encode_msg(wire.KIND_STREAM, auth=auth.model_dump(mode="json")),
        )
        parts: list[str] = []
        while True:
            raw = await self._recv(session_id, self._budget)
            if raw is None:
                return StreamResult(False, rejection="stream_loss")
            msg = wire.decode_msg(raw)
            kind = msg["kind"]
            if kind == wire.KIND_CHUNK:
                parts.append(str(msg.get("data", "")))
            elif kind == wire.KIND_DONE:
                self.close_origin(session_id)
                return StreamResult(True, completion="".join(parts))
            elif kind == wire.KIND_FAILED:
                self.close_origin(session_id)
                return StreamResult(
                    False,
                    rejection=msg.get("reason"),
                    definite=bool(msg.get("definite")),
                )
            elif kind == wire.KIND_REJECT:
                self.close_origin(session_id)
                return StreamResult(False, rejection=msg.get("reason"))
            else:
                self.close_origin(session_id)
                return StreamResult(False, rejection="protocol")

    async def cancel(self, session_id: str) -> None:
        """Poke a cancel to the target to reap the sidecar, and drop the origin."""
        record = await self._sessions.load(session_id)
        if record:
            await self._streams.publish_up(
                self._node_id,
                RelayFrame(
                    kind=RelayFrameKind.CANCEL,
                    session_id=session_id,
                    invocation_id=record.get("invocation_id", ""),
                    idm=record.get("idm", ""),
                    direction=RelayDirection.ORIGIN_TO_TARGET,
                ),
            )
        self.close_origin(session_id)

    def close_origin(self, session_id: str) -> None:
        self._origin.pop(session_id, None)

    async def reap_target(self, session_id: str) -> None:
        bridge = self._targets.pop(session_id, None)
        if bridge is not None:
            await bridge.close()


__all__ = ["ResidentRelayEndpoint", "SidecarConnect"]
