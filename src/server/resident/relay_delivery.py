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
    BinaryRedis,
    DirectionWindow,
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
    responses back to the up stream as target-to-origin frames.

    Each response frame reserves its size against the origin's granted window before it
    is published, so a slow origin backpressures the sidecar rather than flooding the
    relay stream. A cancel closes the connection, which cancels a pump blocked on the
    window, so a full data window never deadlocks cancellation.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        publish: Callable[[RelayFrame], Awaitable[None]],
        frame: RelayFrame,
        window: DirectionWindow,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._publish = publish
        self._frame = frame
        self._window = window
        self._on_closed: Callable[[], Awaitable[None]] | None = None
        self._seq = 0
        self._pump: asyncio.Task[None] | None = None

    def set_on_closed(self, on_closed: Callable[[], Awaitable[None]]) -> None:
        self._on_closed = on_closed

    def start(self) -> None:
        self._pump = asyncio.ensure_future(self._drain())

    async def to_sidecar(self, payload: bytes) -> None:
        await netwire.write_frame(self._writer, payload)

    async def _drain(self) -> None:
        try:
            while True:
                payload = await netwire.read_frame(self._reader)
                self._seq += 1
                await self._window.reserve(len(payload))
                await self._publish(self._response(payload))
        except asyncio.CancelledError:
            return  # a cancel is the reap path, which owns the state cleanup
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError):
            pass
        # The sidecar closed after its terminal (done/failed/reject): reap this node's
        # target-side session state so it does not leak on a normal completion.
        if self._on_closed is not None:
            await self._on_closed()

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
        redis: BinaryRedis,
        node_id: str,
        *,
        connect: SidecarConnect = _default_connect,
        recv_budget_sec: float = 300.0,
        window_bytes: int = 65536,
        connect_tries: int = 20,
        connect_backoff_sec: float = 0.05,
        logger: logging.Logger | None = None,
    ) -> None:
        self._node_id = node_id
        self._streams = RelayStreamStore(redis)
        self._sessions = RelaySessionStore(redis)
        self._connect = connect
        self._budget = recv_budget_sec
        self._window_bytes = window_bytes
        self._connect_tries = max(1, connect_tries)
        self._connect_backoff = connect_backoff_sec
        self._logger = logger or logging.getLogger("resident-relay-endpoint")
        self._origin: dict[str, asyncio.Queue[RelayFrame]] = {}
        self._targets: dict[str, _TargetBridge] = {}
        # In-progress off-pump sidecar connects, so a slow connect never stalls the
        # node's pump loop.
        self._establishing: dict[str, asyncio.Task[None]] = {}
        # Sender-side windows by role: the origin gates its origin-to-target sends, the
        # target gates its target-to-origin responses. Each direction's receiver tracks
        # the cumulative bytes it has drained so its grants advance the sender's window.
        self._o2t_windows: dict[str, DirectionWindow] = {}
        self._t2o_windows: dict[str, DirectionWindow] = {}
        self._o2t_consumed: dict[str, int] = {}
        self._t2o_consumed: dict[str, int] = {}
        # The highest data seq seen per (session, direction), so a bridge re-forward on
        # a crash or mid-batch error never double-delivers to the sidecar/origin.
        self._seen: dict[tuple[str, RelayDirection], int] = {}

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one down frame by this node's role in the session."""
        record = await self._sessions.load(frame.session_id)
        if not record:
            return
        target_here = record.get("target_node") == self._node_id
        origin_here = record.get("origin_node") == self._node_id
        if frame.kind is RelayFrameKind.WINDOW:
            # A window grant advances the sender's flow-control budget only; the sender
            # of a direction is that direction's own node. Grants are idempotent, so a
            # re-forwarded grant is harmless and needs no dedup.
            await self._on_grant(frame, origin_here, target_here)
            return
        if self._is_duplicate(frame):
            # A bridge re-forward (crash or mid-batch retry) re-delivers a frame with
            # a fresh entry id; drop it by per-direction seq so it lands once.
            return
        if frame.direction is RelayDirection.ORIGIN_TO_TARGET and target_here:
            if frame.kind is RelayFrameKind.CANCEL:
                # A cancel closes the sidecar connection so its end-of-stream watch
                # aborts the co-located engine promptly.
                await self.reap_target(frame.session_id)
            else:
                await self._route_to_sidecar(frame, record.get("sidecar_route", ""))
        elif frame.direction is RelayDirection.TARGET_TO_ORIGIN and origin_here:
            queue = self._origin.get(frame.session_id)
            if queue is not None:
                queue.put_nowait(frame)

    def _is_duplicate(self, frame: RelayFrame) -> bool:
        if frame.kind is not RelayFrameKind.DATA or frame.seq == 0:
            return False
        key = (frame.session_id, frame.direction)
        if frame.seq <= self._seen.get(key, 0):
            return True
        self._seen[key] = frame.seq
        return False

    async def _on_target_closed(self, session_id: str, bridge: _TargetBridge) -> None:
        # Reap only if a redrive has not already replaced this session's bridge.
        if self._targets.get(session_id) is bridge:
            self._targets.pop(session_id, None)
            self._t2o_windows.pop(session_id, None)
            self._o2t_consumed.pop(session_id, None)
            self._seen.pop((session_id, RelayDirection.ORIGIN_TO_TARGET), None)

    async def _on_grant(
        self, frame: RelayFrame, origin_here: bool, target_here: bool
    ) -> None:
        # A grant travels back opposite the data it acks (so the root routes it to the
        # sender): a target-to-origin grant advances the origin's send window, and an
        # origin-to-target grant advances the target's. Keying on the return direction
        # also resolves the single-node case, where one node is both origin and target.
        if frame.direction is RelayDirection.TARGET_TO_ORIGIN and origin_here:
            window = self._o2t_windows.get(frame.session_id)
        elif frame.direction is RelayDirection.ORIGIN_TO_TARGET and target_here:
            window = self._t2o_windows.get(frame.session_id)
        else:
            window = None
        if window is not None:
            await window.grant(frame.ack)

    async def _route_to_sidecar(self, frame: RelayFrame, route: str) -> None:
        bridge = self._targets.get(frame.session_id)
        if bridge is not None:
            await self._deliver_to_sidecar(frame, bridge)
            return
        establishing = self._establishing.get(frame.session_id)
        if establishing is not None:
            # A frame arrived while the connection is still being established (rare: the
            # stream follows the ack, which follows a completed establish). Wait, then
            # deliver over the now-bound bridge.
            await establishing
            bridge = self._targets.get(frame.session_id)
            if bridge is not None:
                await self._deliver_to_sidecar(frame, bridge)
            return
        if not route:
            return
        # Establish the sidecar connection off the pump loop, so a slow or never-binding
        # connect backpressures only this session and never stalls the node's stream.
        self._establishing[frame.session_id] = asyncio.ensure_future(
            self._establish_target(frame, route)
        )

    async def _establish_target(self, frame: RelayFrame, route: str) -> None:
        session_id = frame.session_id
        try:
            reader, writer = await self._connect_sidecar(route)
            window = self._t2o_windows.setdefault(
                session_id, DirectionWindow(self._window_bytes)
            )
            bridge = _TargetBridge(reader, writer, self._publish_up, frame, window)
            bridge.set_on_closed(lambda: self._on_target_closed(session_id, bridge))
            self._targets[session_id] = bridge
            bridge.start()
            self._establishing.pop(session_id, None)
            await self._deliver_to_sidecar(frame, bridge)
        except asyncio.CancelledError:
            self._establishing.pop(session_id, None)
        except (ConnectionError, OSError, ValueError):
            # The sidecar never bound (or dropped mid-establish); abandon this attempt
            # so the origin's boundary re-drives on a fresh session rather than hang.
            self._logger.warning("resident sidecar establish failed: %s", session_id)
            self._establishing.pop(session_id, None)
            await self.reap_target(session_id)

    async def _deliver_to_sidecar(
        self, frame: RelayFrame, bridge: _TargetBridge
    ) -> None:
        await bridge.to_sidecar(frame.payload)
        # Having drained an origin-to-target frame to the sidecar, grant the origin its
        # cumulative consumed bytes; the grant returns on the target-to-origin path so
        # the root routes it back to the origin sender.
        consumed = self._o2t_consumed.get(frame.session_id, 0) + len(frame.payload)
        self._o2t_consumed[frame.session_id] = consumed
        await self._emit_grant(frame, RelayDirection.TARGET_TO_ORIGIN, consumed)

    async def _connect_sidecar(self, route: str) -> _Conn:
        # A bootstrap can arrive a beat before the sidecar listener binds/rebinds;
        # retry the loopback connect under a bounded budget so the racing session holds
        # briefly rather than lose the attempt to a needless re-drive.
        last: Exception | None = None
        for _ in range(self._connect_tries):
            try:
                return await self._connect(route)
            except (ConnectionError, OSError) as exc:
                last = exc
                await asyncio.sleep(self._connect_backoff)
        raise last if last is not None else ConnectionError("sidecar connect failed")

    async def _publish_up(self, frame: RelayFrame) -> None:
        await self._streams.publish_up(self._node_id, frame)

    async def _emit_grant(
        self, frame: RelayFrame, direction: RelayDirection, cumulative: int
    ) -> None:
        await self._streams.publish_up(
            self._node_id,
            RelayFrame(
                kind=RelayFrameKind.WINDOW,
                session_id=frame.session_id,
                invocation_id=frame.invocation_id,
                idm=frame.idm,
                direction=direction,
                ack=cumulative,
            ),
        )

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
        self._o2t_windows.setdefault(session_id, DirectionWindow(self._window_bytes))

    async def _send(
        self, session_id: str, invocation_id: str, idm: str, seq: int, payload: bytes
    ) -> None:
        window = self._o2t_windows.get(session_id)
        if window is not None:
            await window.reserve(len(payload))
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
        # Having drained a target-to-origin frame, grant the target its cumulative
        # consumed bytes; the grant returns on the origin-to-target path so the root
        # routes it back to the target sender. A slow origin thus holds no more than its
        # window in flight.
        consumed = self._t2o_consumed.get(session_id, 0) + len(frame.payload)
        self._t2o_consumed[session_id] = consumed
        await self._emit_grant(frame, RelayDirection.ORIGIN_TO_TARGET, consumed)
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
            # An uncertain bootstrap is abandoned: the re-drive mints a fresh session,
            # so reap this one's origin state and record rather than leak it.
            await self.close_origin(session_id)
            return BootstrapResult(False, Transport.CONTROL_RELAY, None, True, [])
        msg = wire.decode_msg(raw)
        if msg["kind"] == wire.KIND_ACK:
            return BootstrapResult(True, Transport.CONTROL_RELAY, None, False, [])
        if msg["kind"] == wire.KIND_REJECT:
            await self.close_origin(session_id)
            return BootstrapResult(
                False, Transport.CONTROL_RELAY, msg.get("reason"), False, []
            )
        await self.close_origin(session_id)
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
                # A stream loss is abandoned: the re-drive mints a fresh session, so
                # reap this one rather than leak its origin state and record.
                await self.close_origin(session_id)
                return StreamResult(False, rejection="stream_loss")
            msg = wire.decode_msg(raw)
            kind = msg["kind"]
            if kind == wire.KIND_CHUNK:
                parts.append(str(msg.get("data", "")))
            elif kind == wire.KIND_DONE:
                await self.close_origin(session_id)
                return StreamResult(True, completion="".join(parts))
            elif kind == wire.KIND_FAILED:
                await self.close_origin(session_id)
                return StreamResult(
                    False,
                    rejection=msg.get("reason"),
                    definite=bool(msg.get("definite")),
                )
            elif kind == wire.KIND_REJECT:
                await self.close_origin(session_id)
                return StreamResult(False, rejection=msg.get("reason"))
            else:
                await self.close_origin(session_id)
                return StreamResult(False, rejection="protocol")

    async def cancel(self, session_id: str) -> None:
        """Poke a cancel to the target to reap the sidecar, and drop the origin.

        The routing record is left in place so the cancel can be bridged to the target;
        the target deletes it once it has reaped the sidecar (the cancel's last use).
        """
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
        self._reap_origin_state(session_id)

    def _reap_origin_state(self, session_id: str) -> None:
        self._origin.pop(session_id, None)
        self._o2t_windows.pop(session_id, None)
        self._t2o_consumed.pop(session_id, None)
        self._seen.pop((session_id, RelayDirection.TARGET_TO_ORIGIN), None)

    async def close_origin(self, session_id: str) -> None:
        # On a stream terminal the origin has the last frame, so reap the origin state
        # and delete the routing record; its stream is then trimmable and does not leak.
        self._reap_origin_state(session_id)
        await self._sessions.delete(session_id)

    async def reap_target(self, session_id: str) -> None:
        establishing = self._establishing.pop(session_id, None)
        if establishing is not None and not establishing.done():
            establishing.cancel()
        bridge = self._targets.pop(session_id, None)
        self._t2o_windows.pop(session_id, None)
        self._o2t_consumed.pop(session_id, None)
        self._seen.pop((session_id, RelayDirection.ORIGIN_TO_TARGET), None)
        # A cancel's last use of the record was routing it here; the target deletes it.
        await self._sessions.delete(session_id)
        if bridge is not None:
            await bridge.close()


__all__ = ["ResidentRelayEndpoint", "SidecarConnect"]
