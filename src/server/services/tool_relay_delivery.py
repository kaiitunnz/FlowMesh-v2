"""Claim-free reverse-rendezvous delivery for remote external-tool carriage.

The tool ``control_relay`` transport carries one bounded external-tool operation without
either end accepting an inbound connection: the in-server origin and the target worker
each attach outward to the tool root bridge, which bridges opaque framed payloads
between their per-node ``xt:*`` streams. It is a single request / reply — one operation
frame up, one result frame down — so it needs neither the windowed streaming nor the
two-phase bootstrap of the resident path, and it holds no claim, admission, or resident
concept: a frame's payload is the tool-sidecar wire body verbatim, keyed only by the
tool relay session, invocation, and idempotency identifiers used for routing and dedup.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from ..network import wire as netwire
from ..network.reverse_relay import (
    TOOL_RELAY_KEYSPACE,
    BinaryRedis,
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
    RelaySessionStore,
    RelayStreamStore,
)

_Conn = tuple[asyncio.StreamReader, asyncio.StreamWriter]
SidecarConnect = Callable[[str], Awaitable[_Conn]]


async def _default_connect(route: str) -> _Conn:
    host, port = netwire.split_host_port(route)
    return await asyncio.open_connection(host, port)


class ToolRelayEndpoint:
    """Per-node local delivery and the origin-side tool ``control_relay`` driver."""

    def __init__(
        self,
        redis: BinaryRedis,
        node_id: str,
        *,
        connect: SidecarConnect = _default_connect,
        recv_budget_sec: float = 60.0,
        connect_tries: int = 20,
        connect_backoff_sec: float = 0.05,
        logger: logging.Logger | None = None,
    ) -> None:
        self._node_id = node_id
        self._streams = RelayStreamStore(redis, TOOL_RELAY_KEYSPACE)
        self._sessions = RelaySessionStore(redis, TOOL_RELAY_KEYSPACE)
        self._connect = connect
        self._budget = recv_budget_sec
        self._connect_tries = max(1, connect_tries)
        self._connect_backoff = connect_backoff_sec
        self._log = logger or logging.getLogger("tool-relay-endpoint")
        # A ``None`` item is the cancel sentinel that wakes a driver blocked in _recv.
        self._origin: dict[str, asyncio.Queue[RelayFrame | None]] = {}
        self._targets: dict[str, asyncio.Task[None]] = {}
        self._seen: dict[tuple[str, RelayDirection], int] = {}
        # Sessions cancelled by the origin: the routing record is retained (not deleted
        # on the delivery's exit) until the root forwards the CANCEL and the record TTL
        # reaps it, so the cancel stays routable to the target.
        self._cancelled: set[str] = set()

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one down frame by this node's role in the session."""
        record = await self._sessions.load(frame.session_id)
        if not record:
            return
        target_here = record.get("target_node") == self._node_id
        origin_here = record.get("origin_node") == self._node_id
        if self._is_duplicate(frame):
            return
        if frame.direction is RelayDirection.ORIGIN_TO_TARGET and target_here:
            if frame.kind is RelayFrameKind.CANCEL:
                await self.reap_target(frame.session_id)
            else:
                self._serve_target(frame, record.get("sidecar_route", ""))
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

    # ---- target role ----

    def _serve_target(self, frame: RelayFrame, route: str) -> None:
        if not route or frame.session_id in self._targets:
            return
        task = asyncio.ensure_future(self._bridge_once(frame, route))
        self._targets[frame.session_id] = task
        task.add_done_callback(lambda _t: self._on_target_done(frame.session_id))

    def _on_target_done(self, session_id: str) -> None:
        # A completed single request/reply retires the target-runner and its dedup key,
        # so a long-lived target node accrues no per-operation state after the exchange.
        self._targets.pop(session_id, None)
        self._seen.pop((session_id, RelayDirection.ORIGIN_TO_TARGET), None)

    async def _bridge_once(self, frame: RelayFrame, route: str) -> None:
        try:
            reader, writer = await self._connect_sidecar(route)
        except (ConnectionError, OSError):
            # The sidecar never bound (or dropped); abandon so the origin re-drives on a
            # fresh session rather than hang. The origin owns the record and reaps it.
            self._log.warning(
                "tool sidecar connect failed session=%s", frame.session_id
            )
            return
        try:
            await netwire.write_frame(writer, frame.payload)
            # Bound the sidecar reply read: a provider that ignores its own timeout must
            # not hang this target forever — the origin re-drives on a fresh session.
            reply = await asyncio.wait_for(netwire.read_frame(reader), self._budget)
            await self._streams.publish_up(
                self._node_id,
                RelayFrame(
                    kind=RelayFrameKind.DATA,
                    session_id=frame.session_id,
                    invocation_id=frame.invocation_id,
                    idm=frame.idm,
                    direction=RelayDirection.TARGET_TO_ORIGIN,
                    seq=1,
                    payload=reply,
                ),
            )
        except (
            TimeoutError,
            asyncio.IncompleteReadError,
            ConnectionError,
            OSError,
            ValueError,
        ):
            # A mid-exchange loss or a stalled reply leaves the origin waiting; it times
            # out and re-drives on a fresh session.
            pass
        finally:
            with contextlib.suppress(OSError):
                writer.close()
                await writer.wait_closed()

    async def reap_target(self, session_id: str) -> None:
        task = self._targets.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._seen.pop((session_id, RelayDirection.ORIGIN_TO_TARGET), None)

    async def _connect_sidecar(self, route: str) -> _Conn:
        last: Exception | None = None
        for _ in range(self._connect_tries):
            try:
                return await self._connect(route)
            except (ConnectionError, OSError) as exc:
                last = exc
                await asyncio.sleep(self._connect_backoff)
        raise last if last is not None else ConnectionError("sidecar connect failed")

    # ---- origin role ----

    async def deliver(
        self,
        session_id: str,
        *,
        invocation_id: str,
        idm: str,
        origin_node: str,
        target_node: str,
        sidecar_route: str,
        operation_payload: bytes,
        recv_timeout: float | None = None,
    ) -> bytes | None:
        """Send one operation over the rr session and await its single reply payload.

        Returns the reply frame's opaque payload (the tool-sidecar wire body), or
        ``None`` on a lost or cancelled exchange — the origin then leaves the durable
        boundary pending for its idm-* re-drive. ``recv_timeout`` bounds the reply wait
        by the operation deadline; without it the endpoint's own receive budget applies.
        """
        await self._sessions.update(
            session_id,
            invocation_id=invocation_id,
            idm=idm,
            origin_node=origin_node,
            target_node=target_node,
            sidecar_route=sidecar_route,
        )
        self._origin.setdefault(session_id, asyncio.Queue[RelayFrame | None]())
        await self._streams.publish_up(
            self._node_id,
            RelayFrame(
                kind=RelayFrameKind.DATA,
                session_id=session_id,
                invocation_id=invocation_id,
                idm=idm,
                direction=RelayDirection.ORIGIN_TO_TARGET,
                seq=1,
                payload=operation_payload,
            ),
        )
        try:
            return await self._recv(
                session_id, recv_timeout if recv_timeout is not None else self._budget
            )
        finally:
            # Reap the origin state even when the caller cancels this coroutine on its
            # outer timeout, so a cancelled exchange leaves no in-memory session behind.
            await self._retire_origin(session_id)

    async def _recv(self, session_id: str, timeout: float) -> bytes | None:
        queue = self._origin.get(session_id)
        if queue is None:
            return None
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            return None
        return frame.payload if frame is not None else None

    async def cancel(self, session_id: str) -> None:
        """Poke a cancel to the target and drop the origin's wait.

        The routing record is deliberately retained: it must survive until the root
        forwards the CANCEL and the target reaps it, so it is left for the session-store
        TTL to reap rather than deleted here.
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
        self._cancelled.add(session_id)
        queue = self._origin.get(session_id)
        if queue is not None:
            queue.put_nowait(None)
        self._drop_origin_local(session_id)

    def _drop_origin_local(self, session_id: str) -> None:
        self._origin.pop(session_id, None)
        self._seen.pop((session_id, RelayDirection.TARGET_TO_ORIGIN), None)

    async def _retire_origin(self, session_id: str) -> None:
        self._drop_origin_local(session_id)
        if session_id in self._cancelled:
            # A cancelled exchange keeps its record for the CANCEL to reach the target;
            # the TTL reaps it. Deleting it here would strand the cancel mid-route.
            self._cancelled.discard(session_id)
            return
        # Best-effort: a delete fault must not propagate out of a possibly-egressed
        # delivery and become a terminal outcome; the record's TTL reaps it instead.
        with contextlib.suppress(Exception):
            await self._sessions.delete(session_id)


__all__ = ["SidecarConnect", "ToolRelayEndpoint"]
