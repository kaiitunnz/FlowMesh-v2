"""The node's purpose-scoped listener for a directly dialed peer session.

A trusted deployment lets an origin open a connection straight to the node hosting its
target: an admitted invocation's selected replica, or the holder of the content a
transfer reads. The connection's purpose, fixed when it is accepted, picks the local
uplink its frames go to — the resident sidecar's or the content lane's — and the
worker's frames for that session answer over the same connection, so neither direction
enters the rendezvous.

Routing is by the frame's control-minted relay session, which the node resolves against
its own durable routing record: the dialer names only that session, never a host, port,
or engine endpoint. The frames stay opaque here. For a resident session the replica
worker's claim gate is the only authority over what reaches an engine. A content session
is handed on only when control routed it to this node, and, under mutual TLS, only for
a dialer whose certificate covers the host control named for the transfer's requester:
the holder's reply goes back over this connection, so it must be the requester's.

A content connection ends when the holder worker it serves ends on this node, so its
requester falls back at once. A connection that stops reading what it is sent is
dropped, so a stalled dialer never holds up its worker's other traffic.
"""

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from typing import Any

from shared.content.config import DEFAULT_TRANSFER_WINDOW_BYTES
from shared.network.frame_stream import FrameStreamError, PeerPurpose
from shared.network.mtls import MutualTlsMaterial
from shared.network.mtls_listener import (
    MAX_CONNECTIONS,
    ConnectionFrameSink,
    ConnectionHandler,
    MutualTlsFrameListener,
    PeerContext,
    QueuedFrameSink,
)
from shared.network.relay_frame import RelayFrame

from ...network.worker_bridge import RelayWorkerBridge

# Remember far more ended workers than end on one node while a transfer to one can
# still arrive.
_ENDED_MEMORY = 4096


class _UplinkConnection(ConnectionHandler):
    """Binds each session this connection carries to the node's local uplink."""

    def __init__(
        self,
        bridge: RelayWorkerBridge,
        sink: QueuedFrameSink,
        peer: PeerContext,
        on_close: Callable[["_UplinkConnection"], None],
        ended: Callable[[str], bool],
        defer: Callable[[Coroutine[Any, Any, None]], None],
    ) -> None:
        self._bridge = bridge
        self._sink = sink
        self._peer = peer
        self._on_close = on_close
        self._ended = ended
        self._defer = defer
        # Each session's routing record, loaded once when the session is admitted.
        self._sessions: dict[str, dict[str, str]] = {}
        # The workers this connection's content sessions are served by.
        self.targets: set[str] = set()

    async def on_frame(self, frame: RelayFrame) -> None:
        record = self._sessions.get(frame.session_id)
        if record is None:
            record = await self._bridge.session(frame.session_id)
            self._admit(record)
            if record:
                # A record not yet written is looked up again on the next frame.
                self._sessions[frame.session_id] = record
                self._bridge.bind_peer(frame.session_id, self._sink)
        await self._bridge.forward(frame, record)

    def _admit(self, record: dict[str, str]) -> None:
        if self._peer.purpose is not PeerPurpose.CONTENT:
            return
        if record.get("target_node") != self._bridge.node_id:
            raise FrameStreamError("a content session control did not route here")
        if self._peer.identities and not self._peer.names.covers(
            record.get("origin_host") or ""
        ):
            raise FrameStreamError("a content session from a dialer not its requester")
        if target := record.get("target_worker"):
            if self._ended(target):
                raise FrameStreamError("a content session to a worker that has ended")
            self.targets.add(target)

    def end(self) -> None:
        """Drop the connection, ending every session it carries."""
        self._sink.abort()

    def close(self) -> None:
        if self._peer.purpose is PeerPurpose.CONTENT:
            # The holder is still streaming to a requester that can no longer read it:
            # cancel it there, and keep its frames off the root relay.
            for session_id, record in self._sessions.items():
                self._defer(self._bridge.drop_peer(session_id, self._sink, record))
        else:
            for session_id in self._sessions:
                self._bridge.release_peer(session_id, self._sink)
        self._sink.close()
        self._on_close(self)


class NodePeerListener:
    """Serves the node's dialed peer connections into its local uplinks."""

    def __init__(
        self,
        *,
        endpoint: str,
        material: MutualTlsMaterial | None,
        bridges: dict[PeerPurpose, RelayWorkerBridge],
        send_limit_bytes: int = 2 * DEFAULT_TRANSFER_WINDOW_BYTES,
        send_stall_sec: float = 60.0,
        content_idle_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._send_limit_bytes = send_limit_bytes
        self._send_stall_sec = send_stall_sec
        self._logger = logger or logging.getLogger("node-peer-listener")
        self._connections: set[_UplinkConnection] = set()
        # Workers that ended here, newest last; an ended id is never registered again.
        self._ended: OrderedDict[str, None] = OrderedDict()
        self._deferred: set[asyncio.Task[None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listener = MutualTlsFrameListener(
            material=material,
            handler=lambda sink, peer: self._open(bridges[peer.purpose], sink, peer),
            purposes=frozenset(bridges),
            # Each purpose has its own slots, so content load never turns a resident
            # dial away, and an idle content connection has no transfer left to carry.
            max_connections=MAX_CONNECTIONS * len(bridges),
            purpose_limits={purpose: MAX_CONNECTIONS for purpose in bridges},
            idle_timeout_sec={PeerPurpose.CONTENT: content_idle_sec},
            logger=self._logger,
        )

    @property
    def port(self) -> int:
        return self._listener.port

    def _open(
        self, bridge: RelayWorkerBridge, sink: ConnectionFrameSink, peer: PeerContext
    ) -> _UplinkConnection:
        # The worker's frames answer over this connection from the supervisor's
        # event stream for that worker, so a dialer that stops reading must never hold
        # that stream: they are queued, and the dialer is dropped past its bound.
        connection = _UplinkConnection(
            bridge,
            QueuedFrameSink(
                sink,
                limit_bytes=self._send_limit_bytes,
                stall_sec=self._send_stall_sec,
                logger=self._logger,
            ),
            peer,
            on_close=self._connections.discard,
            ended=self._ended.__contains__,
            defer=self._defer,
        )
        if peer.purpose is PeerPurpose.CONTENT:
            self._connections.add(connection)
        return connection

    def _defer(self, step: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(step)
        self._deferred.add(task)
        task.add_done_callback(self._settled)

    def _settled(self, task: asyncio.Task[None]) -> None:
        self._deferred.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            self._logger.warning("could not end a dropped peer session: %s", exc)

    def release_worker(self, worker_id: str) -> None:
        """End every content connection carrying a session to a worker that ended.

        Safe from any thread; the connections end on the listener's own loop.
        """
        if (loop := self._loop) is not None:
            loop.call_soon_threadsafe(self._release, worker_id)

    def admit_worker(self, worker_id: str) -> None:
        """Forget that a worker id ended once a worker registers under it again, as
        after the control store lost its id counter. Safe from any thread."""
        if (loop := self._loop) is not None:
            loop.call_soon_threadsafe(self._ended.pop, worker_id, None)

    def _release(self, worker_id: str) -> None:
        self._ended[worker_id] = None
        while len(self._ended) > _ENDED_MEMORY:
            self._ended.popitem(last=False)
        for connection in list(self._connections):
            if worker_id in connection.targets:
                connection.end()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._listener.start_on_endpoint(self._endpoint)

    async def stop(self) -> None:
        await self._listener.stop()
        deferred = list(self._deferred)
        for task in deferred:
            task.cancel()
        await asyncio.gather(*deferred, return_exceptions=True)


__all__ = ["NodePeerListener"]
