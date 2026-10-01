"""The root's SSH relay origin: carries each ingress connection over ``control_relay``.

Both SSH ingresses — the WebSocket proxy and the forward listener — open a relay
session here per client connection. The origin rides one dedicated reverse-relay
stream id, not a worker node: it publishes its frames on that stream's up leg and
consumes its down leg through a root-internal attachment, and the root bridge pump
forwards both directions by the session record. Each session names its target only
by worker and the endpoint id the worker's SSH executor published; the worker
resolves the port itself. Root and supervisors carry the bytes opaquely.
"""

import asyncio
import contextlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

from shared.network.byte_stream import WINDOW_BYTES, ByteStreamChannel
from shared.network.frame_stream import FrameSink
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.network.session import RelaySessionRole
from shared.utils.ids import new_relay_session_id
from shared.utils.json import safe_get

from ..network.reverse_relay import (
    SSH_RELAY_KEYSPACE,
    BinaryRedis,
    RelaySessionStore,
    RelayStreamStore,
)
from ..registries.worker import WorkerRegistry
from ..supervisor.services.reverse_relay_attachment import ReverseRelayAttachment
from ..task.models import TaskRecord, TaskStatus

SSH_EDGE_STREAM_ID = "ssh-edge"
RELAYED_MODES = frozenset({"proxy", "forward"})

# The bridge routes a session's last frames — an abort's cancel, a close's final
# window grants — by its record, so the record outlives the session by this much.
_ENDED_RECORD_TTL_MS = 60_000


@dataclass(frozen=True)
class SshRelayTarget:
    """Where one relayed connection goes: a worker and the endpoint it published."""

    task_id: str
    worker_id: str
    node_id: str
    endpoint_id: str


class SshRelayUnavailable(Exception):
    """The task has no session a relay can reach right now."""


async def resolve_relay_target(
    record: TaskRecord, workers: WorkerRegistry
) -> SshRelayTarget:
    """Return where a connection to ``record``'s session is relayed.

    Only a running task whose session is published in a relayed mode, on a worker
    still registered, has one.
    """
    if record.status != TaskStatus.DISPATCHED:
        raise SshRelayUnavailable("the task is not running")
    ssh_info = safe_get(record.latest_update, "ssh")
    if not isinstance(ssh_info, dict) or ssh_info.get("mode") not in RELAYED_MODES:
        raise SshRelayUnavailable("the task has no relayed SSH session")
    if not (endpoint_id := ssh_info.get("session_id")):
        raise SshRelayUnavailable("the session published no id")
    if not (worker_id := record.assigned_worker):
        raise SshRelayUnavailable("the task has no assigned worker")
    worker = await workers.get_worker_async(worker_id)
    if worker is None:
        raise SshRelayUnavailable(f"worker {worker_id} is not registered")
    return SshRelayTarget(
        task_id=record.task_id,
        worker_id=worker.id,
        node_id=worker.node_id,
        endpoint_id=str(endpoint_id),
    )


class _EdgeSink(FrameSink):
    def __init__(self, streams: RelayStreamStore) -> None:
        self._streams = streams

    async def send(self, frame: RelayFrame) -> None:
        await self._streams.publish_up(SSH_EDGE_STREAM_ID, frame)


class SshRelayOrigin:
    """Opens and tracks the root end of every relayed SSH connection."""

    def __init__(
        self,
        relay_redis: BinaryRedis,
        *,
        window_bytes: int = WINDOW_BYTES,
        refresh_interval_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = RelayStreamStore(relay_redis, SSH_RELAY_KEYSPACE)
        self._sessions = RelaySessionStore(relay_redis, SSH_RELAY_KEYSPACE)
        self._sink = _EdgeSink(self._streams)
        self._window_bytes = window_bytes
        self._refresh_interval_sec = refresh_interval_sec
        self._logger = logger or logging.getLogger("ssh-relay-origin")
        self._attachment = ReverseRelayAttachment(
            relay_redis,
            SSH_EDGE_STREAM_ID,
            self,
            owner=f"{SSH_EDGE_STREAM_ID}:{os.getpid()}",
            keyspace=SSH_RELAY_KEYSPACE,
            logger=self._logger,
        )
        self._channels: dict[str, ByteStreamChannel] = {}
        self._by_task: defaultdict[str, set[str]] = defaultdict(set)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._refresher: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Reap what a previous root left open, then begin consuming the edge stream."""
        self._loop = asyncio.get_running_loop()
        await self.reap_orphans()
        self._attachment.start(self._loop)
        self._refresher = asyncio.create_task(self._refresh_records())

    async def stop(self) -> None:
        """End every live connection at its worker, then stop consuming the edge.

        The root's bridge stops with it, so each cancel goes straight to its target's
        down stream, and each record keeps its TTL for the next root to reap.
        """
        if self._refresher is not None:
            self._refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresher
        channels, self._channels = self._channels, {}
        self._by_task.clear()
        for session_id, channel in channels.items():
            try:
                await self._cancel_at_target(
                    session_id, await self._sessions.load(session_id)
                )
                await channel.abort()
            except Exception:
                self._logger.warning(
                    "Failed to end SSH relay session %s", session_id, exc_info=True
                )
        await self._attachment.stop()

    async def reap_orphans(self) -> int:
        """Cancel every session a previous root opened; return how many.

        The connections died with that root's process, but their worker ends would
        hold the session's sshd connection open — and its idle timer from firing —
        until the task ended.
        """
        reaped = 0
        for session_id in await self._sessions.list_ids():
            if session_id in self._channels:
                continue
            record = await self._sessions.load(session_id)
            if not await self._cancel_at_target(session_id, record):
                continue
            await self._sessions.touch(session_id, _ENDED_RECORD_TTL_MS)
            reaped += 1
        if reaped:
            self._logger.info("Reaped %d SSH relay sessions left open", reaped)
        return reaped

    async def _cancel_at_target(self, session_id: str, record: dict[str, str]) -> bool:
        """Publish a cancel straight to the target of an edge session; return whether
        ``record`` is one."""
        if record.get("origin_node") != SSH_EDGE_STREAM_ID:
            return False
        if target_node := record.get("target_node"):
            await self._streams.publish_down(
                target_node,
                RelayFrame(
                    kind=RelayFrameKind.CANCEL,
                    session_id=session_id,
                    direction=RelayDirection.ORIGIN_TO_TARGET,
                ),
            )
        return True

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one frame from the edge stream to its connection."""
        if (channel := self._channels.get(frame.session_id)) is not None:
            await channel.on_frame(frame)

    async def open(self, target: SshRelayTarget) -> ByteStreamChannel:
        """Open a relay session to ``target`` and send its opening message."""
        session_id = new_relay_session_id()
        await self._sessions.update(
            session_id,
            origin_node=SSH_EDGE_STREAM_ID,
            origin_worker="",
            target_node=target.node_id,
            target_worker=target.worker_id,
            task_id=target.task_id,
        )
        channel = ByteStreamChannel(
            session_id,
            RelaySessionRole.ORIGIN,
            self._sink,
            window_bytes=self._window_bytes,
        )
        self._channels[session_id] = channel
        self._by_task[target.task_id].add(session_id)
        try:
            await channel.send_open(target.endpoint_id)
        except BaseException:
            await self.release(channel, abort=True)
            raise
        return channel

    async def release(self, channel: ByteStreamChannel, *, abort: bool) -> None:
        """Forget a connection once it ends, aborting it at both ends if asked."""
        session_id = channel.session_id
        if self._channels.pop(session_id, None) is None:
            return
        for task_id, ids in list(self._by_task.items()):
            ids.discard(session_id)
            if not ids:
                del self._by_task[task_id]
        with contextlib.suppress(Exception):
            if abort:
                await channel.abort()
            await self._sessions.touch(session_id, _ENDED_RECORD_TTL_MS)

    def close_task(self, task_id: str) -> None:
        """Abort every connection relayed to ``task_id``; callable from any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._close_task(task_id), loop)

    async def _close_task(self, task_id: str) -> None:
        for session_id in list(self._by_task.get(task_id, ())):
            await self._abort(session_id)

    async def _abort(self, session_id: str) -> None:
        if (channel := self._channels.get(session_id)) is not None:
            await self.release(channel, abort=True)

    async def _refresh_records(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_interval_sec)
            for session_id in list(self._channels):
                try:
                    await self._sessions.touch(session_id)
                except Exception:
                    self._logger.debug(
                        "Failed to refresh SSH relay session %s",
                        session_id,
                        exc_info=True,
                    )


__all__ = [
    "RELAYED_MODES",
    "SSH_EDGE_STREAM_ID",
    "SshRelayOrigin",
    "SshRelayTarget",
    "SshRelayUnavailable",
    "resolve_relay_target",
]
