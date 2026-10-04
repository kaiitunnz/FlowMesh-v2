"""This worker's SSH relay lane, on a dedicated asyncio loop.

Each relay session's first message names an endpoint this worker's SSH executor
published; the lane resolves it in the worker's own registry and connects to that
loopback port. An unknown or withdrawn endpoint is refused without a connection.
Produced frames leave as ``SSH_FRAME`` events over the worker's attachment.
"""

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shared.network.byte_stream import (
    ByteStreamChannel,
    StreamClosed,
    StreamMessage,
    splice,
)
from shared.network.frame_stream import WireFrameSink
from shared.network.relay_frame import (
    SSH_FRAME_KIND,
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
)
from shared.network.session import RelaySessionRole
from shared.utils.recent import RecentSet

from .registry import LOOPBACK_HOST, SshEndpointRegistry

# The bridge may re-forward a frame after a restart, so a session's opening frame can
# arrive again once the session has ended or been cancelled; the lane remembers that
# many ended sessions.
_ENDED_MEMORY = 4096
# A stop with its budget spent still gives every connection this long to end.
_MIN_ABORT_SEC = 0.2
# How long a stopped loop waits for its cancelled tasks to unwind.
_DRAIN_SEC = 1.0


@dataclass(eq=False)
class _Connection:
    channel: ByteStreamChannel
    endpoint_id: str | None = None
    task: "asyncio.Task[None] | None" = field(default=None, repr=False)


class SshRelayLane:
    """Serves relayed SSH connections against the endpoints this worker published."""

    def __init__(
        self,
        *,
        registry: SshEndpointRegistry,
        push_frame: Callable[[dict[str, Any]], None],
        open_timeout_sec: float = 30.0,
        connect_timeout_sec: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._sink = WireFrameSink(push_frame)
        self._open_timeout_sec = open_timeout_sec
        self._connect_timeout_sec = connect_timeout_sec
        self._logger = logger or logging.getLogger("ssh-relay-lane")
        self._connections: dict[str, _Connection] = {}
        self._ended: RecentSet[str] = RecentSet(_ENDED_MEMORY)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="flowmesh-ssh-relay", daemon=True
        )

    def start(self) -> None:
        self._registry.add_withdraw_listener(self._on_withdraw)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """End every connection, then the lane, within ``timeout`` seconds."""
        if not self._thread.is_alive():
            return
        deadline = time.monotonic() + timeout
        future = asyncio.run_coroutine_threadsafe(self._abort_all(), self._loop)
        with contextlib.suppress(Exception):
            future.result(timeout=max(_MIN_ABORT_SEC, deadline - time.monotonic()))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def route(self, frame_kind: str, frame: dict[str, Any]) -> bool:
        """Marshal one relay frame onto the lane loop; return whether it is ours."""
        if frame_kind != SSH_FRAME_KIND:
            return False
        handling = self._on_frame(frame)
        try:
            asyncio.run_coroutine_threadsafe(handling, self._loop)
        except RuntimeError:
            # A frame that lands after the lane stopped has no connection to reach.
            handling.close()
        return True

    async def _on_frame(self, wire: dict[str, Any]) -> None:
        frame = RelayFrame.from_wire(wire)
        if frame.direction is not RelayDirection.ORIGIN_TO_TARGET:
            return
        connection = self._connections.get(frame.session_id)
        if connection is None:
            # Only a session's opening frame starts a connection; anything else is
            # left over from one that already ended. The bridge moves a cancel ahead
            # of data, so one can arrive before the opening frame it ends.
            if frame.kind is RelayFrameKind.CANCEL:
                self._ended.add(frame.session_id)
            if (
                frame.kind is not RelayFrameKind.DATA
                or frame.seq != 1
                or frame.session_id in self._ended
            ):
                return
            channel = ByteStreamChannel(
                frame.session_id, RelaySessionRole.TARGET, self._sink
            )
            connection = _Connection(channel)
            self._connections[frame.session_id] = connection
            connection.task = asyncio.ensure_future(self._serve(connection))
        await connection.channel.on_frame(frame)

    async def _serve(self, connection: _Connection) -> None:
        channel = connection.channel
        writer: asyncio.StreamWriter | None = None
        try:
            header, _ = await asyncio.wait_for(
                channel.recv_message(), self._open_timeout_sec
            )
            endpoint_id = str(header.get("endpoint_id") or "")
            if header.get("kind") != StreamMessage.OPEN or not endpoint_id:
                await channel.abort()
                return
            port = self._registry.resolve(endpoint_id)
            if port is None:
                self._logger.warning(
                    "Refusing SSH relay %s: endpoint %s is not published",
                    channel.session_id,
                    endpoint_id,
                )
                await channel.refuse("endpoint not published")
                return
            connection.endpoint_id = endpoint_id
            try:
                reader, socket_writer = await asyncio.wait_for(
                    asyncio.open_connection(LOOPBACK_HOST, port),
                    self._connect_timeout_sec,
                )
            except (OSError, TimeoutError) as exc:
                self._logger.warning(
                    "SSH relay %s could not reach endpoint %s: %s",
                    channel.session_id,
                    endpoint_id,
                    exc,
                )
                await channel.refuse("endpoint unreachable")
                return
            writer = socket_writer
            await splice(channel, reader, socket_writer)
        except (StreamClosed, TimeoutError):
            await channel.abort()
        except Exception:
            self._logger.exception("SSH relay %s failed", channel.session_id)
            await channel.abort()
        finally:
            self._connections.pop(channel.session_id, None)
            self._ended.add(channel.session_id)
            if writer is not None:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

    def _on_withdraw(self, endpoint_id: str) -> None:
        if self._thread.is_alive():
            asyncio.run_coroutine_threadsafe(
                self._end_endpoint(endpoint_id), self._loop
            )

    async def _end_endpoint(self, endpoint_id: str) -> None:
        for connection in list(self._connections.values()):
            if connection.endpoint_id == endpoint_id:
                await self._end(connection)

    async def _abort_all(self) -> None:
        for connection in list(self._connections.values()):
            await self._end(connection)

    @staticmethod
    async def _end(connection: _Connection) -> None:
        await connection.channel.abort()
        if connection.task is not None:
            connection.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await connection.task

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            if tasks := asyncio.all_tasks(self._loop):
                for task in tasks:
                    task.cancel()
                self._loop.run_until_complete(asyncio.wait(tasks, timeout=_DRAIN_SEC))
            self._loop.close()


__all__ = ["SshRelayLane"]
