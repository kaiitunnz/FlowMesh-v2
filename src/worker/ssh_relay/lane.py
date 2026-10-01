"""Runs this worker's SSH relay lane on a dedicated asyncio loop.

The root's SSH ingresses open one relay session per client connection. Each session's
first message names an endpoint this worker's SSH executor published; the lane resolves
it in the worker's own registry and connects to that loopback port, never to an address
a frame names. An unknown or withdrawn endpoint is refused without a connection.
Produced frames leave as ``SSH_FRAME`` events over the authenticated attachment, which
the supervisor bridges onward opaquely.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shared.network.byte_stream import OPEN, ByteStreamChannel, StreamClosed, splice
from shared.network.frame_stream import WireFrameSink
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.network.session import RelaySessionRole

from .registry import SshEndpointRegistry

LOOPBACK_HOST = "127.0.0.1"
SSH_FRAME_KIND = "ssh_frame"


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
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="flowmesh-ssh-relay", daemon=True
        )

    def start(self) -> None:
        self._registry.add_withdraw_listener(self._on_withdraw)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if not self._thread.is_alive():
            return
        future = asyncio.run_coroutine_threadsafe(self._abort_all(), self._loop)
        with contextlib.suppress(Exception):
            future.result(timeout=timeout)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)

    def route(self, frame_kind: str, frame: dict[str, Any]) -> bool:
        """Marshal one relay frame onto the lane loop; return whether it is ours."""
        if frame_kind != SSH_FRAME_KIND:
            return False
        asyncio.run_coroutine_threadsafe(self._on_frame(frame), self._loop)
        return True

    async def _on_frame(self, wire: dict[str, Any]) -> None:
        frame = RelayFrame.from_wire(wire)
        if frame.direction is not RelayDirection.ORIGIN_TO_TARGET:
            return
        connection = self._connections.get(frame.session_id)
        if connection is None:
            # Only a session's opening frame starts a connection; anything else is
            # left over from one that already ended.
            if frame.kind is not RelayFrameKind.DATA or frame.seq != 1:
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
            if header.get("kind") != OPEN or not endpoint_id:
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
        self._loop.run_forever()


__all__ = ["LOOPBACK_HOST", "SSH_FRAME_KIND", "SshRelayLane"]
