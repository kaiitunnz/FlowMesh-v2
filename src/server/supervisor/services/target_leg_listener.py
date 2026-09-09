"""The node's purpose-scoped resident target-leg listener.

A trusted deployment lets the root open a resident invocation's target leg directly to
the node hosting the selected replica instead of carrying it over the reverse-rendezvous
relay. This listener is that entry: it accepts only a mutually authenticated connection
whose certificate carries the pinned root identity, then hands each frame it reads to
the node's local sidecar uplink and returns the worker's frames over the same
connection.

It is not a general TCP proxy. A frame names only its control-minted relay session, and
the node resolves that session's local worker from its own durable routing record — the
dialer supplies no host, port, or engine endpoint. The frames stay opaque here: the
replica worker's claim gate is the only authority over what reaches an engine.
"""

import asyncio
import logging
import socket
import ssl

from shared.network.frame_stream import (
    FrameStreamError,
    read_relay_frame,
    split_host_port,
    write_relay_frame,
)
from shared.network.mtls import MutualTlsMaterial, is_pinned_root, server_context
from shared.network.relay_frame import RelayFrame
from shared.resident.transport import ResidentFrameSink

from ...resident.worker_bridge import ResidentWorkerBridge


class _ConnectionSink(ResidentFrameSink):
    """Returns a session's worker frames over the connection the root opened."""

    def __init__(self, writer: asyncio.StreamWriter, lock: asyncio.Lock) -> None:
        self._writer = writer
        self._lock = lock

    async def send(self, frame: RelayFrame) -> None:
        async with self._lock:
            await write_relay_frame(self._writer, frame)


class NodeTargetLegListener:
    """Serves the node's target-leg connections into its local sidecar uplink."""

    def __init__(
        self,
        *,
        endpoint: str,
        material: MutualTlsMaterial,
        bridge: ResidentWorkerBridge,
        logger: logging.Logger | None = None,
    ) -> None:
        self._host, self._port = split_host_port(endpoint)
        self._material = material
        self._bridge = bridge
        self._logger = logger or logging.getLogger("node-target-leg-listener")
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._serve,
            self._host,
            self._port,
            ssl=server_context(self._material),
            family=socket.AF_INET,
        )

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        try:
            await server.wait_closed()
        except (OSError, asyncio.CancelledError):
            pass

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        ssl_object = writer.get_extra_info("ssl_object")
        if not isinstance(ssl_object, ssl.SSLObject) or not is_pinned_root(
            ssl_object.getpeercert(), self._material.root_identity
        ):
            self._logger.warning("refusing a target-leg dialer that is not the root")
            await _close(writer)
            return
        sink = _ConnectionSink(writer, asyncio.Lock())
        sessions: set[str] = set()
        try:
            while True:
                frame = await read_relay_frame(reader)
                if frame.session_id not in sessions:
                    sessions.add(frame.session_id)
                    self._bridge.bind_target_leg(frame.session_id, sink)
                await self._bridge.on_frame(frame)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except FrameStreamError as exc:
            self._logger.warning("closing a target-leg connection: %s", exc)
        finally:
            for session_id in sessions:
                self._bridge.release_target_leg(session_id)
            await _close(writer)


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError, ssl.SSLError):
        pass


__all__ = ["NodeTargetLegListener"]
