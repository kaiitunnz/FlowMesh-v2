"""The replica worker's claim-gated resident target-leg listener.

A trusted deployment lets the root open this listener directly instead of carrying the
target leg over the reverse-rendezvous relay. It accepts only a mutually authenticated
connection whose certificate carries the pinned root identity, then feeds each frame it
reads to the replica sidecar and returns the sidecar's frames over the same connection.

The listener is transport only: it never inspects a frame's payload, and the sidecar's
claim gate remains the sole authority over which traffic reaches the engine. It fronts
the sidecar, never the resident engine listener.
"""

import asyncio
import logging
import socket
import ssl
from collections.abc import Awaitable, Callable

from shared.network.frame_stream import (
    FrameStreamError,
    read_relay_frame,
    write_relay_frame,
)
from shared.network.mtls import is_pinned_root
from shared.network.relay_frame import RelayFrame
from shared.resident.transport import ResidentFrameSink

# Routes one frame into the replica sidecar, answering over the connection's own sink.
FrameDelivery = Callable[[RelayFrame, ResidentFrameSink], Awaitable[None]]


class _ConnectionSink(ResidentFrameSink):
    """Returns the sidecar's frames over the connection that delivered the request."""

    def __init__(self, writer: asyncio.StreamWriter, lock: asyncio.Lock) -> None:
        self._writer = writer
        self._lock = lock

    async def send(self, frame: RelayFrame) -> None:
        async with self._lock:
            await write_relay_frame(self._writer, frame)


class ResidentDirectListener:
    """Serves the worker's target-leg connections on the resident lane loop."""

    def __init__(
        self,
        *,
        sock: socket.socket,
        ssl_context: ssl.SSLContext,
        root_identity: str,
        deliver: FrameDelivery,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sock = sock
        self._ssl_context = ssl_context
        self._root_identity = root_identity
        self._deliver = deliver
        self._logger = logger or logging.getLogger("resident-direct-listener")
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._serve, sock=self._sock, ssl=self._ssl_context
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
        if not self._is_root(writer):
            self._logger.warning("refusing a target-leg dialer that is not the root")
            await _close(writer)
            return
        sink = _ConnectionSink(writer, asyncio.Lock())
        try:
            while True:
                frame = await read_relay_frame(reader)
                await self._deliver(frame, sink)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except FrameStreamError as exc:
            self._logger.warning("closing a target-leg connection: %s", exc)
        finally:
            await _close(writer)

    def _is_root(self, writer: asyncio.StreamWriter) -> bool:
        """Whether the verified peer certificate carries the pinned root identity.

        Mutual TLS has already proved the certificate chains to the configured CA; the
        pin is what proves the dialer is the root rather than another holder of a
        CA-signed certificate.
        """
        ssl_object = writer.get_extra_info("ssl_object")
        if not isinstance(ssl_object, ssl.SSLObject):
            return False
        return is_pinned_root(ssl_object.getpeercert(), self._root_identity)


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError, ssl.SSLError):
        pass


__all__ = ["ResidentDirectListener"]
