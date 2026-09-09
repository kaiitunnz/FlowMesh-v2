"""A mutually authenticated relay-frame listener.

A trusted target-leg endpoint accepts only a connection whose certificate chains to the
deployment CA and carries the pinned root identity, then reads relay frames from it and
answers over the same connection. Both target-leg endpoints — the replica worker's
claim-gated listener and the node's purpose-scoped listener — serve that shape and
differ only in where a frame goes, which each supplies as a per-connection handler.

The listener is transport only: it reads a frame's framing and hands the frame on whole.

A dialer that opens a socket but never finishes the handshake holds one slot, so the
handshake is deadlined; the connection cap bounds the sessions accepted past it. A
legitimate leg then idles between frames for as long as its invocation runs, so reads
carry no deadline.
"""

import asyncio
import logging
import socket
import ssl
from collections.abc import Callable
from typing import Protocol

from .frame_stream import (
    FrameSink,
    FrameStreamError,
    read_relay_frame,
    split_host_port,
    write_relay_frame,
)
from .mtls import MutualTlsMaterial, is_pinned_root, server_context
from .relay_frame import RelayFrame

HANDSHAKE_TIMEOUT_SEC = 10.0
MAX_CONNECTIONS = 64


class ConnectionHandler(Protocol):
    """Where one connection's frames go, and what its close releases."""

    async def on_frame(self, frame: RelayFrame) -> None: ...

    def close(self) -> None: ...


# Builds the handler for one accepted connection, given the sink that answers over it.
ConnectionHandlerFactory = Callable[[FrameSink], ConnectionHandler]


class ConnectionFrameSink(FrameSink):
    """Answers over the connection a frame arrived on, one writer at a time."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._lock = asyncio.Lock()

    async def send(self, frame: RelayFrame) -> None:
        async with self._lock:
            await write_relay_frame(self._writer, frame)


async def close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError, ssl.SSLError):
        pass


class MutualTlsFrameListener:
    """Serves relay frames to connections that prove they are the root."""

    def __init__(
        self,
        *,
        material: MutualTlsMaterial,
        handler: ConnectionHandlerFactory,
        max_connections: int = MAX_CONNECTIONS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._material = material
        self._handler = handler
        self._max_connections = max_connections
        self._open = 0
        self._logger = logger or logging.getLogger("mtls-frame-listener")
        self._server: asyncio.Server | None = None

    @property
    def port(self) -> int:
        """The bound port, resolved after start (a configured 0 binds any port)."""
        return self._server.sockets[0].getsockname()[1] if self._server else 0

    async def start_on_socket(self, sock: socket.socket) -> None:
        """Serve a socket bound ahead of time, so its port is known before it serves."""
        self._server = await asyncio.start_server(
            self._serve,
            sock=sock,
            ssl=server_context(self._material),
            ssl_handshake_timeout=HANDSHAKE_TIMEOUT_SEC,
        )

    async def start_on_endpoint(self, endpoint: str) -> None:
        host, port = split_host_port(endpoint)
        self._server = await asyncio.start_server(
            self._serve,
            host,
            port,
            ssl=server_context(self._material),
            ssl_handshake_timeout=HANDSHAKE_TIMEOUT_SEC,
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
        if self._open >= self._max_connections:
            self._logger.warning("refusing a target-leg connection over the cap")
            await close_writer(writer)
            return
        if not self._is_root(writer):
            self._logger.warning("refusing a target-leg dialer that is not the root")
            await close_writer(writer)
            return
        self._open += 1
        handler = self._handler(ConnectionFrameSink(writer))
        try:
            while True:
                await handler.on_frame(await read_relay_frame(reader))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except FrameStreamError as exc:
            self._logger.warning("closing a target-leg connection: %s", exc)
        finally:
            self._open -= 1
            handler.close()
            await close_writer(writer)

    def _is_root(self, writer: asyncio.StreamWriter) -> bool:
        """Whether the verified peer certificate carries the pinned root identity.

        Mutual TLS has already proved the certificate chains to the configured CA; the
        pin is what proves the dialer is the root, whose dialing identity no target is
        given.
        """
        ssl_object = writer.get_extra_info("ssl_object")
        if not isinstance(ssl_object, ssl.SSLObject):
            return False
        return is_pinned_root(ssl_object.getpeercert(), self._material.root_identity)


__all__ = [
    "HANDSHAKE_TIMEOUT_SEC",
    "MAX_CONNECTIONS",
    "ConnectionFrameSink",
    "ConnectionHandler",
    "ConnectionHandlerFactory",
    "MutualTlsFrameListener",
    "close_writer",
]
