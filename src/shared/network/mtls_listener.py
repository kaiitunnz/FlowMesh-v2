"""A mutually authenticated relay-frame listener for a directly dialed offload.

A target endpoint accepts only a connection whose certificate chains to the deployment
CA, then reads relay frames from it and answers over the same connection. Membership of
the CA is the whole of the peer check here: it proves a registered worker or node is
dialing, and the replica's claim gate fences the session to the invocation control
admitted. Both target endpoints — the replica worker's claim-gated listener and the
node's purpose-scoped listener — serve that shape and differ only in where a frame goes,
which each supplies as a per-connection handler.

The listener is transport only: it reads a frame's framing and hands the frame on whole.

A dialer that opens a socket but never finishes the handshake holds one slot, so the
handshake is deadlined; the connection cap bounds the sessions accepted past it. A
legitimate connection then idles between frames for as long as its invocation runs, so
reads carry no deadline.

An operator may run a deployment on a trusted network without mutual TLS. That posture
is explicit, warns on every listener it starts, and carries no peer identity, so the
route policy that selected the pair remains the only thing admitting the traffic.
"""

import asyncio
import contextlib
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
from .mtls import MutualTlsMaterial, peer_identities, server_context
from .relay_frame import RelayFrame

HANDSHAKE_TIMEOUT_SEC = 10.0
MAX_CONNECTIONS = 64
# A closed connection whose peer never reads ends at its own transport close, so the
# shutdown waits only briefly for the reads to notice before it stops caring.
SHUTDOWN_TIMEOUT_SEC = 5.0


class ConnectionHandler(Protocol):
    """Where one connection's frames go, and what its close releases."""

    async def on_frame(self, frame: RelayFrame) -> None: ...

    def close(self) -> None: ...


# Builds the handler for one accepted connection, given the sink that answers over it
# and the verified identities its dialer presented (empty without mutual TLS).
ConnectionHandlerFactory = Callable[[FrameSink], ConnectionHandler]

# Whether the verified identities belong to an origin this endpoint admits.
PeerAdmission = Callable[[frozenset[str]], bool]


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
    """Serves relay frames to origins the control plane registered."""

    def __init__(
        self,
        *,
        material: MutualTlsMaterial | None,
        handler: ConnectionHandlerFactory,
        admits: PeerAdmission,
        max_connections: int = MAX_CONNECTIONS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._material = material
        self._handler = handler
        self._admits = admits
        self._max_connections = max_connections
        self._open = 0
        self._logger = logger or logging.getLogger("offload-frame-listener")
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.StreamWriter] = set()

    @property
    def port(self) -> int:
        """The bound port, resolved after start (a configured 0 binds any port)."""
        return self._server.sockets[0].getsockname()[1] if self._server else 0

    async def start_on_socket(self, sock: socket.socket) -> None:
        """Serve a socket bound ahead of time, so its port is known before it serves."""
        context = self._tls_context()
        self._server = await asyncio.start_server(
            self._serve,
            sock=sock,
            ssl=context,
            ssl_handshake_timeout=HANDSHAKE_TIMEOUT_SEC if context else None,
        )

    async def start_on_endpoint(self, endpoint: str) -> None:
        host, port = split_host_port(endpoint)
        context = self._tls_context()
        self._server = await asyncio.start_server(
            self._serve,
            host,
            port,
            family=socket.AF_INET,
            ssl=context,
            ssl_handshake_timeout=HANDSHAKE_TIMEOUT_SEC if context else None,
        )

    def _tls_context(self) -> ssl.SSLContext | None:
        if self._material is None:
            self._logger.warning(
                "serving offload frames without mutual TLS: the deployment is "
                "configured for a trusted network, so a dialer proves no identity"
            )
            return None
        return server_context(self._material)

    async def stop(self) -> None:
        """Close the listener and every connection it is still serving.

        A legitimate connection idles between frames for as long as its invocation runs,
        so waiting for the reads to end on their own would hold a node's shutdown open
        for as long as a peer keeps its socket. Closing them ends those reads.
        """
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        connections, self._connections = self._connections, set()
        for writer in connections:
            with contextlib.suppress(OSError, ssl.SSLError):
                writer.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=SHUTDOWN_TIMEOUT_SEC)
        except (OSError, asyncio.CancelledError, TimeoutError):
            pass

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._open >= self._max_connections:
            self._logger.warning("refusing an offload connection over the cap")
            await close_writer(writer)
            return
        identities = self._peer_identities(writer)
        # Without mutual TLS a dialer presents no identity to check, so the trusted-pair
        # policy that selected the route is the only gate; the start-up warning is where
        # that posture is surfaced. With mutual TLS the identity must be a known origin.
        if self._material is not None and not self._admits(identities):
            self._logger.warning(
                "refusing an offload dialer that is not a known origin"
            )
            await close_writer(writer)
            return
        self._open += 1
        self._connections.add(writer)
        handler = self._handler(ConnectionFrameSink(writer))
        try:
            while True:
                await handler.on_frame(await read_relay_frame(reader))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except FrameStreamError as exc:
            self._logger.warning("closing an offload connection: %s", exc)
        finally:
            self._open -= 1
            self._connections.discard(writer)
            handler.close()
            await close_writer(writer)

    def _peer_identities(self, writer: asyncio.StreamWriter) -> frozenset[str]:
        """The identities the verified peer presented, empty without mutual TLS.

        Mutual TLS has already proved the certificate chains to the configured CA; the
        identity is what proves which registered origin is dialing, since the CA signs
        for every party in the deployment.
        """
        ssl_object = writer.get_extra_info("ssl_object")
        if not isinstance(ssl_object, ssl.SSLObject):
            return frozenset()
        return peer_identities(ssl_object.getpeercert())


__all__ = [
    "HANDSHAKE_TIMEOUT_SEC",
    "MAX_CONNECTIONS",
    "ConnectionFrameSink",
    "ConnectionHandler",
    "ConnectionHandlerFactory",
    "MutualTlsFrameListener",
    "PeerAdmission",
    "close_writer",
]
