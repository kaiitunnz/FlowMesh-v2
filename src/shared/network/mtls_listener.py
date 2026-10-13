"""A mutually authenticated relay-frame listener for a directly dialed peer session.

A target endpoint accepts only a connection whose certificate chains to the deployment
CA, then reads relay frames from it and answers over the same connection. Membership of
the CA is the whole of the peer check here: it proves a registered worker or node is
dialing, and what the connection may carry is the handler's decision. Every target
endpoint — the replica worker's claim-gated listener, a worker's content listener and
the node's listener — serves that shape and differs only in where a frame goes, which
each supplies as a per-connection handler given the dialer's verified names and the
connection's purpose.

The listener is transport only: it reads a frame's framing and hands the frame on whole.
It answers two exchanges itself. The target accepts a dialer's connection before the
dialer sends a session on it: a request is answered ``accepted``, echoing its purpose,
while a connection over the cap, from an origin it does not admit, or for a purpose the
listener does not serve is answered ``busy``, ``refused`` or ``unserved`` and closed, so
the dialer learns of a refusal before it writes anything of the session. A dialer that
sends a relay frame without an accept carries a resident session.
A probe opening a connection is echoed and closed, so a reachability check covers the
handshake, origin check and framing a session uses without reaching a handler.

A dialer that opens a socket but never finishes the handshake holds one slot, so the
handshake is deadlined, and so is the opening of a session: its first frame, and after
an accept the first relay frame, which every dialer sends at once. The connection cap
bounds the sessions accepted past that. A legitimate connection then idles between
frames for as long as its invocation runs, so later reads carry no deadline.

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
from dataclasses import dataclass
from typing import Any, Protocol

from .frame_stream import (
    AcceptFrame,
    AcceptStatus,
    FrameSink,
    FrameStreamError,
    PeerPurpose,
    ProbeFrame,
    read_relay_frame,
    read_stream_frame,
    split_host_port,
    write_accept,
    write_probe,
    write_relay_frame,
)
from .mtls import MutualTlsMaterial, PeerNames, peer_identities, server_context
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


@dataclass(frozen=True)
class PeerContext:
    """What a listener verified about one accepted connection.

    ``identities`` and ``names`` are empty without mutual TLS: the dialer then proves
    nothing about who it is.
    """

    purpose: PeerPurpose
    identities: frozenset[str] = frozenset()
    names: PeerNames = PeerNames()


# Builds the handler for one accepted connection, given the sink that answers over it
# and what the listener verified about the connection.
ConnectionHandlerFactory = Callable[[FrameSink, PeerContext], ConnectionHandler]

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

    def abort(self) -> None:
        """Drop the connection at once, so its dialer sees it end."""
        self._writer.transport.abort()


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
        purposes: frozenset[PeerPurpose] = frozenset({PeerPurpose.RESIDENT}),
        max_connections: int = MAX_CONNECTIONS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._material = material
        self._handler = handler
        self._admits = admits
        self._purposes = purposes
        self._max_connections = max_connections
        self._open = 0
        self._logger = logger or logging.getLogger("peer-frame-listener")
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
                "serving peer frames without mutual TLS: the deployment is "
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
            self._logger.warning("refusing a peer connection over the cap")
            await self._decline(writer, AcceptStatus.BUSY)
            return
        peer_cert = self._peer_cert(writer)
        identities = peer_identities(peer_cert)
        # Without mutual TLS a dialer presents no identity to check, so the trusted-pair
        # policy that selected the route is the only gate; the start-up warning is where
        # that posture is surfaced. With mutual TLS the identity must be a known origin.
        if self._material is not None and not self._admits(identities):
            self._logger.warning("refusing a peer dialer that is not a known origin")
            await self._decline(writer, AcceptStatus.REFUSED)
            return
        self._open += 1
        self._connections.add(writer)
        handler: ConnectionHandler | None = None
        try:
            first = await asyncio.wait_for(
                read_stream_frame(reader), HANDSHAKE_TIMEOUT_SEC
            )
            if isinstance(first, ProbeFrame):
                await write_probe(writer, first.payload)
                writer.close()
                return
            purpose = PeerPurpose.RESIDENT
            if isinstance(first, AcceptFrame):
                if first.status is not AcceptStatus.REQUEST:
                    raise FrameStreamError("a dialer sent an accept answer")
                requested = _purpose_of(first.purpose)
                if requested is None or requested not in self._purposes:
                    self._logger.warning(
                        "refusing a peer connection for an unserved purpose"
                    )
                    await write_accept(writer, AcceptStatus.UNSERVED, first.purpose)
                    return
                purpose = requested
                await write_accept(writer, AcceptStatus.ACCEPTED, purpose)
                first = await asyncio.wait_for(
                    read_relay_frame(reader), HANDSHAKE_TIMEOUT_SEC
                )
            elif purpose not in self._purposes:
                self._logger.warning(
                    "closing a resident session on a listener that serves none"
                )
                return
            handler = self._handler(
                ConnectionFrameSink(writer),
                PeerContext(purpose, identities, PeerNames.of(peer_cert)),
            )
            await handler.on_frame(first)
            while True:
                await handler.on_frame(await read_relay_frame(reader))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except FrameStreamError as exc:
            self._logger.warning("closing a peer connection: %s", exc)
        finally:
            self._open -= 1
            self._connections.discard(writer)
            if handler is not None:
                handler.close()
            await close_writer(writer)

    async def _decline(
        self, writer: asyncio.StreamWriter, status: AcceptStatus
    ) -> None:
        """Answer a connection the listener will not serve, without reading from it."""
        with contextlib.suppress(OSError, ssl.SSLError):
            await write_accept(writer, status)
        await close_writer(writer)

    def _peer_cert(self, writer: asyncio.StreamWriter) -> dict[str, Any] | None:
        """The verified peer's certificate, None without mutual TLS.

        Mutual TLS has already proved the certificate chains to the configured CA; its
        names are what prove which registered origin is dialing, since the CA signs for
        every party in the deployment.
        """
        ssl_object = writer.get_extra_info("ssl_object")
        if not isinstance(ssl_object, ssl.SSLObject):
            return None
        return ssl_object.getpeercert()


def _purpose_of(raw: str | None) -> PeerPurpose | None:
    """The purpose an accept request names; absent is resident, unknown is None."""
    if raw is None:
        return PeerPurpose.RESIDENT
    try:
        return PeerPurpose(raw)
    except ValueError:
        return None


__all__ = [
    "HANDSHAKE_TIMEOUT_SEC",
    "MAX_CONNECTIONS",
    "ConnectionFrameSink",
    "ConnectionHandler",
    "ConnectionHandlerFactory",
    "MutualTlsFrameListener",
    "PeerAdmission",
    "PeerContext",
    "close_writer",
]
