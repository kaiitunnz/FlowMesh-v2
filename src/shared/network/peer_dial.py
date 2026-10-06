"""Dialing a peer frame listener, and classifying how a peer path failed.

A peer origin and the reachability probe open the same connection to a target's peer
frame listener — over mutual TLS where the deployment configures it, verifying that the
certificate covers the dialed host — and classify a failure into the same path evidence.
An origin carrying a session also waits for the target to accept the connection, since a
TLS 1.3 handshake completes on the dialer's side before the target has checked the
dialer's certificate or its own capacity.
"""

import asyncio
import socket
import ssl

from shared.schemas.network import RouteObservationOutcome

from .frame_stream import (
    AcceptFrame,
    AcceptStatus,
    FrameStreamError,
    read_stream_frame,
    split_host_port,
    write_accept,
)

# A failed dial raises a socket or TLS failure, or a parse failure of the endpoint.
PEER_DIAL_ERRORS = (OSError, ValueError)


class PeerAcceptError(OSError):
    """The target did not accept a dialed connection.

    ``outcome`` is the path evidence the refusal carries, or ``None`` for a target at
    its connection cap: that is load on a healthy path, not evidence about it.
    """

    def __init__(self, outcome: RouteObservationOutcome | None, detail: str) -> None:
        super().__init__(detail)
        self.outcome = outcome


async def open_peer_connection(
    endpoint: str, ssl_context: ssl.SSLContext | None, timeout: float | None = None
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a stream to the peer listener at ``endpoint`` (``host:port``)."""
    host, port = split_host_port(endpoint)
    return await asyncio.wait_for(
        asyncio.open_connection(
            host,
            port,
            ssl=ssl_context,
            server_hostname=host if ssl_context is not None else None,
        ),
        timeout=timeout,
    )


async def open_accepted_connection(
    endpoint: str, ssl_context: ssl.SSLContext | None, timeout: float
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a stream to the peer listener at ``endpoint`` once the target accepts it.

    The connect, handshake and accept share one ``timeout``. A refusal raises
    ``PeerAcceptError``. Under mutual TLS a target that closes without answering
    rejected the dialer's certificate in its own side of the handshake, which a TLS 1.3
    dialer's side completes without seeing, so that close records as a TLS failure.
    """
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await open_peer_connection(endpoint, ssl_context)
            await write_accept(writer, AcceptStatus.REQUEST)
            try:
                answer = await read_stream_frame(reader)
            except asyncio.IncompleteReadError as exc:
                raise PeerAcceptError(
                    (
                        RouteObservationOutcome.TLS_FAILURE
                        if ssl_context is not None
                        else RouteObservationOutcome.ROUTE_FAILURE
                    ),
                    "the target closed the connection without accepting it",
                ) from exc
            except FrameStreamError as exc:
                raise PeerAcceptError(
                    RouteObservationOutcome.ROUTE_FAILURE,
                    "the target answered with broken framing",
                ) from exc
    except BaseException:
        if writer is not None:
            writer.transport.abort()
        raise
    status = answer.status if isinstance(answer, AcceptFrame) else None
    if status is AcceptStatus.ACCEPTED:
        return reader, writer
    writer.transport.abort()
    if status is AcceptStatus.BUSY:
        raise PeerAcceptError(None, "the target is at its connection cap")
    if status is AcceptStatus.REFUSED:
        raise PeerAcceptError(
            RouteObservationOutcome.TLS_FAILURE, "the target refused this origin"
        )
    raise PeerAcceptError(
        RouteObservationOutcome.ROUTE_FAILURE, "the target answered out of turn"
    )


def classify_peer_error(exc: BaseException) -> RouteObservationOutcome:
    """The path evidence a failed dial or a lost peer connection records."""
    if isinstance(exc, PeerAcceptError) and exc.outcome is not None:
        return exc.outcome
    if isinstance(exc, ssl.SSLError):
        return RouteObservationOutcome.TLS_FAILURE
    if isinstance(exc, socket.gaierror):
        return RouteObservationOutcome.DNS_FAILURE
    if isinstance(exc, ConnectionRefusedError):
        return RouteObservationOutcome.CONNECT_FAILURE
    if isinstance(exc, TimeoutError):
        return RouteObservationOutcome.TIMEOUT
    return RouteObservationOutcome.ROUTE_FAILURE


__all__ = [
    "PEER_DIAL_ERRORS",
    "PeerAcceptError",
    "classify_peer_error",
    "open_accepted_connection",
    "open_peer_connection",
]
