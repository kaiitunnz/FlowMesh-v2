"""Dialing a peer frame listener, and classifying how a peer path failed.

A peer origin and the reachability probe open the same connection to a target's peer
frame listener — over mutual TLS where the deployment configures it, verifying that the
certificate covers the dialed host — and classify a failure into the same path evidence.
"""

import asyncio
import socket
import ssl

from shared.schemas.network import RouteObservationOutcome

from .frame_stream import split_host_port

# A failed dial raises a socket or TLS failure, or a parse failure of the endpoint.
PEER_DIAL_ERRORS = (OSError, ValueError)


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


def classify_peer_error(exc: BaseException) -> RouteObservationOutcome:
    """The path evidence a failed dial or a lost peer connection records."""
    if isinstance(exc, ssl.SSLError):
        return RouteObservationOutcome.TLS_FAILURE
    if isinstance(exc, socket.gaierror):
        return RouteObservationOutcome.DNS_FAILURE
    if isinstance(exc, ConnectionRefusedError):
        return RouteObservationOutcome.CONNECT_FAILURE
    if isinstance(exc, TimeoutError):
        return RouteObservationOutcome.TIMEOUT
    return RouteObservationOutcome.ROUTE_FAILURE


__all__ = ["PEER_DIAL_ERRORS", "classify_peer_error", "open_peer_connection"]
