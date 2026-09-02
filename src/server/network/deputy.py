"""The origin-side route deputy.

Runs on the origin node and executes a resolved candidate ladder in order, echoing a
payload over each transport until one round-trips. It executes only the candidates the
control plane resolved — it never scans an address or invents a peer — and returns a
classified observation per attempt so the control plane can update reachability.
"""

import asyncio
import socket
import ssl
from dataclasses import dataclass

from . import wire
from .state import (
    ResolvedRoute,
    RouteCandidate,
    RouteObservationOutcome,
    Transport,
)


@dataclass
class EchoOutcome:
    """The deputy's result: the transport that carried the echo (if any) and the
    per-candidate classified observations."""

    selected_transport: Transport | None
    echoed: bytes | None
    observations: list[tuple[Transport, RouteObservationOutcome]]


async def run_echo(
    resolved: ResolvedRoute,
    payload: bytes,
    *,
    connect_budget_sec: float,
) -> EchoOutcome:
    """Attempt each forward-dial candidate in order; stop at the first that round-trips.

    ``control_relay`` carries no dialable first hop — its origin hop endpoint is empty —
    so this forward-dial diagnostic skips it rather than dialing an empty address and
    recording a spurious demotion for the guaranteed reverse-relay base.
    """
    observations: list[tuple[Transport, RouteObservationOutcome]] = []
    for candidate in resolved.candidates:
        if candidate.transport is Transport.CONTROL_RELAY:
            continue
        outcome, echoed = await _attempt(candidate, payload, connect_budget_sec)
        observations.append((candidate.transport, outcome))
        if outcome is RouteObservationOutcome.VERIFIED:
            return EchoOutcome(candidate.transport, echoed, observations)
    return EchoOutcome(None, None, observations)


async def _attempt(
    candidate: RouteCandidate, payload: bytes, budget: float
) -> tuple[RouteObservationOutcome, bytes | None]:
    try:
        return await asyncio.wait_for(_drive(candidate, payload), timeout=budget)
    except TimeoutError:
        return RouteObservationOutcome.TIMEOUT, None
    except ConnectionRefusedError:
        return RouteObservationOutcome.CONNECT_FAILURE, None
    except ssl.SSLError:
        return RouteObservationOutcome.TLS_FAILURE, None
    except socket.gaierror:
        return RouteObservationOutcome.DNS_FAILURE, None
    except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError):
        return RouteObservationOutcome.ROUTE_FAILURE, None


async def _drive(
    candidate: RouteCandidate, payload: bytes
) -> tuple[RouteObservationOutcome, bytes | None]:
    connect_host, connect_port = wire.split_host_port(candidate.hops[0].endpoint)
    reader, writer = await asyncio.open_connection(connect_host, connect_port)
    try:
        # Each intermediate hop is a target-addressed relay: it reads one leading
        # frame naming its next hop, dials it, and byte-relays the rest. Writing a
        # frame per hop after the first therefore chains the caller through the
        # relay ladder to the terminal sidecar; a single-hop direct route writes
        # none and speaks straight to the sidecar.
        for hop in candidate.hops[1:]:
            await wire.write_frame(writer, hop.endpoint.encode())
        return await _echo_exchange(reader, writer, payload)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, asyncio.CancelledError):
            pass


async def _echo_exchange(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, payload: bytes
) -> tuple[RouteObservationOutcome, bytes | None]:
    await wire.write_frame(writer, payload)
    status = await reader.readexactly(1)
    if status == wire.STATUS_APP_ERROR:
        return RouteObservationOutcome.APPLICATION_ERROR, None
    if status == wire.STATUS_OK:
        echoed = await reader.readexactly(len(payload))
        return RouteObservationOutcome.VERIFIED, echoed
    # Only OK / APP_ERROR exist over the echo; a nonstandard status is treated as a path
    # failure here. When real transports land, a status carrying an auth/fence rejection
    # must classify as non-demoting to preserve the never-demote-on-non-path invariant.
    return RouteObservationOutcome.ROUTE_FAILURE, None
