"""The origin-side route deputy.

Runs on the origin node and probes a resolved candidate ladder in order until one
answers. It dials only the candidates the control plane resolved — it never scans an
address or invents a peer — and returns a classified observation per attempt so the
control plane can update reachability.

A probe dials a candidate's first hop as a peer origin does, over the same mutual TLS
and relay-frame stream, and the listener there answers it. A connection that fails, a
handshake the dialer rejects, a probe left unanswered, or an answer that echoes other
bytes is a path failure. A connection the listener closes after the handshake without
answering, as a listener that predates probes or refuses the deputy's identity does, is
not evidence about the path.
"""

import asyncio
import socket
import ssl
from dataclasses import dataclass

from shared.network.frame_stream import (
    FrameStreamError,
    ProbeFrame,
    read_stream_frame,
    split_host_port,
    write_probe,
)

from .state import (
    ResolvedRoute,
    RouteObservationOutcome,
    Transport,
)


@dataclass
class EchoOutcome:
    """The deputy's result: the transport that answered the probe (if any) and the
    per-candidate classified observations."""

    selected_transport: Transport | None
    echoed: bytes | None
    observations: list[tuple[Transport, RouteObservationOutcome]]


async def run_echo(
    resolved: ResolvedRoute,
    payload: bytes,
    *,
    connect_budget_sec: float,
    ssl_context: ssl.SSLContext | None = None,
) -> EchoOutcome:
    """Probe each forward-dial candidate in order; stop at the first that answers.

    ``control_relay`` carries no dialable first hop, so this forward-dial diagnostic
    skips it. ``ssl_context`` is the node's peer client context, or ``None`` where the
    deployment runs its peer listeners without mutual TLS.
    """
    observations: list[tuple[Transport, RouteObservationOutcome]] = []
    for candidate in resolved.candidates:
        if candidate.transport is Transport.CONTROL_RELAY or not candidate.hops:
            continue
        outcome, echoed = await _probe(
            candidate.hops[0].endpoint, payload, connect_budget_sec, ssl_context
        )
        observations.append((candidate.transport, outcome))
        if outcome is RouteObservationOutcome.VERIFIED:
            return EchoOutcome(candidate.transport, echoed, observations)
    return EchoOutcome(None, None, observations)


async def _probe(
    endpoint: str,
    payload: bytes,
    budget: float,
    ssl_context: ssl.SSLContext | None,
) -> tuple[RouteObservationOutcome, bytes | None]:
    host, port = split_host_port(endpoint)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                host,
                port,
                ssl=ssl_context,
                server_hostname=host if ssl_context is not None else None,
            ),
            timeout=budget,
        )
    except TimeoutError:
        return RouteObservationOutcome.TIMEOUT, None
    except ssl.SSLError:
        return RouteObservationOutcome.TLS_FAILURE, None
    except socket.gaierror:
        return RouteObservationOutcome.DNS_FAILURE, None
    except ConnectionRefusedError:
        return RouteObservationOutcome.CONNECT_FAILURE, None
    except OSError:
        return RouteObservationOutcome.ROUTE_FAILURE, None
    try:
        await write_probe(writer, payload)
        answer = await asyncio.wait_for(read_stream_frame(reader), timeout=budget)
    except TimeoutError:
        return RouteObservationOutcome.TIMEOUT, None
    except FrameStreamError:
        return RouteObservationOutcome.ROUTE_FAILURE, None
    except (asyncio.IncompleteReadError, OSError):
        return RouteObservationOutcome.APPLICATION_ERROR, None
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, asyncio.CancelledError):
            pass
    if not isinstance(answer, ProbeFrame) or answer.payload != payload:
        return RouteObservationOutcome.ROUTE_FAILURE, None
    return RouteObservationOutcome.VERIFIED, answer.payload
