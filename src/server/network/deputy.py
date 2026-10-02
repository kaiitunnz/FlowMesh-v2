"""The origin-side route deputy.

Runs on the origin node and probes a resolved candidate ladder in order until one
answers. It dials only the candidates the control plane resolved — it never scans an
address or invents a peer — and returns a classified observation per attempt so the
control plane can update reachability.

A probe dials a candidate's first hop as a peer origin does, over the same mutual TLS
and relay-frame stream, and the listener there answers it. A connection that fails, a
handshake the dialer rejects, a probe left unanswered, or an answer that echoes other
bytes is a path failure. A connection the listener closes after the handshake without
answering is not evidence about the path: a listener that reads no probes and one that
refuses the dialer's identity both close that way and cannot be told apart.
"""

import asyncio
import ssl
from dataclasses import dataclass

from shared.network.frame_stream import (
    MAX_PROBE_BYTES,
    FrameStreamError,
    ProbeFrame,
    read_stream_frame,
    write_probe,
)
from shared.network.peer_dial import (
    PEER_DIAL_ERRORS,
    classify_peer_error,
    open_peer_connection,
)

from .state import (
    ResolvedRoute,
    RouteCandidate,
    RouteObservationOutcome,
    Transport,
)


@dataclass
class ProbeOutcome:
    """The transport that answered the probe, if any, and the classified observation
    of each candidate probed."""

    selected_transport: Transport | None
    observations: list[tuple[Transport, RouteObservationOutcome]]


def dialable_candidates(resolved: ResolvedRoute) -> list[RouteCandidate]:
    """The candidates a probe dials, in order: each forward-dial transport with a hop.

    ``control_relay`` carries no dialable first hop, so a probe skips it.
    """
    return [
        candidate
        for candidate in resolved.candidates
        if candidate.transport is not Transport.CONTROL_RELAY and candidate.hops
    ]


async def run_probe(
    resolved: ResolvedRoute,
    payload: bytes,
    *,
    connect_budget_sec: float,
    ssl_context: ssl.SSLContext | None = None,
) -> ProbeOutcome:
    """Probe each dialable candidate in order; stop at the first that answers.

    Each candidate gets ``connect_budget_sec`` for its whole exchange.
    ``ssl_context`` is the node's peer client context, or ``None`` where the deployment
    runs its peer listeners without mutual TLS. A payload over ``MAX_PROBE_BYTES``
    raises ``ValueError`` before anything is dialed.
    """
    if len(payload) > MAX_PROBE_BYTES:
        raise ValueError(f"probe payload exceeds {MAX_PROBE_BYTES} bytes")
    observations: list[tuple[Transport, RouteObservationOutcome]] = []
    for candidate in dialable_candidates(resolved):
        outcome = await _probe(
            candidate.hops[0].endpoint, payload, connect_budget_sec, ssl_context
        )
        observations.append((candidate.transport, outcome))
        if outcome is RouteObservationOutcome.VERIFIED:
            return ProbeOutcome(candidate.transport, observations)
    return ProbeOutcome(None, observations)


async def _probe(
    endpoint: str,
    payload: bytes,
    budget: float,
    ssl_context: ssl.SSLContext | None,
) -> RouteObservationOutcome:
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(budget):
            try:
                reader, writer = await open_peer_connection(endpoint, ssl_context)
            except PEER_DIAL_ERRORS as exc:
                return classify_peer_error(exc)
            try:
                await write_probe(writer, payload)
                answer = await read_stream_frame(reader)
            except FrameStreamError:
                return RouteObservationOutcome.ROUTE_FAILURE
            except (asyncio.IncompleteReadError, OSError):
                return RouteObservationOutcome.APPLICATION_ERROR
    except TimeoutError:
        return RouteObservationOutcome.TIMEOUT
    finally:
        # A graceful TLS close waits on the peer's close_notify, which a stalled peer
        # never sends; the probe is over either way.
        if writer is not None:
            writer.transport.abort()
    if not isinstance(answer, ProbeFrame) or answer.payload != payload:
        return RouteObservationOutcome.ROUTE_FAILURE
    return RouteObservationOutcome.VERIFIED
