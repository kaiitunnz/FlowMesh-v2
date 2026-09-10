"""The origin's directly dialed carriage for an admitted resident invocation.

Where a deployment declares the origin-to-target pair trusted, the origin carries its
attempt over a socket it opens itself: ``worker_direct`` reaches the selected worker's
claim-gated replica-sidecar listener, ``node_relay`` reaches the target node's
purpose-scoped listener, which hands the session to its local sidecar uplink. Both carry
the same frames as the relay, so the handoff, route authorization, fences, windows, and
cancellation are unchanged, and the target-side claim gate remains the only authority
over the traffic.

The origin is whichever participant control resolved as the route's source. For a
workflow boundary that is the invocation's own worker, so its payload reaches the target
without entering the root or the rendezvous at all. For a gated serve request the root
is itself the origin, and dials on its own behalf.

A dial that fails before any frame reaches the target records a classified path
observation and falls through to the relay base under the same claim, request identity,
and held credit. Once a frame has been written the attempt never switches transport: a
loss from there leaves the outcome ambiguous, which the origin reports as uncertain with
its credit held. Such a loss records the same observation, so the re-drive resolves the
transport as demoted and carries the relay base — an attempt is never replayed across
transports, and a path that keeps failing stops being selected. Only a transport loss
observes; a fence, tenant, descriptor, application, or engine rejection arrives as a
frame and settles the boundary without touching the path.
"""

import asyncio
import contextlib
import logging
import socket
import ssl
from collections.abc import Awaitable, Callable

from shared.network.frame_stream import (
    FrameStreamError,
    read_relay_frame,
    split_host_port,
    write_relay_frame,
)
from shared.network.mtls import peer_matches
from shared.network.relay_frame import RelayFrame
from shared.schemas.network import RouteObservationOutcome, Transport

from .carriage import CONTROL_RELAY, CarriageUnavailable, ResidentCarriagePlan
from .transport import ResidentFrameSink

# Delivers one frame the target returned into the origin's own session.
InboundSink = Callable[[RelayFrame], Awaitable[None]]
# Records one session's classified path evidence for the control plane.
ObservationSink = Callable[[str, Transport, RouteObservationOutcome], None]


class DirectCarriageLost(OSError):
    """A dialed carriage failed after delivery, leaving the outcome ambiguous."""


def _classify(exc: BaseException) -> RouteObservationOutcome:
    if isinstance(exc, ssl.SSLError):
        return RouteObservationOutcome.TLS_FAILURE
    if isinstance(exc, socket.gaierror):
        return RouteObservationOutcome.DNS_FAILURE
    if isinstance(exc, ConnectionRefusedError):
        return RouteObservationOutcome.CONNECT_FAILURE
    if isinstance(exc, TimeoutError):
        return RouteObservationOutcome.TIMEOUT
    return RouteObservationOutcome.ROUTE_FAILURE


class _DirectSink(ResidentFrameSink):
    """One attempt's dialed socket, falling back to the relay base before delivery."""

    def __init__(
        self,
        *,
        carriage: "DirectOffloadCarriage",
        session_id: str,
        endpoint: str,
        transport: Transport,
        expects: frozenset[str],
    ) -> None:
        self._carriage = carriage
        self._session_id = session_id
        self._endpoint = endpoint
        self._transport = transport
        self._expects = expects
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._on_base = False
        self._closing = False

    async def send(self, frame: RelayFrame) -> None:
        if self._on_base:
            await self._carriage.send_on_base(frame)
            return
        if self._writer is None and not await self._dial():
            await self._carriage.send_on_base(frame)
            return
        assert self._writer is not None
        try:
            await write_relay_frame(self._writer, frame)
        except (OSError, FrameStreamError) as exc:
            self._observe_loss(exc)
            self.close()
            raise DirectCarriageLost(f"carriage lost for {self._session_id}") from exc

    async def _dial(self) -> bool:
        """Open the socket, or fall back to the relay and record the path evidence."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    *split_host_port(self._endpoint), ssl=self._carriage.ssl_context
                ),
                timeout=self._carriage.connect_budget_sec,
            )
        except (OSError, ssl.SSLError, TimeoutError) as exc:
            self._carriage.observe(self._session_id, self._transport, _classify(exc))
            self._carriage.log.info(
                "%s unavailable for %s, carrying the relay base: %s",
                self._transport.value,
                self._session_id,
                exc,
            )
            self._on_base = True
            return False
        if not self._carriage.peer_admitted(writer, self._expects):
            self._carriage.observe(
                self._session_id, self._transport, RouteObservationOutcome.TLS_FAILURE
            )
            self._carriage.log.warning(
                "%s for %s presented an identity control did not select",
                self._transport.value,
                self._session_id,
            )
            with contextlib.suppress(OSError):
                writer.close()
            self._on_base = True
            return False
        self._writer = writer
        self._carriage.observe(
            self._session_id, self._transport, RouteObservationOutcome.VERIFIED
        )
        self._reader_task = asyncio.ensure_future(self._pump(reader))
        return True

    async def _pump(self, reader: asyncio.StreamReader) -> None:
        """Deliver the target's frames into this attempt until the socket ends."""
        try:
            while True:
                await self._carriage.deliver(await read_relay_frame(reader))
        except (asyncio.IncompleteReadError, OSError, FrameStreamError) as exc:
            self._observe_loss(exc)
        except asyncio.CancelledError:
            raise

    def _observe_loss(self, exc: BaseException) -> None:
        """Record a transport loss, unless this attempt is already being released.

        Releasing ends the read with the same errors a genuine loss raises, so a session
        torn down on its terminal would otherwise demote a healthy transport.

        The demotion steers the next drive, so it reaches a loss the socket surfaces at
        once. A loss the origin only notices at its own stream deadline can outlive the
        demotion's negative TTL, and that attempt re-drives the way it would over the
        relay.
        """
        if self._closing:
            return
        self._carriage.observe(self._session_id, self._transport, _classify(exc))

    def close(self) -> None:
        self._closing = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        writer, self._writer = self._writer, None
        if writer is not None:
            with contextlib.suppress(OSError):
                writer.close()


class DirectOffloadCarriage:
    """Realizes a plan's transport: a trusted dialed socket, or the relay base.

    A plan naming the target's identity refuses a peer that verifies against the
    deployment CA but is some other party. A plan naming none falls back to what
    mutual TLS already proved — the CA issues an identity only to a registered worker
    or node — with the target's claim gate fencing the session to the admitted
    invocation. Without mutual TLS there is no identity at all, and the trusted-pair
    policy that selected the route is the only gate.
    """

    def __init__(
        self,
        *,
        base: ResidentFrameSink,
        deliver: InboundSink,
        observe: ObservationSink,
        ssl_context: ssl.SSLContext | None,
        connect_budget_sec: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self._base = base
        self.deliver = deliver
        self.observe = observe
        self.ssl_context = ssl_context
        self.connect_budget_sec = connect_budget_sec
        self.log = logger or logging.getLogger("direct-offload-carriage")
        self._sinks: dict[str, _DirectSink] = {}

    def select(self, plan: ResidentCarriagePlan) -> ResidentFrameSink:
        """The sink for this attempt.

        A plan naming the relay carries the base sink. A plan naming an offload this
        origin cannot open is refused rather than relayed silently, so a selection never
        rides a transport other than the one control chose.
        """
        if plan.selected_transport == CONTROL_RELAY:
            return self._base
        if not plan.selected_endpoint:
            raise CarriageUnavailable(plan.selected_transport)
        sink = _DirectSink(
            carriage=self,
            session_id=plan.session_id,
            endpoint=plan.selected_endpoint,
            transport=Transport(plan.selected_transport),
            expects=(
                frozenset({plan.selected_identity})
                if plan.selected_identity
                else frozenset()
            ),
        )
        self._sinks[plan.session_id] = sink
        return sink

    def peer_admitted(
        self, writer: asyncio.StreamWriter, expects: frozenset[str]
    ) -> bool:
        """Whether the dialed peer is the target control selected for this session."""
        if self.ssl_context is None:
            return True
        ssl_object = writer.get_extra_info("ssl_object")
        if not isinstance(ssl_object, ssl.SSLObject):
            return False
        if not expects:
            return True
        return peer_matches(ssl_object.getpeercert(), expects)

    async def send_on_base(self, frame: RelayFrame) -> None:
        await self._base.send(frame)

    def close(self, session_id: str) -> None:
        """Release one attempt's socket, on its terminal or its reap."""
        sink = self._sinks.pop(session_id, None)
        if sink is not None:
            sink.close()

    def close_all(self) -> None:
        for session_id in list(self._sinks):
            self.close(session_id)


__all__ = [
    "DirectCarriageLost",
    "DirectOffloadCarriage",
    "InboundSink",
    "ObservationSink",
]
