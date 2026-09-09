"""The root's forward-dialed target-leg carriage.

A trusted deployment lets the root carry an admitted resident invocation's target leg
over a direct socket instead of the reverse-rendezvous relay: ``worker_direct`` opens
the selected worker's claim-gated replica-sidecar listener, ``node_relay`` opens the
target node's purpose-scoped listener, which hands the session to its local sidecar
uplink. Both legs carry the same frames as the relay, so the handoff, route
authorization, fences, windows, and cancellation are unchanged and the target-side claim
gate remains the only authority over the traffic.

The root bridges opaque frames: it reads a frame's routing identity and writes it
through, and never parses a payload, engine token, cursor, or window.

A dial that fails before any frame reaches the target records a classified path
observation and falls through to the relay base under the same claim, request identity,
and held credit. Once a frame has been written the leg never switches transport: a loss
from there on leaves the invocation's outcome ambiguous, which the origin reports as
uncertain with its credit held.
"""

import asyncio
import contextlib
import logging
import socket
import ssl
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from shared.network.frame_stream import (
    FrameStreamError,
    read_relay_frame,
    split_host_port,
    write_relay_frame,
)
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.relay_frame import RelayFrame
from shared.resident.carriage import (
    CONTROL_RELAY,
    CarriageUnavailable,
    ResidentCarriagePlan,
)
from shared.resident.transport import ResidentFrameSink

from ..config import NetworkPlaneConfig
from ..network.rendezvous import TARGET_LEG
from ..network.state import RouteObservationOutcome, Transport

# Delivers one frame the target returned toward the invocation's origin.
InboundSink = Callable[[RelayFrame], Awaitable[None]]
# Records one session's classified target-leg path evidence.
ObservationSink = Callable[[str, Transport, RouteObservationOutcome], None]
# Counts one carried frame and its payload bytes on a named leg and transport.
LegMeter = Callable[[str, str, int], None]


@dataclass(frozen=True)
class TargetLegSupport:
    """What a root locus needs to open trusted target legs over its own base sink.

    One instance carries the deployment's dialing context and the shared observation and
    per-leg counters, so the gated serve ingress and the rendezvous bridge open their
    legs the same way over different bases. ``ssl_context`` is ``None`` where the
    deployment admits no offload.
    """

    ssl_context: ssl.SSLContext | None
    observe: "ObservationSink"
    meter: "LegMeter"
    connect_budget_sec: float

    def carriage(
        self,
        base: ResidentFrameSink,
        deliver: "InboundSink",
        logger: logging.Logger | None = None,
    ) -> "TargetLegCarriage":
        return TargetLegCarriage(
            base=base,
            deliver=deliver,
            observe=self.observe,
            meter=self.meter,
            ssl_context=self.ssl_context,
            connect_budget_sec=self.connect_budget_sec,
            logger=logger,
        )


class TargetLegLost(OSError):
    """A target leg failed after delivery, leaving the attempt's outcome ambiguous."""


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


class _TargetLegSink(ResidentFrameSink):
    """One session's target leg: a lazily dialed socket over the relay base."""

    def __init__(
        self,
        *,
        carriage: "TargetLegCarriage",
        session_id: str,
        endpoint: str,
        transport: Transport,
    ) -> None:
        self._carriage = carriage
        self._session_id = session_id
        self._endpoint = endpoint
        self._transport = transport
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._on_base = False

    async def send(self, frame: RelayFrame) -> None:
        if self._on_base:
            await self._carriage.send_on_base(frame, self._session_id)
            return
        if self._writer is None and not await self._dial():
            await self._carriage.send_on_base(frame, self._session_id)
            return
        assert self._writer is not None
        try:
            await write_relay_frame(self._writer, frame)
        except (OSError, FrameStreamError) as exc:
            self.close()
            raise TargetLegLost(f"target leg lost for {self._session_id}") from exc
        self._carriage.meter(self._transport.value, len(frame.payload))

    async def _dial(self) -> bool:
        """Open the leg, or fall back to the relay base and record the path evidence."""
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
                "target leg %s unavailable for %s, carrying the relay base: %s",
                self._transport.value,
                self._session_id,
                exc,
            )
            self._on_base = True
            return False
        self._writer = writer
        self._carriage.observe(
            self._session_id, self._transport, RouteObservationOutcome.VERIFIED
        )
        self._reader_task = asyncio.ensure_future(self._pump(reader))
        return True

    async def _pump(self, reader: asyncio.StreamReader) -> None:
        """Deliver the target's frames toward the origin until the leg ends."""
        try:
            while True:
                frame = await read_relay_frame(reader)
                self._carriage.meter(self._transport.value, len(frame.payload))
                await self._carriage.deliver(frame)
        except (asyncio.IncompleteReadError, OSError, FrameStreamError):
            pass
        except asyncio.CancelledError:
            raise

    def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        writer, self._writer = self._writer, None
        if writer is not None:
            with contextlib.suppress(OSError):
                writer.close()


class TargetLegCarriage:
    """Realizes a plan's target leg: a trusted direct socket, or the relay base."""

    def __init__(
        self,
        *,
        base: ResidentFrameSink,
        deliver: InboundSink,
        observe: ObservationSink,
        meter: LegMeter,
        ssl_context: ssl.SSLContext | None,
        connect_budget_sec: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self._base = base
        self.deliver = deliver
        self.observe = observe
        self._meter = meter
        self.ssl_context = ssl_context
        self.connect_budget_sec = connect_budget_sec
        self.log = logger or logging.getLogger("target-leg-carriage")
        self._legs: dict[str, _TargetLegSink] = {}

    def select(self, plan: ResidentCarriagePlan) -> ResidentFrameSink:
        """The sink for this attempt's target leg.

        A plan naming no offload carries the relay base. A plan naming one this root
        cannot open is refused rather than relayed silently, so a selection never rides
        a transport other than the one control chose.
        """
        if plan.target_leg_transport == CONTROL_RELAY:
            return self._base
        if self.ssl_context is None or not plan.target_leg_endpoint:
            raise CarriageUnavailable(plan.target_leg_transport)
        leg = _TargetLegSink(
            carriage=self,
            session_id=plan.session_id,
            endpoint=plan.target_leg_endpoint,
            transport=Transport(plan.target_leg_transport),
        )
        self._legs[plan.session_id] = leg
        return leg

    def meter(self, transport: str, payload_bytes: int) -> None:
        self._meter(TARGET_LEG, transport, payload_bytes)

    async def send_on_base(self, frame: RelayFrame, session_id: str) -> None:
        self._meter(TARGET_LEG, CONTROL_RELAY, len(frame.payload))
        await self._base.send(frame)

    def close(self, session_id: str) -> None:
        """Release one session's leg, on its terminal or its reap."""
        leg = self._legs.pop(session_id, None)
        if leg is not None:
            leg.close()

    def close_all(self) -> None:
        for session_id in list(self._legs):
            self.close(session_id)


def bridge_offload_selector(
    carriage: TargetLegCarriage,
) -> Callable[[str, dict[str, str]], ResidentFrameSink | None]:
    """Select the root's offloaded sink for a relayed session from its routing record.

    The rendezvous bridge routes by session; this turns the record control wrote for
    that session back into the carriage plan the root realizes its target leg from.
    """

    def select(session_id: str, record: dict[str, str]) -> ResidentFrameSink | None:
        transport = record.get("target_leg_transport") or CONTROL_RELAY
        if transport == CONTROL_RELAY:
            return None
        return carriage.select(
            ResidentCarriagePlan(
                session_id=session_id,
                target_leg_transport=transport,
                target_leg_endpoint=record.get("target_leg_endpoint") or "",
                route_epoch=int(record.get("route_epoch") or 0),
            )
        )

    return select


def build_target_leg_support(
    config: NetworkPlaneConfig,
    *,
    observe: "ObservationSink",
    meter: "LegMeter",
    logger: logging.Logger | None = None,
) -> TargetLegSupport:
    """The root's dialing support for trusted target legs.

    A deployment that admits no offload gets support with no dialing context, so every
    target leg carries the relay base.
    """
    context: ssl.SSLContext | None = None
    if config.target_leg.enabled:
        context = client_context(
            MutualTlsMaterial.from_b64(
                ca_b64=config.target_leg.ca_b64,
                cert_b64=config.target_leg.cert_b64,
                key_b64=config.target_leg.key_b64,
                root_identity=config.target_leg.root_identity,
            )
        )
        if logger is not None:
            logger.info("Trusted resident target-leg offloads are enabled")
    return TargetLegSupport(
        ssl_context=context,
        observe=observe,
        meter=meter,
        connect_budget_sec=config.connect_budget_sec,
    )


__all__ = [
    "TargetLegCarriage",
    "TargetLegLost",
    "TargetLegSupport",
    "bridge_offload_selector",
    "build_target_leg_support",
]
