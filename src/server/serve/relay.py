"""The root's gated serve origin executor.

A gated serve ingress, ``proxy`` or ``forward``, is the registered transport-only
``RouteOrigin`` for a task-addressed external invocation. This executor is its
transport: it holds the root-internal rendezvous attachment that consumes the edge
stream's down leg and publishes origin-produced frames onto its up leg, and runs the
shared serve origin drive over the transport control selected for each attempt. Where
the root can dial a peer, an admitted pair's attempt rides a socket the root opens to
the target, and every other attempt rides ``control_relay`` over the attachment.

The drive itself — the two-phase bootstrap, the opaque response relay, and the fenced
terminal reported to control — is shared by both modes, so one credit-reporting path
serves them.
"""

import asyncio
import logging
import os
from typing import Protocol

from opentelemetry.trace import Tracer

from shared.network.relay_frame import RelayFrame
from shared.resident.carriage import ResidentCarriagePlan
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.envelope import ServeRequestEnvelope
from shared.resident.peer_carriage import PeerDialer, origin_carriage
from shared.resident.reports import ResidentRouteObservation
from shared.resident.serve_drive import ServeControl, ServeOriginDrive

from ..network.reverse_relay import (
    RESIDENT_RELAY_KEYSPACE,
    BinaryRedis,
    EdgeStreamSink,
    RelayStreamStore,
)
from ..supervisor.services.reverse_relay_attachment import ReverseRelayAttachment

# The gated serve edge rides one dedicated reverse-relay stream id — not a worker node —
# so its return frames land on a stream it consumes in the root process, distinct from
# the root supervisor's own node stream. The root bridge pump sweeps this id to forward
# both directions by the session record.
SERVE_EDGE_STREAM_ID = "serve-edge"

__all__ = ["SERVE_EDGE_STREAM_ID", "ServeOriginControl", "ServeRelayExecutor"]


class ServeOriginControl(ServeControl, Protocol):
    """The control the root serve origin reports transitions and path evidence to."""

    def on_route_observation(self, observation: ResidentRouteObservation) -> None: ...


class ServeRelayExecutor:
    """Runs the serve origin drive over the root's own rendezvous attachment, and over
    the peer sockets it dials where it can."""

    def __init__(
        self,
        *,
        relay_redis: BinaryRedis,
        edge_id: str,
        control: ServeOriginControl,
        peer: PeerDialer | None = None,
        window_bytes: int = 65536,
        stream_deadline_sec: float = 300.0,
        auth_deadline_sec: float = 60.0,
        tracer: Tracer | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = RelayStreamStore(relay_redis, RESIDENT_RELAY_KEYSPACE)
        self._edge_id = edge_id
        self._control = control
        self._logger = logger or logging.getLogger("serve-relay")
        self._attachment = ReverseRelayAttachment(
            relay_redis,
            edge_id,
            self,
            owner=f"serve-edge:{os.getpid()}",
            keyspace=RESIDENT_RELAY_KEYSPACE,
        )
        self.dials_peers = peer is not None
        self._carriage = origin_carriage(
            EdgeStreamSink(self._streams, edge_id),
            peer,
            deliver=self.on_frame,
            report=control.on_route_observation,
            logger=self._logger,
        )
        self._drive = ServeOriginDrive(
            carriage=self._carriage,
            control=control,
            window_bytes=window_bytes,
            stream_deadline_sec=stream_deadline_sec,
            auth_deadline_sec=auth_deadline_sec,
            tracer=tracer,
            logger=logger,
        )

    @property
    def edge_id(self) -> str:
        return self._edge_id

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Begin consuming the edge stream's down leg."""
        self._attachment.start(loop)

    async def stop(self) -> None:
        """Release every attempt's carriage, open no more, then stop the attachment."""
        self._drive.stop()
        await self._attachment.stop()

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one inbound relay frame to its session (the attachment's delivery)."""
        await self._drive.on_frame(frame)

    def open(
        self,
        *,
        session_id: str,
        invocation_id: str,
        idm: str,
        task_id: str,
        call_correlation: str,
        handoff: AdmissionHandoff,
        envelope: ServeRequestEnvelope,
        plan: ResidentCarriagePlan,
        traceparent: str | None = None,
    ) -> None:
        """Start one origin drive: send the bootstrap and stream the response."""
        self._drive.open(
            session_id=session_id,
            invocation_id=invocation_id,
            idm=idm,
            task_id=task_id,
            call_correlation=call_correlation,
            handoff=handoff,
            envelope=envelope,
            plan=plan,
            traceparent=traceparent,
        )

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None:
        """Deliver control's post-acceptance route authorization to a waiting drive."""
        self._drive.authorize(session_id, auth)

    def close(self, session_id: str) -> None:
        """Cancel and forget one drive, e.g. on a fenced terminal or reap."""
        self._drive.close(session_id)
