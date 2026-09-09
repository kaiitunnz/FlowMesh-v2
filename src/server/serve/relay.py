"""The root-local proxy ingress's transport-only relay executor.

The root-local gated serve ingress is a registered transport-only ``RouteOrigin`` for a
task-addressed external invocation. This executor is its transport: it holds the
root-internal rendezvous attachment that consumes the edge stream's down leg, publishes
origin-produced frames onto its up leg, and runs the shared serve origin drive over that
sink. The root is itself the origin here, so a trusted target-leg offload carries the
whole invocation over a direct socket.

The drive itself — the two-phase bootstrap, the opaque response relay, and the fenced
terminal reported to control — is shared with the root forward ingress, so one
credit-reporting path serves both gated modes.
"""

import asyncio
import logging
import os

from shared.network.relay_frame import RelayFrame
from shared.resident.carriage import (
    ClaimGatedServiceCarriage,
    ControlRelayCarriage,
    ResidentCarriagePlan,
)
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.envelope import ServeRequestEnvelope
from shared.resident.serve_drive import ServeControl, ServeOriginDrive
from shared.resident.transport import ResidentFrameSink

from ..network.reverse_relay import BinaryRedis, RelayStreamStore
from ..resident.target_leg import TargetLegCarriage, TargetLegSupport
from ..supervisor.services.reverse_relay_attachment import ReverseRelayAttachment

# The gated serve edge rides one dedicated reverse-relay stream id — not a worker node —
# so its return frames land on a stream it consumes in the root process, distinct from
# the root supervisor's own node stream. The root bridge pump sweeps this id to forward
# both directions by the session record.
SERVE_EDGE_STREAM_ID = "serve-edge"

__all__ = ["SERVE_EDGE_STREAM_ID", "ServeControl", "ServeRelayExecutor"]


class _EdgeSink(ResidentFrameSink):
    """Publishes one origin-produced frame to the edge stream for the root to bridge."""

    def __init__(self, streams: RelayStreamStore, edge_id: str) -> None:
        self._streams = streams
        self._edge_id = edge_id

    async def send(self, frame: RelayFrame) -> None:
        await self._streams.publish_up(self._edge_id, frame)


class ServeRelayExecutor:
    """Runs the serve origin drive over the root's own rendezvous attachment."""

    def __init__(
        self,
        *,
        relay_redis: BinaryRedis,
        edge_id: str,
        control: ServeControl,
        target_leg: TargetLegSupport | None = None,
        window_bytes: int = 65536,
        stream_deadline_sec: float = 300.0,
        auth_deadline_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = RelayStreamStore(relay_redis)
        self._edge_id = edge_id
        self._target_legs: TargetLegCarriage | None = None
        self._attachment = ReverseRelayAttachment(
            relay_redis, edge_id, self, owner=f"serve-edge:{os.getpid()}"
        )
        edge_sink = _EdgeSink(self._streams, edge_id)
        carriage: ClaimGatedServiceCarriage = ControlRelayCarriage(edge_sink)
        if target_leg is not None:
            self._target_legs = target_leg.carriage(edge_sink, self.on_frame, logger)
            carriage = self._target_legs
        self._drive = ServeOriginDrive(
            carriage=carriage,
            control=control,
            window_bytes=window_bytes,
            stream_deadline_sec=stream_deadline_sec,
            auth_deadline_sec=auth_deadline_sec,
            logger=logger,
        )

    @property
    def edge_id(self) -> str:
        return self._edge_id

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Begin consuming the edge stream's down leg."""
        self._attachment.start(loop)

    async def stop(self) -> None:
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
        )

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None:
        """Deliver control's post-acceptance route authorization to a waiting drive."""
        self._drive.authorize(session_id, auth)

    def close(self, session_id: str) -> None:
        """Cancel and forget one drive, e.g. on a fenced terminal or reap."""
        self._drive.close(session_id)
        if self._target_legs is not None:
            self._target_legs.close(session_id)
