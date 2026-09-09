"""The forward ingress's origin lane.

The ingress runs the same origin drive the root-local proxy runs, over the worker's own
frame sink. What differs is where each relayed frame goes: the engine's head and body
frames belong to the client connected to this worker, so they go straight to that
connection rather than back through control, while the sidecar's attested
acknowledgement and fenced terminal go up to control, which validates every fence and
owns the credit.

The lane reads only relay frame kinds. It never parses an engine body, assembles a
completion, or holds a credential, so hosting it does not make the ingress an executor.
"""

import logging
from collections.abc import Callable

from shared.resident.carriage import ClaimGatedServiceCarriage, ResidentCarriagePlan
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.envelope import ServeRequestEnvelope
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentOpOutcome,
    ResidentStreamChunk,
    ResidentStreamHead,
    ResidentStreamStatus,
)
from shared.resident.serve_drive import ServeOriginDrive

from .channel import ServeIngressChannel

AckSink = Callable[[ResidentBootstrapAck], None]
OutcomeSink = Callable[[ResidentOpOutcome], None]
# Reports the response committed to this ingress's own client (invocation, status,
# headers) up to control, so a post-commit loss fails rather than re-drives.
CommittedSink = Callable[[str, int, tuple[tuple[str, str], ...]], None]


class ServeIngressLane:
    """Drives one worker's forward-ingress invocations and routes their frames."""

    def __init__(
        self,
        *,
        carriage: ClaimGatedServiceCarriage,
        report_ack: AckSink,
        report_outcome: OutcomeSink,
        report_committed: CommittedSink,
        window_bytes: int = 65536,
        stream_deadline_sec: float = 300.0,
        auth_deadline_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._report_ack = report_ack
        self._report_outcome = report_outcome
        self._report_committed = report_committed
        self._channels: dict[str, ServeIngressChannel] = {}
        self._committed: set[str] = set()
        self._drive = ServeOriginDrive(
            carriage=carriage,
            control=self,
            window_bytes=window_bytes,
            stream_deadline_sec=stream_deadline_sec,
            auth_deadline_sec=auth_deadline_sec,
            logger=logger,
        )

    @property
    def drive(self) -> ServeOriginDrive:
        return self._drive

    def begin(
        self,
        *,
        session_id: str,
        invocation_id: str,
        idm: str,
        task_id: str,
        call_correlation: str,
        handoff: AdmissionHandoff,
        envelope: ServeRequestEnvelope,
        channel: ServeIngressChannel,
        plan: ResidentCarriagePlan,
    ) -> None:
        """Start one admitted request's drive, delivering its frames to ``channel``."""
        self._channels[invocation_id] = channel
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
        self._drive.authorize(session_id, auth)

    def close(self, session_id: str, invocation_id: str) -> None:
        self._drive.close(session_id)
        self._channels.pop(invocation_id, None)
        self._committed.discard(invocation_id)

    def on_bootstrap_ack(self, ack: ResidentBootstrapAck) -> None:
        self._report_ack(ack)

    def on_stream_head(self, head: ResidentStreamHead) -> None:
        if (channel := self._channels.get(head.invocation_id)) is not None:
            channel.head(head.status, head.headers)
        # The head commits the response to this ingress's own client, so control marks
        # it flushed and never re-drives it. Report it once per invocation, ahead of the
        # outcome that rides the same ordered event stream.
        if head.invocation_id not in self._committed:
            self._committed.add(head.invocation_id)
            self._report_committed(head.invocation_id, head.status, head.headers)

    def on_stream_chunk(self, chunk: ResidentStreamChunk) -> None:
        if (channel := self._channels.get(chunk.invocation_id)) is not None:
            channel.chunk(chunk.payload)

    def on_outcome(self, outcome: ResidentOpOutcome) -> None:
        # Control consumes the terminal and owns the credit; the connection is finished
        # either way, so a client that is already gone changes nothing about the claim.
        self._report_outcome(outcome)
        self._committed.discard(outcome.invocation_id)
        channel = self._channels.pop(outcome.invocation_id, None)
        if channel is None:
            return
        if outcome.status is ResidentStreamStatus.SUCCESS:
            channel.complete()
        else:
            channel.fail(outcome.error or "resident serve error")
