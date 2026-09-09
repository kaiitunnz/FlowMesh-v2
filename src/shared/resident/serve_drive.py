"""The transport-only origin drive for one task-addressed serve invocation.

A gated serve ingress is the registered transport-only ``RouteOrigin`` for an external
serve request. This drive owns the origin side of that invocation's reverse-relay
session: it carries the frozen envelope to the selected replica's claim-gated sidecar,
relays the opaque response frames back, and reports the sidecar's attested
acknowledgement and terminal to control, which validates every fence. The sidecar
constructs and parses the engine request, owns its credential, and serves the response.

The drive reads only relay frame kinds — never a body, cursor, or window — so an ingress
running it applies no engine semantics and assembles nothing. Its success terminal
carries no manifest: the live relay is the serve-data mode. Both gated ingresses run
this same drive over their own frame sink, so one fenced-terminal and credit-reporting
path serves the root-local proxy and a worker-hosted forward ingress alike.
"""

import asyncio
import logging
from typing import Protocol

from shared.network.relay_frame import RelayFrame

from .carriage import (
    CarriageUnavailable,
    ClaimGatedServiceCarriage,
    ResidentCarriagePlan,
)
from .contracts import AdmissionHandoff, RouteAuthorization
from .envelope import ServeRequestEnvelope
from .reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamChunk,
    ResidentStreamHead,
    ResidentStreamStatus,
)
from .session import ResidentRelaySession, ResidentSessionRole
from .wire import (
    KIND_ACK,
    KIND_BOOTSTRAP,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_HEAD,
    KIND_REJECT,
    KIND_STREAM,
)


class ServeControl(Protocol):
    """The subset of resident-capacity control this drive reports transitions to."""

    def on_bootstrap_ack(self, ack: ResidentBootstrapAck) -> None: ...

    def on_stream_head(self, head: ResidentStreamHead) -> None: ...

    def on_stream_chunk(self, chunk: ResidentStreamChunk) -> None: ...

    def on_outcome(self, outcome: ResidentOpOutcome) -> None: ...


class _Drive:
    def __init__(
        self,
        session: ResidentRelaySession,
        task_id: str,
        call_correlation: str,
        invocation_id: str,
    ) -> None:
        self.session = session
        self.task_id = task_id
        self.call_correlation = call_correlation
        self.invocation_id = invocation_id
        self.authorization: asyncio.Future[RouteAuthorization] = (
            asyncio.get_event_loop().create_future()
        )
        self.task: asyncio.Task[None] | None = None


class ServeOriginDrive:
    """Drives the origin side of every gated serve invocation's relay session.

    Per invocation, an origin session sends the frozen bootstrap and authorized-stream
    frames and reads the sidecar's response, reporting the acknowledgement and terminal
    to control while teeing each response frame to the client. The attempt's frame sink
    comes from the carriage control's plan selects, and the caller routes inbound frames
    in through ``on_frame``.
    """

    def __init__(
        self,
        *,
        carriage: ClaimGatedServiceCarriage,
        control: ServeControl,
        window_bytes: int = 65536,
        stream_deadline_sec: float = 300.0,
        auth_deadline_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._carriage = carriage
        self._control = control
        self._window_bytes = window_bytes
        self._stream_deadline = stream_deadline_sec
        self._auth_deadline = auth_deadline_sec
        self._logger = logger or logging.getLogger("serve-relay")
        self._by_session: dict[str, _Drive] = {}

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one inbound relay frame to its session."""
        drive = self._by_session.get(frame.session_id)
        if drive is not None:
            await drive.session.on_frame(frame)

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
        try:
            sink = self._carriage.select(plan)
        except CarriageUnavailable as exc:
            # Control selected a transport this ingress has no carriage for; hold the
            # credit uncertain rather than open a session on the wrong sink.
            self._control.on_outcome(
                ResidentOpOutcome(
                    task_id=task_id,
                    call_correlation=call_correlation,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    status=ResidentStreamStatus.UNCERTAIN,
                    manifest=None,
                    error=f"no carriage for transport {exc}",
                )
            )
            return
        session = ResidentRelaySession(
            session_id=session_id,
            invocation_id=invocation_id,
            idm=idm,
            role=ResidentSessionRole.ORIGIN,
            sink=sink,
            window_bytes=self._window_bytes,
        )
        drive = _Drive(session, task_id, call_correlation, invocation_id)
        self._by_session[session_id] = drive
        drive.task = asyncio.ensure_future(self._drive(drive, handoff, envelope))

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None:
        """Deliver control's post-acceptance route authorization to a waiting drive."""
        drive = self._by_session.get(session_id)
        if drive is not None and not drive.authorization.done():
            drive.authorization.set_result(auth)

    def close(self, session_id: str) -> None:
        """Cancel and forget one drive, e.g. on a fenced terminal or reap."""
        drive = self._by_session.pop(session_id, None)
        if drive is None:
            return
        task = drive.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _drive(
        self, drive: _Drive, handoff: AdmissionHandoff, envelope: ServeRequestEnvelope
    ) -> None:
        try:
            await drive.session.send_body_wire(
                KIND_BOOTSTRAP,
                envelope.body,
                handoff=handoff.model_dump(mode="json"),
                request=envelope.header_fields(),
            )
            ack = await drive.session.recv_body_wire(self._stream_deadline)
            if not self._handle_ack(drive, ack[0] if ack is not None else None):
                return
            try:
                auth = await asyncio.wait_for(
                    drive.authorization, timeout=self._auth_deadline
                )
            except TimeoutError:
                self._control.on_outcome(
                    self._uncertain(drive, "authorization not delivered")
                )
                return
            await drive.session.send_wire(
                KIND_STREAM, auth=auth.model_dump(mode="json")
            )
            await self._stream(drive)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any escape holds the credit uncertain
            self._logger.exception("serve relay drive failed")
            self._control.on_outcome(
                self._uncertain(drive, f"serve relay error: {exc}")
            )
        finally:
            self._by_session.pop(drive.session.session_id, None)

    def _handle_ack(self, drive: _Drive, ack: dict[str, object] | None) -> bool:
        if ack is None:
            self._control.on_bootstrap_ack(
                self._ack(drive, ResidentBootstrapOutcome.UNCERTAIN)
            )
            return False
        kind = ack.get("kind")
        if kind == KIND_ACK:
            self._control.on_bootstrap_ack(
                self._ack(drive, ResidentBootstrapOutcome.ACKED)
            )
            return True
        if kind == KIND_REJECT:
            reason = ack.get("reason")
            self._control.on_bootstrap_ack(
                self._ack(
                    drive,
                    ResidentBootstrapOutcome.REJECTED,
                    rejection=str(reason) if reason is not None else None,
                )
            )
            return False
        self._control.on_bootstrap_ack(
            self._ack(drive, ResidentBootstrapOutcome.UNCERTAIN)
        )
        return False

    async def _stream(self, drive: _Drive) -> None:
        while True:
            received = await drive.session.recv_body_wire(self._stream_deadline)
            if received is None:
                self._control.on_outcome(self._uncertain(drive, "serve stream lost"))
                return
            msg, body = received
            kind = msg.get("kind")
            if kind == KIND_HEAD:
                # The engine response's own status and headers, relayed opaquely so the
                # client response carries them ahead of the streamed body.
                self._control.on_stream_head(
                    ResidentStreamHead(
                        invocation_id=drive.invocation_id,
                        session_id=drive.session.session_id,
                        status=int(msg.get("status", 200)),
                        headers=tuple(
                            (str(item[0]), str(item[1]))
                            for item in msg.get("headers") or ()
                            if item
                        ),
                    )
                )
            elif kind == KIND_CHUNK:
                self._control.on_stream_chunk(
                    ResidentStreamChunk(
                        invocation_id=drive.invocation_id,
                        session_id=drive.session.session_id,
                        payload=body,
                    )
                )
            elif kind == KIND_DONE:
                # Live-only serve: the fenced status terminal releases the credit; no
                # completion is assembled or materialized.
                self._control.on_outcome(
                    self._outcome(drive, ResidentStreamStatus.SUCCESS)
                )
                return
            elif kind == KIND_FAILED:
                if bool(msg.get("definite")):
                    self._control.on_outcome(
                        self._outcome(
                            drive,
                            ResidentStreamStatus.DEFINITE_FAILURE,
                            error=f"resident engine refused: {msg.get('reason')}",
                        )
                    )
                else:
                    self._control.on_outcome(
                        self._uncertain(drive, f"resident engine: {msg.get('reason')}")
                    )
                return
            else:
                self._control.on_outcome(
                    self._uncertain(drive, "serve stream protocol")
                )
                return

    @staticmethod
    def _ack(
        drive: _Drive,
        outcome: ResidentBootstrapOutcome,
        *,
        rejection: str | None = None,
    ) -> ResidentBootstrapAck:
        return ResidentBootstrapAck(
            task_id=drive.task_id,
            call_correlation=drive.call_correlation,
            invocation_id=drive.invocation_id,
            session_id=drive.session.session_id,
            outcome=outcome,
            rejection=rejection,
        )

    @staticmethod
    def _outcome(
        drive: _Drive,
        status: ResidentStreamStatus,
        *,
        error: str | None = None,
    ) -> ResidentOpOutcome:
        return ResidentOpOutcome(
            task_id=drive.task_id,
            call_correlation=drive.call_correlation,
            invocation_id=drive.invocation_id,
            session_id=drive.session.session_id,
            status=status,
            manifest=None,
            error=error,
        )

    @classmethod
    def _uncertain(cls, drive: _Drive, detail: str) -> ResidentOpOutcome:
        return cls._outcome(drive, ResidentStreamStatus.UNCERTAIN, error=detail)
