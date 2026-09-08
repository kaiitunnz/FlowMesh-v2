"""The gated serve edge's transport-only relay executor.

The gated serve edge is the registered transport-only ``RouteOrigin`` for a
task-addressed external invocation. Unlike a workflow call, no caller-worker origin
driver is involved: this executor drives the origin side of the invocation's
reverse-relay session directly in the root, carrying the binding-derived request and
relaying opaque response frames. It never constructs or parses an engine request,
assembles a completion, materializes a result, or owns an engine credential — the
selected replica worker's claim-gated sidecar does all of that. It reads only relay
frame kinds (transport framing) and reports the sidecar's attested acknowledgement and
terminal to control, which validates every fence.

Distinct from the worker ``ResidentOriginDriver``: it tees each response frame to the
client through control rather than assembling and materializing a completion, and its
success terminal carries no manifest — the live relay is the only serve-data mode.
"""

import asyncio
import logging
import os
from typing import Protocol

from shared.network.relay_frame import RelayFrame
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamChunk,
    ResidentStreamStatus,
)
from shared.resident.session import ResidentRelaySession, ResidentSessionRole
from shared.resident.transport import ResidentFrameSink
from shared.resident.wire import (
    KIND_ACK,
    KIND_BOOTSTRAP,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_REJECT,
    KIND_STREAM,
)

from ..network.reverse_relay import BinaryRedis, RelayStreamStore
from ..supervisor.services.reverse_relay_attachment import ReverseRelayAttachment

# The gated serve edge rides one dedicated reverse-relay stream id — not a worker node —
# so its return frames land on a stream it consumes in the root process, distinct from
# the root supervisor's own node stream. The root bridge pump sweeps this id to forward
# both directions by the session record.
SERVE_EDGE_STREAM_ID = "serve-edge"


class ServeControl(Protocol):
    """The subset of resident-capacity control this executor reports transitions to."""

    def on_bootstrap_ack(self, ack: ResidentBootstrapAck) -> None: ...

    def on_stream_chunk(self, chunk: ResidentStreamChunk) -> None: ...

    def on_outcome(self, outcome: ResidentOpOutcome) -> None: ...


class _EdgeSink(ResidentFrameSink):
    """Publishes one origin-produced frame to the edge stream for the root to bridge."""

    def __init__(self, streams: RelayStreamStore, edge_id: str) -> None:
        self._streams = streams
        self._edge_id = edge_id

    async def send(self, frame: RelayFrame) -> None:
        await self._streams.publish_up(self._edge_id, frame)


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


class ServeRelayExecutor:
    """Drives the origin side of every gated serve invocation's relay session.

    One main-process reverse-relay attachment consumes the edge stream's down leg and
    routes each frame to its session; per invocation, an origin session sends the
    binding-derived bootstrap and authorized-stream frames and reads the sidecar's
    response, reporting the acknowledgement and terminal to control while teeing each
    response frame to the client.
    """

    def __init__(
        self,
        *,
        relay_redis: BinaryRedis,
        edge_id: str,
        control: ServeControl,
        window_bytes: int = 65536,
        stream_deadline_sec: float = 300.0,
        auth_deadline_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._streams = RelayStreamStore(relay_redis)
        self._edge_id = edge_id
        self._control = control
        self._window_bytes = window_bytes
        self._stream_deadline = stream_deadline_sec
        self._auth_deadline = auth_deadline_sec
        self._logger = logger or logging.getLogger("serve-relay")
        self._sink = _EdgeSink(self._streams, edge_id)
        self._attachment = ReverseRelayAttachment(
            relay_redis, edge_id, self, owner=f"serve-edge:{os.getpid()}"
        )
        self._by_session: dict[str, _Drive] = {}

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
        request_payload: str,
    ) -> None:
        """Start one origin drive: send the bootstrap and stream the response."""
        session = ResidentRelaySession(
            session_id=session_id,
            invocation_id=invocation_id,
            idm=idm,
            role=ResidentSessionRole.ORIGIN,
            sink=self._sink,
            window_bytes=self._window_bytes,
        )
        drive = _Drive(session, task_id, call_correlation, invocation_id)
        self._by_session[session_id] = drive
        drive.task = asyncio.ensure_future(self._drive(drive, handoff, request_payload))

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
        self, drive: _Drive, handoff: AdmissionHandoff, request_payload: str
    ) -> None:
        try:
            await drive.session.send_wire(
                KIND_BOOTSTRAP,
                handoff=handoff.model_dump(mode="json"),
                request=request_payload,
            )
            ack = await drive.session.recv_wire(self._stream_deadline)
            if not self._handle_ack(drive, ack):
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
            msg = await drive.session.recv_wire(self._stream_deadline)
            if msg is None:
                self._control.on_outcome(self._uncertain(drive, "serve stream lost"))
                return
            kind = msg.get("kind")
            if kind == KIND_CHUNK:
                self._control.on_stream_chunk(
                    ResidentStreamChunk(
                        invocation_id=drive.invocation_id,
                        session_id=drive.session.session_id,
                        payload=str(msg.get("data", "")),
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
