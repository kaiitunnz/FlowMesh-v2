"""The origin worker's resident invocation driver.

The agent's worker drives its own resident model boundary: it sends the claim-bound
bootstrap and the raw request over the invocation's windowed session, reports the engine
acknowledgement to control, streams the response under the route authorization control
issues, assembles the completion, and materializes it into the content store — reporting
only the bounded fenced manifest. The raw request never leaves the worker except over
the data path; only the completed manifest and a fenced terminal release the credit.

Every transition is safe under loss: an ambiguous bootstrap or stream reports
``UNCERTAIN`` so the credit is held and the boundary re-drives, and a re-drive that
finds an already materialized outcome re-reports it rather than re-running the engine.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shared.network.relay_frame import RelayFrame
from shared.outcome import FabricContentStore, OutcomeManifest
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamChunk,
    ResidentStreamStatus,
)
from shared.resident.wire import (
    KIND_ACK,
    KIND_BOOTSTRAP,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_REJECT,
    KIND_STREAM,
)

from .session import ResidentRelaySession, ResidentSessionRole
from .transport import ResidentFrameSink

AckSink = Callable[[ResidentBootstrapAck], None]
OutcomeSink = Callable[[ResidentOpOutcome], None]
StreamChunkSink = Callable[[ResidentStreamChunk], None]


@dataclass(frozen=True)
class ResidentOriginRequest:
    """What control hands the origin worker to drive one bootstrap attempt.

    ``session_id`` is fresh per attempt; ``handoff`` is the claim-bound fence control
    minted for the reserved claim; ``request_payload`` is the worker-private raw request
    the driver sends over the data path. ``tee`` marks an ingress request whose response
    frames are teed to control as they stream, so the ingress relays them to the client.
    """

    task_id: str
    call_correlation: str
    session_id: str
    handoff: AdmissionHandoff
    request_payload: str | None
    tee: bool = False


@dataclass
class _Origin:
    request: ResidentOriginRequest
    session: ResidentRelaySession
    authorization: "asyncio.Future[RouteAuthorization]"
    task: "asyncio.Task[None] | None" = None


class ResidentOriginDriver:
    """Drives the origin side of a worker's resident boundaries."""

    def __init__(
        self,
        *,
        sink: ResidentFrameSink,
        content_store: FabricContentStore | None,
        report_ack: AckSink,
        report_outcome: OutcomeSink,
        report_stream_chunk: StreamChunkSink | None = None,
        window_bytes: int = 65536,
        stream_deadline_sec: float = 300.0,
        auth_deadline_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sink = sink
        self._content_store = content_store
        self._report_ack = report_ack
        self._report_outcome = report_outcome
        self._report_stream_chunk = report_stream_chunk
        self._window_bytes = window_bytes
        self._stream_deadline = stream_deadline_sec
        self._auth_deadline = auth_deadline_sec
        self._logger = logger or logging.getLogger("resident-origin-driver")
        self._by_session: dict[str, _Origin] = {}
        self._by_call: dict[str, _Origin] = {}

    def begin(self, request: ResidentOriginRequest) -> None:
        """Start one bootstrap attempt for a control-relayed handoff."""
        self._reap(request.call_correlation)
        session = ResidentRelaySession(
            session_id=request.session_id,
            invocation_id=request.handoff.invocation_id,
            idm=request.handoff.idempotency_key or "",
            role=ResidentSessionRole.ORIGIN,
            sink=self._sink,
            window_bytes=self._window_bytes,
        )
        origin = _Origin(
            request=request,
            session=session,
            authorization=asyncio.get_running_loop().create_future(),
        )
        self._by_session[request.session_id] = origin
        self._by_call[request.call_correlation] = origin
        origin.task = asyncio.ensure_future(self._drive(origin))

    def authorize(self, call_correlation: str, auth: RouteAuthorization) -> None:
        """Deliver control's post-acceptance route authorization to a waiting driver."""
        origin = self._by_call.get(call_correlation)
        if origin is not None and not origin.authorization.done():
            origin.authorization.set_result(auth)

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one inbound relay frame to its session."""
        origin = self._by_session.get(frame.session_id)
        if origin is not None:
            await origin.session.on_frame(frame)

    def reap(self, call_correlation: str) -> None:
        """Cancel and forget a driver, e.g. on a fenced cancellation terminal."""
        self._reap(call_correlation)

    def _reap(self, call_correlation: str) -> None:
        origin = self._by_call.pop(call_correlation, None)
        if origin is None:
            return
        self._by_session.pop(origin.request.session_id, None)
        task = origin.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _drive(self, origin: _Origin) -> None:
        req = origin.request
        idm = req.handoff.idempotency_key
        try:
            if (prior := self._prior_manifest(idm)) is not None:
                # A post-manifest re-drive: the outcome already committed, so re-report
                # the recorded reference rather than re-running the engine.
                self._report_outcome(
                    self._outcome(req, ResidentStreamStatus.SUCCESS, manifest=prior)
                )
                return
            if req.request_payload is None:
                # The raw request was not captured on this worker (a re-drive landed on
                # a worker that never held it): hold the credit and re-drive rather than
                # bootstrapping an empty-prompt request that could settle a bogus
                # success.
                self._report_outcome(self._uncertain(req, "request not captured here"))
                return
            await origin.session.send_wire(
                KIND_BOOTSTRAP,
                handoff=req.handoff.model_dump(mode="json"),
                request=req.request_payload,
            )
            ack = await origin.session.recv_wire(self._stream_deadline)
            if not self._handle_ack(req, ack):
                return
            try:
                auth = await asyncio.wait_for(
                    origin.authorization, timeout=self._auth_deadline
                )
            except TimeoutError:
                self._report_outcome(
                    self._uncertain(req, "authorization not delivered")
                )
                return
            await origin.session.send_wire(
                KIND_STREAM, auth=auth.model_dump(mode="json")
            )
            await self._stream(origin)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any escape holds the credit uncertain
            self._logger.exception("resident origin drive failed")
            self._report_outcome(self._uncertain(req, f"origin drive error: {exc}"))
        finally:
            self._by_session.pop(req.session_id, None)
            if self._by_call.get(req.call_correlation) is origin:
                self._by_call.pop(req.call_correlation, None)

    def _handle_ack(
        self, req: ResidentOriginRequest, ack: dict[str, Any] | None
    ) -> bool:
        """Report the bootstrap outcome; return whether to proceed to the stream."""
        if ack is None:
            self._report_ack(self._ack(req, ResidentBootstrapOutcome.UNCERTAIN))
            return False
        if ack.get("kind") == KIND_ACK:
            self._report_ack(self._ack(req, ResidentBootstrapOutcome.ACKED))
            return True
        if ack.get("kind") == KIND_REJECT:
            self._report_ack(
                self._ack(
                    req, ResidentBootstrapOutcome.REJECTED, rejection=ack.get("reason")
                )
            )
            return False
        self._report_ack(self._ack(req, ResidentBootstrapOutcome.UNCERTAIN))
        return False

    async def _stream(self, origin: _Origin) -> None:
        req = origin.request
        parts: list[str] = []
        while True:
            msg = await origin.session.recv_wire(self._stream_deadline)
            if msg is None:
                self._report_outcome(self._uncertain(req, "resident stream lost"))
                return
            kind = msg.get("kind")
            if kind == KIND_CHUNK:
                data = str(msg.get("data", ""))
                parts.append(data)
                if req.tee and self._report_stream_chunk is not None:
                    self._report_stream_chunk(
                        ResidentStreamChunk(
                            invocation_id=req.handoff.invocation_id,
                            session_id=req.session_id,
                            payload=data,
                        )
                    )
            elif kind == KIND_DONE:
                self._finalize(req, "".join(parts))
                return
            elif kind == KIND_FAILED:
                if bool(msg.get("definite")):
                    self._report_outcome(
                        self._outcome(
                            req,
                            ResidentStreamStatus.DEFINITE_FAILURE,
                            error=f"resident engine refused: {msg.get('reason')}",
                        )
                    )
                else:
                    self._report_outcome(
                        self._uncertain(req, f"resident engine: {msg.get('reason')}")
                    )
                return
            else:
                self._report_outcome(self._uncertain(req, "resident stream protocol"))
                return

    def _finalize(self, req: ResidentOriginRequest, completion: str) -> None:
        """Materialize the completion into the content store and report the manifest."""
        idm = req.handoff.idempotency_key
        if self._content_store is None or idm is None:
            self._report_outcome(
                self._outcome(
                    req,
                    ResidentStreamStatus.DEFINITE_FAILURE,
                    error="no content store to materialize the resident completion",
                )
            )
            return
        manifest = self._content_store.materialize(
            idm, completion.encode(), media_type="text/plain"
        )
        self._report_outcome(
            self._outcome(req, ResidentStreamStatus.SUCCESS, manifest=manifest)
        )

    def _prior_manifest(self, idm: str | None) -> OutcomeManifest | None:
        if self._content_store is None or idm is None:
            return None
        with contextlib.suppress(Exception):
            return self._content_store.find(idm)
        return None

    @staticmethod
    def _ack(
        req: ResidentOriginRequest,
        outcome: ResidentBootstrapOutcome,
        *,
        rejection: str | None = None,
    ) -> ResidentBootstrapAck:
        return ResidentBootstrapAck(
            task_id=req.task_id,
            call_correlation=req.call_correlation,
            invocation_id=req.handoff.invocation_id,
            session_id=req.session_id,
            outcome=outcome,
            rejection=rejection,
        )

    @staticmethod
    def _outcome(
        req: ResidentOriginRequest,
        status: ResidentStreamStatus,
        *,
        manifest: OutcomeManifest | None = None,
        error: str | None = None,
    ) -> ResidentOpOutcome:
        return ResidentOpOutcome(
            task_id=req.task_id,
            call_correlation=req.call_correlation,
            invocation_id=req.handoff.invocation_id,
            session_id=req.session_id,
            status=status,
            manifest=manifest,
            error=error,
        )

    @classmethod
    def _uncertain(cls, req: ResidentOriginRequest, detail: str) -> ResidentOpOutcome:
        return cls._outcome(req, ResidentStreamStatus.UNCERTAIN, error=detail)
