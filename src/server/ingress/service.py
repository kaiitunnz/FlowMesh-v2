"""The inference ingress edge handler.

An authenticated external principal submits a request against a published alias. The
edge authenticates and quota-limits the principal, resolves the alias under the caller's
tenant, selects a designated origin worker (a placement decision, never a replica or
credit), holds the raw request, and asks resident admission to raise the same claim a
workflow consumer would. The designated worker drives the two-phase resident protocol,
constructs the engine request, and streams the response; the edge relays those opaque
frames to the client unparsed and records the fenced ingress-terminal fact that releases
the credit. The edge never instantiates an engine client, issues engine HTTP, parses a
response, or assembles a completion.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from lumid_hooks import PrincipalContext

from shared.utils.ids import new_idempotency_key, new_invocation_id

from ..resident.service import (
    IngressOrigination,
    ResidentCapacityControl,
)
from ..resident.state import (
    AdmissionProfile,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
)
from ..task.v2.representations.operators import ServiceDependency
from .aliases import AliasCatalog
from .quota import PrincipalQuota
from .state import IngressTerminal, IngressTerminalStatus, IngressTerminalStore

# Selects a live worker to serve as an ingress request's origin deputy, or None.
WorkerSelector = Callable[[], str | None]

_CALL_CORRELATION = "ingress"

_TERMINAL_STATUS = {
    ClaimTerminalReason.COMPLETED: IngressTerminalStatus.COMPLETED,
    ClaimTerminalReason.CANCELLED: IngressTerminalStatus.CANCELLED,
}


class AliasNotFound(Exception):
    """The requested alias is not published."""


class TenantNotAuthorized(Exception):
    """The caller's tenant is not authorized for the requested alias."""


class NoDeputyAvailable(Exception):
    """No live worker is available to serve as the request's origin deputy."""


@dataclass(frozen=True)
class IngressEvent:
    """One event on a request's response stream: a chunk, or a terminal."""

    kind: str  # "chunk" | "done" | "error"
    payload: str = ""
    detail: str | None = None

    @property
    def terminal(self) -> bool:
        return self.kind != "chunk"


class IngressResult:
    """The response stream a submitted ingress request delivers to its client."""

    def __init__(self, stream: "_IngressStream") -> None:
        self._stream = stream

    async def events(self) -> AsyncIterator[IngressEvent]:
        """Yield each teed response frame, ending on the fenced terminal."""
        while True:
            event = await self._stream.queue.get()
            yield event
            if event.terminal:
                return


@dataclass
class _RequestContext:
    """The identity and payload one ingress request drives every attempt under."""

    invocation_id: str
    idempotency_key: str
    subject: InvocationSubject
    dependency: ServiceDependency
    profile: AdmissionProfile
    request_payload: str


class _IngressStream:
    """One in-flight ingress request's client stream and origination state.

    It implements the delivery seam resident-capacity control routes an ingress
    invocation's outcome through: teed frames enqueue for the client, the fenced
    terminal records a durable ingress fact and closes the stream, and an uncertain loss
    re-drives the origination onto a freshly selected deputy under the held claim.
    """

    def __init__(self, ingress: "InferenceIngress", context: _RequestContext):
        self._ingress = ingress
        self._context = context
        self.queue: asyncio.Queue[IngressEvent] = asyncio.Queue()
        self._closed = False
        self._flushed = False

    @property
    def invocation_id(self) -> str:
        return self._context.invocation_id

    @property
    def principal_id(self) -> str:
        return self._context.subject.id

    @property
    def request_payload(self) -> str:
        return self._context.request_payload

    def origination_on(self, origin_worker: str) -> IngressOrigination:
        """The origination for one attempt on the selected deputy.

        Every attempt reuses the stable ``invocation_id`` and worker-lane correlation,
        so admission resumes the in-flight claim rather than raising a successor.
        """
        return IngressOrigination(
            invocation_id=self._context.invocation_id,
            idempotency_key=self._context.idempotency_key,
            task_id=self._context.invocation_id,
            call_correlation=_CALL_CORRELATION,
            subject=self._context.subject,
            dependency=self._context.dependency,
            profile=self._context.profile,
            origin_worker=origin_worker,
            delivery=self,
        )

    def tee(self, payload: str) -> None:
        # Once the client response is closed (a post-flush loss failed it, or it already
        # terminated), a re-drive's frames are dropped rather than duplicated onto it.
        if self._closed:
            return
        self._flushed = True
        self.queue.put_nowait(IngressEvent(kind="chunk", payload=payload))

    def record_terminal(self, reason: ClaimTerminalReason, detail: str | None) -> None:
        self._ingress.record_terminal(self.invocation_id, reason, detail)

    def complete(self) -> None:
        self._finish(IngressEvent(kind="done"))

    def fail(self, detail: str) -> None:
        self._finish(IngressEvent(kind="error", detail=detail))

    def _finish(self, event: IngressEvent) -> None:
        if not self._closed:
            self._closed = True
            self.queue.put_nowait(event)
            self._ingress.release_quota(self.principal_id)

    def redrive(self) -> None:
        # A loss after bytes already reached the client cannot transparently re-stream:
        # re-teeing onto the same connection would duplicate the delivered prefix. Fail
        # this client response instead (the caller retries as a fresh request). A loss
        # before any flush re-drives transparently. Either way the held credit still
        # reconciles through the re-drive, settling on its own fenced terminal.
        if self._flushed:
            self._finish(
                IngressEvent(
                    kind="error", detail="resident stream lost after partial delivery"
                )
            )
        self._ingress.redrive(self)


class InferenceIngress:
    """Authenticates, admits, and streams controlled external inference requests."""

    def __init__(
        self,
        *,
        catalog: AliasCatalog,
        quota: PrincipalQuota,
        terminals: IngressTerminalStore,
        control: ResidentCapacityControl,
        select_worker: WorkerSelector,
        persist: Callable[[], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._catalog = catalog
        self._quota = quota
        self._terminals = terminals
        self._control = control
        self._select_worker = select_worker
        self._persist = persist or (lambda: None)
        self._logger = logger or logging.getLogger("inference-ingress")

    def submit(
        self, principal: PrincipalContext, alias_name: str, request_payload: str
    ) -> IngressResult:
        """Authenticate, admit, and begin streaming one ingress request.

        Pre-admission rejections raise before any credit: an unpublished alias, an
        unauthorized tenant, an exhausted per-principal quota, and no available deputy
        each raise and consume no resident credit.
        """
        alias = self._catalog.get(alias_name)
        if alias is None:
            raise AliasNotFound(alias_name)
        if not alias.authorizes(principal.org_id):
            raise TenantNotAuthorized(alias_name)
        self._quota.acquire(principal.principal_id)
        try:
            deputy = self._select_worker()
            if deputy is None:
                raise NoDeputyAvailable()
            dependency = alias.dependency()
            context = _RequestContext(
                invocation_id=new_invocation_id(),
                idempotency_key=new_idempotency_key(),
                subject=InvocationSubject(
                    kind=InvocationSubjectKind.INGRESS,
                    id=principal.principal_id,
                    tenant=principal.org_id,
                ),
                dependency=dependency,
                profile=AdmissionProfile(
                    engine_batch_key=dependency.engine_batch_key,
                    tenant=principal.org_id,
                    max_output_tokens=alias.max_output_tokens,
                ),
                request_payload=request_payload,
            )
            stream = _IngressStream(self, context)
            if not self._drive_attempt(stream, deputy):
                raise NoDeputyAvailable()
            return IngressResult(stream)
        except Exception:
            self._quota.release(principal.principal_id)
            raise

    def redrive(self, stream: _IngressStream) -> None:
        """Re-drive an uncertain ingress request onto a freshly selected deputy."""
        deputy = self._select_worker()
        if deputy is None or not self._drive_attempt(stream, deputy):
            stream.fail("no worker available to re-drive the ingress request")

    def _drive_attempt(self, stream: _IngressStream, deputy: str) -> bool:
        """Inject the retained request into the deputy and originate the attempt.

        The raw request is (re-)injected into the selected deputy's worker-private
        custody under the stable worker-lane correlation before the origination relays
        the handoff, so a deputy freshly selected on a re-drive holds the request the
        origin driver peeks. Returns whether the injection reached the deputy.
        """
        origination = stream.origination_on(deputy)
        if not self._control.inject_ingress_request(
            deputy,
            origination.task_id,
            origination.call_correlation,
            stream.request_payload,
        ):
            return False
        self._control.originate_ingress(origination)
        return True

    def record_terminal(
        self, invocation_id: str, reason: ClaimTerminalReason, detail: str | None
    ) -> None:
        """Record the fenced ingress-terminal fact before the credit releases."""
        status = _TERMINAL_STATUS.get(reason, IngressTerminalStatus.FAILED)
        if self._terminals.record(
            IngressTerminal(invocation_id=invocation_id, status=status, detail=detail)
        ):
            self._persist()

    def release_quota(self, principal_id: str) -> None:
        """Return the principal's in-flight slot when its request stream closes."""
        self._quota.release(principal_id)
