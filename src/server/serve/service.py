"""The gated task-ID serve edge.

An authenticated FlowMesh principal with task-read access to a live serve task submits
an inference request against its task ID. The edge resolves only that task's live
``ServeTaskResidencyBinding``, derives the bounded canonical request descriptor and the
fixed profile the caller cannot widen, records a durable external-principal
``Invocation``
with no ``DS`` state, and asks the same Admission controller to raise the same
``ServiceClaim`` against only the binding's own allocation group. The selected replica
worker's claim-gated sidecar constructs the engine request, owns the credential, and
streams the response; the edge relays those opaque frames to the client unparsed and
records the fenced status terminal that releases the credit. The edge never instantiates
an engine client, issues engine HTTP, parses a response, or assembles a completion.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.envelope import ServeRequestEnvelope
from shared.utils.ids import new_idempotency_key, new_invocation_id

from ..resident.service import ResidentCapacityControl, ServeOrigination
from ..resident.state import (
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
)
from ..task.v2.representations.operators import ServiceInterface
from .binding import ServeBindingStore, ServeTaskResidencyBinding
from .ingress import ServeAccessMode, ServeIngressRegistry
from .relay import ServeRelayExecutor
from .state import ServeStatusTerminal, ServeTerminalStatus, ServeTerminalStore

# The gate keys each serve invocation's worker-lane correlation per invocation, so a
# terminal reap of one invocation never disturbs a peer's live drive on the shared edge.
_CALL_CORRELATION_PREFIX = "serve/"

# The bound on a client's teed-frame backlog. A client that stops draining cannot pin
# unbounded memory: further frames are dropped past the bound, while the terminal always
# lands (evicting the oldest frame if needed) so the client stream still closes.
_TEE_QUEUE_MAX = 2048

_TERMINAL_STATUS = {
    ClaimTerminalReason.COMPLETED: ServeTerminalStatus.COMPLETED,
    ClaimTerminalReason.CANCELLED: ServeTerminalStatus.CANCELLED,
}

_STATUS_REASON = {
    ServeTerminalStatus.COMPLETED: ClaimTerminalReason.COMPLETED,
    ServeTerminalStatus.CANCELLED: ClaimTerminalReason.CANCELLED,
    ServeTerminalStatus.FAILED: ClaimTerminalReason.FAILED,
}


def _call_correlation(invocation_id: str) -> str:
    return f"{_CALL_CORRELATION_PREFIX}{invocation_id}"


class BindingNotFound(Exception):
    """No live standing serve binding exists for the requested task ID."""


class MethodNotAllowed(Exception):
    """The request method is not one the binding permits."""


class IngressUnavailable(Exception):
    """The binding pins a gated ingress this deployment has not registered."""


class WrongIngress(Exception):
    """The request arrived on a gated ingress other than the one its binding pins."""


class ServeTransport(Protocol):
    """How one gated ingress's origin relay is opened, authorized, and reaped.

    The root-local proxy drives its relay in this process; a forward ingress drives it
    on its own worker and is reached over that worker's authenticated attachment. Both
    carry the same frozen envelope under the same fence.
    """

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
    ) -> None: ...

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None: ...

    def close(self, session_id: str) -> None: ...


@dataclass(frozen=True)
class ServeEvent:
    """One event on a request's response stream: the head, a chunk, or a terminal."""

    kind: str  # "head" | "chunk" | "done" | "error"
    payload: bytes = b""
    detail: str | None = None
    status: int = 200
    headers: tuple[tuple[str, str], ...] = ()

    @property
    def terminal(self) -> bool:
        return self.kind in ("done", "error")


class ServeResult:
    """The response stream a submitted serve request delivers to its client."""

    def __init__(self, stream: "_ServeStream") -> None:
        self._stream = stream

    async def events(self) -> AsyncIterator[ServeEvent]:
        """Yield each relayed response frame, ending on the fenced terminal."""
        while True:
            event = await self._stream.queue.get()
            yield event
            if event.terminal:
                return

    def close_client(self) -> None:
        """Stop delivering to a gone client without releasing the held credit."""
        self._stream.close_client()


@dataclass
class _RequestContext:
    """The identity and payload one serve request drives every attempt under."""

    invocation_id: str
    idempotency_key: str
    subject: InvocationSubject
    binding: ServeTaskResidencyBinding
    envelope: ServeRequestEnvelope
    transport: ServeTransport
    # A root-local proxy delivers response frames to a client attached to this process;
    # a forward ingress delivers them to its own client and reports only that the
    # response is committed, so nothing is enqueued here.
    local_delivery: bool


class _ServeStream:
    """One in-flight serve request's client stream and origination state.

    It implements the delivery seam resident-capacity control routes a serve
    invocation's transport and outcome through: it opens and authorizes the origin relay
    to the sidecar, teed frames enqueue for the client, the fenced terminal records a
    durable external status fact and closes the stream, and an uncertain loss re-drives
    the origination onto a fresh session under the held claim.
    """

    def __init__(self, edge: "GatedServe", context: _RequestContext) -> None:
        self._edge = edge
        self._context = context
        self.queue: asyncio.Queue[ServeEvent] = asyncio.Queue(maxsize=_TEE_QUEUE_MAX)
        self._closed = False
        self._flushed = False

    @property
    def invocation_id(self) -> str:
        return self._context.invocation_id

    def origination(self) -> ServeOrigination:
        """The origination one attempt drives; every attempt reuses the stable identity
        so admission resumes the in-flight claim rather than raising a successor."""
        binding = self._context.binding
        return ServeOrigination(
            invocation_id=self._context.invocation_id,
            idempotency_key=self._context.idempotency_key,
            task_id=self._context.invocation_id,
            call_correlation=_call_correlation(self._context.invocation_id),
            subject=self._context.subject,
            family=binding.family,
            dependency=binding.dependency(),
            profile=binding.profile(descriptor_digest=self._context.envelope.digest()),
            envelope=self._context.envelope,
            delivery=self,
        )

    def open(self, session_id: str, handoff: AdmissionHandoff) -> None:
        self._context.transport.open(
            session_id=session_id,
            invocation_id=self._context.invocation_id,
            idm=self._context.idempotency_key,
            task_id=self._context.invocation_id,
            call_correlation=_call_correlation(self._context.invocation_id),
            handoff=handoff,
            envelope=self._context.envelope,
        )

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None:
        self._context.transport.authorize(session_id, auth)

    def close_session(self, session_id: str) -> None:
        self._context.transport.close(session_id)

    def close_client(self) -> None:
        # The client stream is gone (it disconnected). Stop teeing so a still-running
        # drive cannot pin memory; the credit is untouched — only its fenced terminal
        # releases it, so the drive settles the claim independent of this stream.
        self._closed = True

    def head(self, status: int, headers: tuple[tuple[str, str], ...]) -> None:
        # The engine response head commits the client response's status and headers
        # ahead of its body. Once committed, a later loss can no longer transparently
        # re-stream under a fresh attempt's head, so the head marks the response
        # flushed: a post-head loss fails this response rather than re-driving it.
        if self._closed:
            return
        # Marking the response flushed is what stops a later loss from re-driving over
        # bytes a client already holds, so it happens for both ingresses — a forward
        # ingress reports its own commit here even though its frames are already gone.
        self._flushed = True
        if self._context.local_delivery:
            self._offer(ServeEvent(kind="head", status=status, headers=headers))

    def tee(self, payload: bytes) -> None:
        # Once the client response is closed (a post-flush loss failed it, or it already
        # terminated), a re-drive's frames are dropped rather than duplicated onto it.
        if self._closed:
            return
        self._flushed = True
        if self._context.local_delivery:
            self._offer(ServeEvent(kind="chunk", payload=payload))

    def _offer(self, event: ServeEvent) -> None:
        # A frame past the backlog bound is dropped rather than buffered without limit,
        # so a client that stops draining cannot pin unbounded memory. The terminal
        # never takes this path, so a dropped frame still yields a closing stream.
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def record_terminal(self, reason: ClaimTerminalReason, detail: str | None) -> None:
        self._edge.record_terminal(self.invocation_id, reason, detail)

    def complete(self) -> None:
        self._finish(ServeEvent(kind="done"))

    def fail(self, detail: str) -> None:
        self._finish(ServeEvent(kind="error", detail=detail))

    def _finish(self, event: ServeEvent) -> None:
        if self._closed:
            return
        self._closed = True
        self._edge.forget(self.invocation_id)
        # The terminal must reach the consumer so its stream closes; if the bounded
        # backlog is full, evict the oldest frame to make room for it.
        while True:
            try:
                self.queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

    def redrive(self) -> None:
        # When the client response can no longer receive a transparent re-stream — bytes
        # already committed to it (``_flushed``), or the client is gone (``_closed``) —
        # re-running the engine would only discard its output. Terminalize the credit
        # FAILED through the fenced path instead of re-driving, and fail this client
        # response (the caller retries as a fresh, separately admitted request). A loss
        # before any flush, with the client still connected, re-drives transparently on
        # a fresh session under the held claim, settling on its own fenced terminal.
        if self._flushed or self._closed:
            self._edge.control.fail_serve(
                self.invocation_id, self, "resident stream lost after partial delivery"
            )
            return
        self._edge.control.redrive_serve(self.origination())


class GatedServe:
    """Resolves a live serve binding, admits, and streams one task-addressed request."""

    def __init__(
        self,
        *,
        bindings: ServeBindingStore,
        terminals: ServeTerminalStore,
        control: ResidentCapacityControl,
        relay: ServeRelayExecutor,
        ingresses: ServeIngressRegistry,
        forward_transport: ServeTransport | None = None,
        persist: Callable[[], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._bindings = bindings
        self._terminals = terminals
        self.ingresses = ingresses
        self._forward_transport = forward_transport
        self.control = control
        self.relay = relay
        self._persist = persist or (lambda: None)
        self._logger = logger or logging.getLogger("gated-serve")
        self._streams: dict[str, _ServeStream] = {}

    def committed(
        self, invocation_id: str, status: int, headers: tuple[tuple[str, str], ...]
    ) -> None:
        """Mark a forward ingress's response committed to its own client.

        The ingress delivers response frames itself, so control never sees them. Without
        this signal a loss after the ingress had already written a status, headers, and
        part of a body would look like a loss before anything was delivered and re-drive
        the engine over a response the client already holds.
        """
        stream = self._streams.get(invocation_id)
        if stream is not None:
            stream.head(status, headers)

    def submit(
        self,
        principal_id: str,
        tenant: str,
        serve_task_id: str,
        envelope: ServeRequestEnvelope,
        arrived_on: ServeAccessMode,
    ) -> ServeResult:
        """Resolve the live binding, admit the request, and begin streaming.

        The caller has already authenticated and passed task-read authorization at the
        router, and its request is frozen into the transparent envelope relayed to the
        engine. A missing live binding or a method the binding does not permit raises
        before any credit. The binding's ``interface`` selects the family it was adopted
        under; it constrains neither the path nor the body, which the engine resolves.
        """
        binding = self._bindings.live(serve_task_id)
        if binding is None:
            raise BindingNotFound(serve_task_id)
        if envelope.method not in {m.upper() for m in binding.allowed_methods}:
            raise MethodNotAllowed(envelope.method)
        # A binding pins one exposure mode, so serving it on another ingress would make
        # the mode a hint rather than policy: an operator who pinned forward to keep
        # serve traffic off the root would still carry it there whenever a client used
        # the root URL.
        if arrived_on is not binding.access_mode:
            raise WrongIngress(binding.access_mode)
        # A binding pinned to an ingress this deployment has not registered is
        # unavailable: fail closed rather than serve it over the other mode.
        if self.ingresses.live(binding.access_mode) is None:
            raise IngressUnavailable(binding.access_mode)
        local = arrived_on is ServeAccessMode.PROXY
        transport: ServeTransport | None = (
            self.relay if local else self._forward_transport
        )
        if transport is None:
            raise IngressUnavailable(binding.access_mode)
        context = _RequestContext(
            invocation_id=new_invocation_id(),
            idempotency_key=new_idempotency_key(),
            subject=InvocationSubject(
                kind=InvocationSubjectKind.EXTERNAL, id=principal_id, tenant=tenant
            ),
            binding=binding,
            envelope=envelope,
            transport=transport,
            local_delivery=local,
        )
        stream = _ServeStream(self, context)
        self._streams[context.invocation_id] = stream
        self.control.originate_serve(stream.origination())
        return ServeResult(stream)

    def adopt(
        self,
        serve_task_id: str,
        access_mode: ServeAccessMode = ServeAccessMode.PROXY,
    ) -> None:
        """Adopt a live public serve task as a standing resident allocation.

        Called when the serve task reports its engine endpoint. It validates the model
        under the resident allowed-model policy, registers the task's residency binding
        and per-task allocation family, and adopts the running replica. A disallowed
        model creates neither binding, family, replica, nor route. The work runs on the
        control loop so all store access stays single-threaded with admission.
        """

        def _adopt() -> None:
            if self._bindings.live(serve_task_id) is not None:
                # Already adopted: repeated endpoint reports for a live serve task are a
                # no-op, so admission never re-registers or duplicates its allocation.
                return
            endpoint = self.control.probe_serve_endpoint(serve_task_id)
            if endpoint is None or not endpoint.model:
                return
            if not self.control.serve_model_allowed(endpoint.model):
                self._logger.info(
                    "serve task %s model %r is not in the allowed models; not adopted",
                    serve_task_id,
                    endpoint.model,
                )
                return
            try:
                interface = ServiceInterface(endpoint.interface)
            except ValueError:
                interface = ServiceInterface.CHAT
            binding = self._bindings.adopt(
                serve_task_id,
                service_ref=endpoint.model,
                interface=interface,
                isolation=None,
                adapter=None,
                adapter_source=None,
                engine_batch_key=f"{endpoint.model}|{interface.value}",
                max_output_tokens=None,
                access_mode=access_mode,
            )
            self.control.adopt_serve_replica(
                serve_task_id=serve_task_id,
                family=binding.family,
                dependency=binding.dependency(),
                endpoint=endpoint,
                binding_generation=binding.binding_generation,
            )
            self._persist()

        self.control.call_on_loop(_adopt)

    def drain(self, serve_task_id: str) -> None:
        """Drain a stopped serve task's binding and standing replica.

        The binding is marked draining and removed so new calls are refused; the
        standing replica is drained so it takes no new claims while accepted claims
        reconcile on their own terminals.
        """

        def _drain() -> None:
            if self._bindings.get(serve_task_id) is None:
                return
            self._bindings.drain(serve_task_id)
            self.control.drain_serve_replica(serve_task_id)
            self._bindings.remove(serve_task_id)
            self._persist()

        self.control.call_on_loop(_drain)

    def forget(self, invocation_id: str) -> None:
        """Drop a settled request's stream once its claim has reached a terminal."""
        self._streams.pop(invocation_id, None)

    def record_terminal(
        self, invocation_id: str, reason: ClaimTerminalReason, detail: str | None
    ) -> None:
        """Record the fenced external status fact before the credit releases."""
        status = _TERMINAL_STATUS.get(reason, ServeTerminalStatus.FAILED)
        if self._terminals.record(
            ServeStatusTerminal(
                invocation_id=invocation_id, status=status, detail=detail
            )
        ):
            self._persist()

    def reconcile_terminals(self) -> None:
        """Replay recorded external status terminals through the FSM on startup.

        A crash between recording a terminal fact and releasing its claim leaves the
        claim rehydrated UNCERTAIN with credit held. Replaying each recorded terminal
        settles it, so the crash window never strands a credit. Idempotent on a terminal
        claim.
        """
        for terminal in self._terminals.all():
            self.control.reconcile_serve_terminal(
                terminal.invocation_id, _STATUS_REASON[terminal.status]
            )
