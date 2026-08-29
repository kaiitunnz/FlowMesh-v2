"""The fabric-owned mediation between a harness adapter and the orchestration engine.

The adapter drives an agent's local turns and yields typed boundary requests before they
execute; the engine validates each against the agent's signature and authority, creates
the durable request, and assigns the idempotency key. This mediation carries a deferred
request into the engine and, on the corresponding durable outcome, injects it back into
the adapter at its originating call. It is the seam that keeps every side effect on the
fabric facade: an adapter whose native bypass paths are not disabled is refused.

The adapter runs on a worker and emits a worker-safe :class:`BoundaryRequest`; the
engine consumes the server's durable :class:`BoundaryEvent`. This module maps between
them so the co-located reference loop and the split worker/server path share one path.
"""

from shared.harness import (
    BoundaryRequest,
    DeliveredOutcome,
    HarnessAdapter,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)

from ...task.v2.representations.operators import BoundaryEventKind
from ..engine import OrchestrationEngine
from ..state import BoundaryEvent

# Boundaries whose durable outcome is immediate: a spawn's child creation and a seal
# both ack on the routing turn. A state access is a query the engine resolves inline,
# not a deferred boundary, so it carries no injected ack.
_ACK_ON_ROUTE = frozenset({BoundaryEventKind.SPAWN, BoundaryEventKind.SPAWN_SEAL})


class HarnessConfigError(RuntimeError):
    """Raised when an adapter is run in an unsupported configuration."""


def to_boundary_event(
    request: BoundaryRequest, *, continuation: str | None = None
) -> BoundaryEvent:
    """Lift a worker-emitted request into the engine's durable boundary envelope.

    The fabric-assigned identity (activation, idempotency key, invocation id, injection
    target, denial, settled value) stays the engine's to mint; only the continuation
    capsule is supplied here, from the step's returned capsule.
    """
    return BoundaryEvent(
        kind=request.kind,
        call_correlation=request.call_correlation,
        interface=request.interface,
        child_ref=request.child_ref,
        child_region_ref=request.child_region_ref,
        request_payload=request.request_payload,
        state_ref=request.state_ref,
        continuation=continuation,
    )


class AgentEpisode:
    """One agent activation's mediated run-to-yield loop over a harness adapter.

    A resume injects the pending outcomes, steps the adapter, and routes any deferred
    boundary request into the engine. An immediately durable outcome — a denial, a child
    creation, or a seal acknowledgement — is queued for the next resume; a model or tool
    result lands later through :meth:`deliver` when the dispatcher settles it.
    """

    def __init__(self, engine: OrchestrationEngine, adapter: HarnessAdapter) -> None:
        if not adapter.bypass_disabled():
            raise HarnessConfigError(
                "harness adapter must disable native tool and subagent bypass paths"
            )
        self._engine = engine
        self._adapter = adapter
        self._capsule: HarnessCapsule | None = None
        self._pending: list[DeliveredOutcome] = []

    def resume(self, task_id: str, activation_id: str) -> HarnessResult:
        """Step the adapter and route any deferred boundary into the engine."""
        outcomes = self._pending
        self._pending = []
        result = self._adapter.start(
            activation_id, capsule=self._capsule, outcomes=outcomes
        )
        self._capsule = result.capsule
        if result.kind is HarnessResultKind.BOUNDARY and result.request is not None:
            continuation = result.capsule.blob if result.capsule is not None else None
            event = to_boundary_event(result.request, continuation=continuation)
            self._engine.route_boundary_event(task_id, event)
            self._queue_immediate(activation_id, result.request)
        return result

    def deliver(self, outcome: DeliveredOutcome) -> None:
        """Queue an externally settled outcome to inject on the next resume."""
        self._pending.append(outcome)

    @property
    def capsule(self) -> HarnessCapsule | None:
        return self._capsule

    def _queue_immediate(self, activation_id: str, request: BoundaryRequest) -> None:
        corr = request.call_correlation
        if corr is None:
            return
        envelope = self._engine.boundary_envelope(activation_id, corr)
        if envelope is None:
            return
        if envelope.denial is not None:
            self.deliver(
                DeliveredOutcome(
                    call_correlation=corr,
                    idempotency_key=envelope.idempotency_key,
                    kind=OutcomeKind.DENIED,
                    denial=envelope.denial,
                )
            )
        elif request.kind in _ACK_ON_ROUTE:
            self.deliver(
                DeliveredOutcome(
                    call_correlation=corr,
                    idempotency_key=envelope.idempotency_key,
                    kind=OutcomeKind.RESULT,
                )
            )
