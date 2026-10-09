"""The orchestration engine over the transparent structured-region physical plan.

The engine owns semantic readiness: it turns settled records into ready work items,
incrementally materializing the activation graph rather than precreating attempts.
Static top-level leaf and agent operators materialize eagerly and dispatch through the
runtime; control operators (branch, merge, spawn, join, loop) settle inside the ledger
and never dispatch; spawn children and loop iterations materialize as records flow. An
agent is both dispatchable (a run-to-yield episode) and a scope owner: it yields
validated boundary requests the engine turns into durable work, and its spawn_agent
facade opens a child-init scope per selected child region, keyed by (agent activation,
region), lazily on that region's first child. Progress capabilities (child-init and
loop-time) are the closure authority — a region closes only when its combined account
seals and drains, never on an observed empty set. Scheduler/worker placement stays a
physical decision that never changes what the engine considers ready.
"""

import contextlib
import functools
import logging
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Any

from server.telemetry.tracing import NULL_CONTROL_TRACER, ControlPlaneTracer
from shared.content import ContentReference
from shared.harness import DeliveredOutcome
from shared.inference import InputResolutionBinding
from shared.outcome import OutcomeManifest
from shared.private_state import (
    OwnerFence,
    PrivateStateAttachment,
    PrivateStateBinding,
    StateBundleManifest,
)
from shared.telemetry.semconv import ControlPlaneStage, ControlPlaneWindow
from shared.tools.contract import MediatedOperationPermit
from shared.utils import (
    new_activation_id,
    new_authority_grant_id,
    new_scope_id,
    new_work_item_id,
)

from ...task.v2.representations.admission import ResidentAdmissionBinding
from ...task.v2.representations.bundle import PersistedV2Workflow
from ...task.v2.representations.operators import (
    AgentOperator,
    BranchRegion,
    EffectClass,
    LeafOperator,
    LoopContextRegion,
    OperatorKind,
    SelectionRule,
    ServiceDependency,
    SpawnRegion,
)
from ...task.v2.representations.plan import EpisodeSpec, InferenceEmbodimentMenu
from ...task.v2.representations.results import CardinalityKind, ResultDeclaration
from ...task.v2.representations.template import LogicalWorkflowTemplate
from ..guardrails import ScopeBudget
from ..outcomes import check_admissible
from ..state import (
    TERMINAL_WORK_ITEM_STATUSES,
    AcceptedInput,
    Activation,
    AuthorityGrant,
    BoundaryEvent,
    BranchDecision,
    Continuation,
    ControlState,
    ControlStatus,
    DelegatedAuthorityGrant,
    DenialKind,
    EmbodimentSelection,
    InputPreparation,
    InputResolution,
    Invocation,
    IterationResolution,
    LedgerSnapshot,
    LoopInstance,
    Occurrence,
    OccurrencePlace,
    ProgressAxis,
    ProgressCapability,
    PublicationOutcome,
    RecoveryDisposition,
    ResultPublication,
    ResultSlot,
    Scope,
    ValueRef,
    WorkflowInstance,
    WorkItem,
)
from ..telemetry import NULL_SPAN_EMITTER, TelemetrySpanEmitter
from ..tool_dispatch import AgentInputPlan, FacadeTurnGroup, ToolInvocationEnvelope
from .advance import Advance
from .attempts import AttemptLifecycle
from .authority import AuthorityLedger
from .boundaries import BoundaryLedger
from .boundary_routing import EpisodeBoundaryRouter
from .cancellation import ScopeCancellation
from .contexts import RegionContexts
from .dataflow import RegionFlow
from .embodiments import EmbodimentLedger
from .failures import FailureLedger
from .inputs import AcceptedInputLedger
from .ledger import OrchestrationLedger, control_key
from .loops import LoopProgress
from .occurrences import OccurrenceFactory
from .publications import PublicationLedger
from .scopes import ScopeProgress
from .snapshot import SnapshotCodec
from .spawns import SpawnRegions
from .topology import CONTROL_KINDS, PlanTopology, child_bodies, effect_recovery

_logger = logging.getLogger("orchestration-engine")


def _drive_span(
    engine: "OrchestrationEngine", candidate: str, window: ControlPlaneWindow
) -> AbstractContextManager[Any]:
    """Resolve the stage span for one transition, or nothing if resolving it fails.

    Resolution runs eagerly -- it reads the work item, derives ids and opens the span --
    inside a call that is about to mutate durable state. Synthesis elsewhere absorbs its
    own faults for exactly that reason; this is the one telemetry seam wrapping a
    mutation, so it absorbs them here too rather than failing the transition.
    """
    try:
        wi = engine._ledger.work_item_for_task(candidate)
        if wi is not None:
            return engine._control.episode_stage(
                ControlPlaneStage.DS_DRIVE,
                window,
                engine._ledger.workflow_instance.instance_id,
                wi.work_item_id,
            )
        return engine._control.workflow_stage(
            ControlPlaneStage.DS_DRIVE,
            window,
            engine._ledger.workflow_instance.instance_id,
        )
    except Exception as exc:
        _logger.debug("Control-plane span setup failed for %s: %s", candidate, exc)
        return contextlib.nullcontext()


def _ds_drive(
    window: ControlPlaneWindow,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap an engine transition in the explicitly-parented ``ds_drive`` span.

    Parents on the episode owning the wrapped call's first positional argument (a task
    id) when one resolves to a work item; falls back to the workflow when it does not
    — ``cancel_scope`` is called with a scope id rather than a task id, which
    resolves no work item.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(
            self: "OrchestrationEngine", candidate: str, *args: Any, **kwargs: Any
        ) -> Any:
            if not self._control.enabled:
                return fn(self, candidate, *args, **kwargs)
            with _drive_span(self, candidate, window):
                return fn(self, candidate, *args, **kwargs)

        return wrapper

    return decorator


class OrchestrationEngine:
    """Drives one workflow instance's semantic readiness over its durable ledger."""

    def __init__(
        self,
        snapshot: LedgerSnapshot,
        bundle: PersistedV2Workflow,
        *,
        budget: ScopeBudget | None = None,
        control: ControlPlaneTracer | None = None,
        emitter: TelemetrySpanEmitter | None = None,
    ) -> None:
        self._topology = PlanTopology(bundle)
        self._failures = FailureLedger()
        self._ledger = OrchestrationLedger(
            snapshot,
            self._topology,
            self._failures,
            emitter if emitter is not None else NULL_SPAN_EMITTER,
        )
        self._budget = (budget or ScopeBudget()).pinned(snapshot.max_loop_iterations)
        self._initial = Advance()
        self._control = control if control is not None else NULL_CONTROL_TRACER

        self._publication = PublicationLedger(self._ledger, self._topology)
        self._embodiments = EmbodimentLedger(self._ledger, self._topology)
        self._inputs = AcceptedInputLedger(self._ledger, self._topology)
        self._authority = AuthorityLedger(self._ledger, self._topology)
        self._boundaries = BoundaryLedger(self._ledger, self._authority)
        self._scope_progress = ScopeProgress(
            self._ledger, self._failures, self._authority, self._budget
        )
        self._flow = RegionFlow(
            self._ledger,
            self._topology,
            self._failures,
            self._publication,
            self._inputs,
            self._authority,
            self._scope_progress,
        )
        self._factory = OccurrenceFactory(
            self._ledger, self._topology, self._scope_progress
        )
        self._loops = LoopProgress(
            self._ledger,
            self._topology,
            self._publication,
            self._scope_progress,
            self._flow,
            self._factory,
            self._budget,
        )
        self._spawns = SpawnRegions(
            self._ledger,
            self._topology,
            self._inputs,
            self._authority,
            self._scope_progress,
            self._flow,
            self._factory,
            self._budget,
        )
        self._contexts = RegionContexts(
            self._ledger, self._topology, self._loops, self._spawns
        )
        self._flow.contexts = self._contexts
        self._attempt_lifecycle = AttemptLifecycle(
            self._ledger,
            self._publication,
            self._embodiments,
            self._inputs,
            self._boundaries,
            self._flow,
            self._spawns,
        )
        self._cancellation = ScopeCancellation(
            self._ledger,
            self._topology,
            self._publication,
            self._scope_progress,
            self._flow,
            self._attempt_lifecycle,
        )
        self._router = EpisodeBoundaryRouter(
            self._ledger,
            self._topology,
            self._authority,
            self._boundaries,
            self._spawns,
            self._budget,
        )
        self._codec = SnapshotCodec(
            self._ledger,
            self._topology,
            self._failures,
            self._publication,
            self._embodiments,
            self._inputs,
            self._authority,
            self._boundaries,
            self._attempt_lifecycle,
            self._budget,
        )
        self._codec.restore(snapshot)
        self._flow.adopt_stored_controls()

        # Binds the emitter to this engine's own live collections (mutated in place,
        # never reassigned) and re-derives every already-settled entity from them --
        # a no-op fresh build, the restart-recovery pass on a rehydrated one.
        self._ledger.emitter.attach(
            activations=self._ledger.activations,
            scopes=self._ledger.scopes,
            work_items=self._ledger.work_items,
            attempts=self._ledger.attempts,
            invocations=self._ledger.invocations,
            trace=self._ledger.trace,
            scope_closed=self._ledger.scope_closed,
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        instance_id: str,
        owner_id: str,
        org_id: str,
        bundle: PersistedV2Workflow,
        *,
        policy_envelope: str | None = None,
        granted_interfaces: frozenset[str] | None = None,
        budget: ScopeBudget | None = None,
        control: ControlPlaneTracer | None = None,
        emitter: TelemetrySpanEmitter | None = None,
    ) -> "OrchestrationEngine":
        """Materialize an engine from a compiled bundle.

        Eagerly materializes the static prefix (top-level leaf operators that are not
        spawn child templates) as one activation, work item, and continuation each, plus
        a control activation per region/agent operator. Spawn children and loop
        iterations materialize later, as records flow. ``granted_interfaces`` pins the
        root grant's invoke face; when omitted every requested interface is granted.
        """
        template = bundle.template
        replay = {
            b.source_ref: b.replay_contract
            for b in template.effect_boundaries
            if b.source_ref
        }
        kind_by_id = {op.operator_id: op.kind for op in template.operators}
        # A region's entry body is materialized dynamically per spawn, never dispatched
        # eagerly.
        child_body_refs = child_bodies(template)
        requested: set[str] = set()
        for op in template.operators:
            if isinstance(op, LeafOperator):
                check_admissible(
                    op.operator_id,
                    op.profile.effect,
                    replay.get(op.operator_id),
                    op.residency_only,
                )
                if op.profile.effect is EffectClass.EXTERNAL_EFFECT:
                    requested.add(op.operator_id)
            elif isinstance(op, (AgentOperator, SpawnRegion)):
                # A declared authority ceiling is granted absent an explicit policy, so
                # an agent's declared invoke/delegate faces survive attenuation.
                requested.update(op.authority.invoke)
                requested.update(op.authority.delegate)

        scope = Scope(scope_id=new_scope_id(), instance_id=instance_id, depth=0)
        invoke = requested if granted_interfaces is None else set(granted_interfaces)
        grant = AuthorityGrant(
            grant_id=new_authority_grant_id(),
            instance_id=instance_id,
            policy_id=f"policy:{instance_id}",
            invoke=tuple(sorted(invoke)),
            delegate=tuple(sorted(invoke)),
        )
        instance = WorkflowInstance(
            instance_id=instance_id,
            owner_id=owner_id,
            org_id=org_id,
            template_version=template.version.content_digest,
            plan_version=bundle.plan.plan_version.content_digest,
            policy_envelope=policy_envelope,
            root_grant_id=grant.grant_id,
        )

        preds: dict[str, set[str]] = {
            op.operator_id: set() for op in template.operators
        }
        for edge in template.edges:
            if not edge.is_forward or edge.to_op not in preds:
                continue
            if (
                kind_by_id.get(edge.from_op) is OperatorKind.SPAWN
                and kind_by_id.get(edge.to_op) is OperatorKind.JOIN
            ):
                continue  # spawn->join binding, not a record edge
            preds[edge.to_op].add(edge.from_op)

        activations: list[Activation] = []
        work_items: list[WorkItem] = []
        continuations: list[Continuation] = []
        # A region definition's members occur only as its loop or child enters it.
        members = template.definition_of()
        for op in template.operators:
            if op.operator_id in members:
                continue
            activation = Activation(
                activation_id=new_activation_id(),
                instance_id=instance_id,
                scope_id=scope.scope_id,
                operator_id=op.operator_id,
                kind=op.kind.value,
            )
            activations.append(activation)
            dispatchable = (
                isinstance(op, (LeafOperator, AgentOperator))
                and op.operator_id not in child_body_refs
            )
            if not dispatchable:
                if op.kind in CONTROL_KINDS:
                    continuations.append(
                        Continuation(
                            work_item_id=control_key(op.operator_id),
                            waiting_on=set(preds[op.operator_id]),
                        )
                    )
                continue
            effect, recovery = effect_recovery(op)
            work_item = WorkItem(
                work_item_id=new_work_item_id(),
                activation_id=activation.activation_id,
                operator_id=op.operator_id,
                legacy_task_id=op.operator_id,
                effect_class=effect,
                recovery=recovery,
                replay_contract=replay.get(op.operator_id),
            )
            work_items.append(work_item)
            required_ports = (
                set(op.declared_input_ports) if isinstance(op, AgentOperator) else set()
            )
            continuations.append(
                Continuation(
                    work_item_id=work_item.work_item_id,
                    waiting_on=set(preds[op.operator_id]),
                    required_ports=required_ports,
                )
            )

        slots = [
            ResultSlot(
                instance_id=instance_id,
                output_id=decl.output_id,
                source_operator_id=decl.source_ref,
            )
            for decl in template.result_declarations
            if decl.cardinality is CardinalityKind.SINGLETON
            and decl.source_ref not in members
        ]
        snapshot = LedgerSnapshot(
            instance=instance,
            root_scope=scope,
            root_grant=grant,
            scopes=[scope],
            activations=activations,
            work_items=work_items,
            continuations=continuations,
            result_slots=slots,
            max_loop_iterations=(budget or ScopeBudget()).max_loop_iterations,
        )
        engine = cls(snapshot, bundle, budget=budget, control=control, emitter=emitter)
        engine._initial = engine._flow.open_roots()
        return engine

    def initial_advance(self) -> Advance:
        """Ready/failed roots admitted at submission time."""
        return self._initial

    # ------------------------------------------------------------------ #
    # Physical attempt lifecycle (dispatchable leaves)
    # ------------------------------------------------------------------ #

    @_ds_drive(ControlPlaneWindow.QUEUE)
    def on_dispatched(self, task_id: str, worker_id: str | None) -> None:
        """Record a physical attempt and issue or reissue the work item's invocation."""
        self._attempt_lifecycle.on_dispatched(task_id, worker_id)

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_started(self, task_id: str) -> None:
        self._attempt_lifecycle.on_started(task_id)

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_succeeded(
        self,
        task_id: str,
        *,
        empty: bool = False,
        content: ContentReference | None = None,
    ) -> Advance:
        """Settle a work item on success and release its successors.

        ``empty`` marks a conditional-skip settlement, resolving the declared output to
        an explicit-empty publication rather than a value. ``content`` is the stored
        result the settled value is bound to; it binds once, with the settlement, so a
        later success for the same work item cannot re-point it.
        """
        return self._contexts.sweep(
            self._attempt_lifecycle.on_succeeded(task_id, empty=empty, content=content)
        )

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_failed(self, task_id: str, error: str, *, retryable: bool) -> Advance:
        """Retry a work item as a fresh attempt, or settle it and cascade failure."""
        return self._contexts.sweep(
            self._attempt_lifecycle.on_failed(task_id, error, retryable=retryable)
        )

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_returned(self, task_id: str) -> bool:
        """Close an in-flight attempt handed back without an outcome, and re-ready it;
        returns whether there was one.

        The attempt is not charged: the work item runs again under its invocation.
        """
        return self._attempt_lifecycle.on_returned(task_id)

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_uncertain(self, task_id: str, error: str | None = None) -> Advance:
        """Resolve a lost acknowledgement, route loss, or failure that may follow the
        work item's external effect.

        ``error`` is the executor's message for a reported failure; the attempt keeps
        it, and a work item that cannot run again fails with it beside the reason.
        """
        return self._contexts.sweep(
            self._attempt_lifecycle.on_uncertain(task_id, error)
        )

    def record_continuation(self, task_id: str, continuation: str) -> None:
        """Record the continuation an episode's step yielded, which its next dispatch
        resumes from."""
        self._router.record_continuation(task_id, continuation)

    def route_boundary_event(self, task_id: str, event: BoundaryEvent) -> Advance:
        """Route an episode's boundary request back into the ledger, validated first.

        For an agent operator the engine validates the request against the operator's
        boundary signature and effective authority before it creates any work; an
        undeclared event, tool, model interface, or child region settles as a durable
        typed denial injected into the continuation, never a silent no-op. A recorded
        (activation, call correlation) is a re-drive: it maps to its fabric-assigned
        idempotency key and creates no second request, so a fresh harness call id can
        never duplicate a target effect. A spawn selects one declared region by role and
        materializes one child under that region's child-init scope; a spawn seal closes
        that region; an invocation or effect records durable request state before the
        work item suspends, so waiting holds no worker; a yield persists the capsule; a
        state access records the declared reference.
        """
        return self._contexts.sweep(self._router.route_boundary_event(task_id, event))

    def route_facade_turn_group(self, task_id: str, group: FacadeTurnGroup) -> Advance:
        """Record a model turn's facade group and route each member kind-specifically.

        Each ordered member is authority-checked and recorded independently under one
        shared group id and the work item's single continuation. A spawn member admits
        one child in source order and settles at admission with a deterministic
        acceptance ack (never a child result); a search member defers as an invocation
        that holds the resume gate. A per-member budget overflow is a typed quota
        outcome that leaves an accepted sibling untouched. The lane suspends once when
        an await-outcome member is still unresolved; a group with none (a spawn-only
        turn) re-readies at once, injecting the ordered acceptance vector on its next
        step. A re-drive of the same group reuses its recorded members and creates no
        duplicate child or invocation.
        """
        return self._contexts.sweep(
            self._router.route_facade_turn_group(task_id, group)
        )

    def retries_on_loss(self, task_id: str) -> bool:
        """Whether the loss of the task's worker runs its work item again, as
        ``on_uncertain`` resolves it, rather than failing it."""
        return self._attempt_lifecycle.retries_on_loss(task_id)

    def awaits_worker_held_boundary(self, task_id: str) -> bool:
        """Whether the task is suspended on an unsettled boundary whose raw request
        only its capturing worker holds."""
        return self._boundaries.awaits_worker_held_boundary(task_id)

    def suspending_worker(self, task_id: str) -> str | None:
        """The worker whose step suspended the task on a mediated boundary that awaits
        its outcome, or None when the task is not suspended on one."""
        return self._attempt_lifecycle.suspending_worker(task_id)

    def has_open_facade_group(self, task_id: str) -> bool:
        """Whether a recorded facade group for this episode still holds the resume gate.

        A group is open only while an await-outcome member is unsettled; a spawn-only
        group closes at admission, so the fence lets the next turn issue another group.
        """
        return self._boundaries.has_open_facade_group(task_id)

    def group_dispatch_envelopes(
        self, task_id: str, group_id: str
    ) -> list[ToolInvocationEnvelope]:
        """The dispatch envelopes for a group's still-unresolved invocation members."""
        return self._boundaries.group_dispatch_envelopes(task_id, group_id)

    def deliver_boundary_outcome(self, task_id: str, call_correlation: str) -> Advance:
        """Re-ready a boundary-suspended work item once its outcome is durable.

        A mediated request's durable outcome — a model or tool result, or a denial —
        lets the episode resume: the work item returns to READY for a fresh attempt that
        injects the outcome at its originating call. Only a boundary-suspended work item
        with a recorded envelope for the call is resumed — a delivery to a running or
        settled item, or for an unrecorded call, is a no-op. An episode has one
        outstanding boundary at a time, so the recorded call is the one it awaits.
        """
        return self._contexts.sweep(
            self._router.deliver_boundary_outcome(task_id, call_correlation)
        )

    def mark_pending_outcome(self, task_id: str, call_correlation: str | None) -> None:
        """Record (or clear) the settled boundary whose outcome the next resume injects.

        Cleared as each step is processed, so a step never re-injects a prior step's
        already-consumed outcome.
        """
        self._router.mark_pending_outcome(task_id, call_correlation)

    def close_latest_attempt(self, task_id: str) -> None:
        """Settle a still-running attempt of a continuing episode, bounding its history.

        A continue-boundary re-dispatches without suspending, so its finished attempt is
        marked succeeded here rather than left perpetually running.
        """
        self._attempt_lifecycle.close_latest_attempt(task_id)

    def pending_tool_dispatches(self) -> list[ToolInvocationEnvelope]:
        """Mediated boundaries suspended with no durable outcome, for a restart.

        A model or tool boundary suspends off-lane while its handler settles it; a crash
        before that settle leaves the work item blocked with an issued invocation and no
        recorded outcome. Returns each as a full dispatch envelope so the runtime routes
        it back to its handler by (kind, interface) — a search to the broker, a model to
        the gateway — never misrouting on the recovered kind.
        """
        return self._boundaries.pending_tool_dispatches()

    def tool_dispatch_envelope(
        self, task_id: str, call_correlation: str
    ) -> ToolInvocationEnvelope | None:
        """The dispatch envelope for a recorded mediated boundary, or None if absent."""
        return self._boundaries.tool_dispatch_envelope(task_id, call_correlation)

    def pending_tool_dispatch(
        self, task_id: str, call_correlation: str
    ) -> ToolInvocationEnvelope | None:
        """The dispatch envelope for a still-pending mediated boundary, or None.

        Returns an envelope only while the boundary is suspended with no durable outcome
        and its work item is blocked, so a held re-drive re-issues exactly the off-lane
        dispatch a restart would; a settled, terminalized, or cancelled boundary yields
        nothing.
        """
        return self._boundaries.pending_tool_dispatch(task_id, call_correlation)

    def mint_operation_permit(
        self,
        task_id: str,
        call_correlation: str,
        *,
        target_id: str,
        target_generation: int,
        max_results: int,
        timeout_sec: float,
        result_char_cap: int,
        deadline_epoch: float,
        credential: str | None = None,
        deployment_credential: bool = False,
    ) -> MediatedOperationPermit | None:
        """A one-use permit for a recorded worker-originated boundary, or None.

        The engine owns the durable identity and authority: it fills the invocation,
        idempotency key, request digest, interface, subject, and the policy epoch the
        boundary was admitted under. The caller supplies the audience (the agent's
        worker and its generation), the policy-bounded budget the operation runs in, and
        the provider credential authority: an optional per-call ``credential`` resolved
        for a workflow's pinned model key, or ``deployment_credential``, which grants
        the egressing worker its deployment key.
        Returns None for a boundary that carries no digest — i.e. one the worker did not
        originate — so a re-mint never fabricates authorization the boundary lacks.
        """
        return self._router.mint_operation_permit(
            task_id,
            call_correlation,
            target_id=target_id,
            target_generation=target_generation,
            max_results=max_results,
            timeout_sec=timeout_sec,
            result_char_cap=result_char_cap,
            deadline_epoch=deadline_epoch,
            credential=credential,
            deployment_credential=deployment_credential,
        )

    def authorize_model_turn(
        self,
        task_id: str,
        call_correlation: str,
        request_digest: str,
        *,
        target_id: str,
        target_generation: int,
        timeout_sec: float,
        result_char_cap: int,
        deadline_epoch: float,
        credential: str | None = None,
        deployment_credential: bool = False,
    ) -> MediatedOperationPermit | None:
        """A one-use permit for a held agent's in-turn model egress, or None on denial.

        A harness that holds its lane across a model call has no recorded suspending
        boundary, so the engine mints from the worker's propose rather than a ledger
        event: it validates the activation's model-invoke authority against the operator
        face, mints a fresh invocation identity, fences the permit on the worker's
        ``request_digest``, and binds the audience to the agent's worker. The permit
        authorizes one egress and records no resumable state; the turn's durable
        progress rests on its turn-completion boundaries. A missing activation, or a
        model invocation outside the operator's face, returns None, which the caller
        relays as a definitive denial.
        """
        return self._router.authorize_model_turn(
            task_id,
            call_correlation,
            request_digest,
            target_id=target_id,
            target_generation=target_generation,
            timeout_sec=timeout_sec,
            result_char_cap=result_char_cap,
            deadline_epoch=deadline_epoch,
            credential=credential,
            deployment_credential=deployment_credential,
        )

    def boundary_settleable(self, task_id: str, call_correlation: str) -> bool:
        """Whether a mediated settle for this exact boundary is still legal.

        A settle is legal only while the boundary's work item is still BLOCKED — its
        episode suspended on the mediated call — and the boundary itself is unresolved.
        A cancelled or otherwise non-blocked work item, or an already-settled or denied
        boundary, is absorbing: a late or duplicate model, tool, or resident delivery is
        audit evidence only. It must not stamp an outcome, re-ready the episode, replace
        the durable correlation, or release a resident credit a second time. Gate on the
        BLOCKED work item, not a task record's status, since cancellation can leave the
        record CANCELLING while the work item is already CANCELLED.
        """
        return self._boundaries.boundary_settleable(task_id, call_correlation)

    def settle_boundary_outcome(
        self,
        task_id: str,
        call_correlation: str,
        *,
        value: str | None = None,
        ref: OutcomeManifest | None = None,
    ) -> Advance:
        """Persist a mediated outcome and re-ready the suspended episode.

        The outcome — an inline ``value`` or a reference-backed ``ref`` manifest — lands
        durably on the boundary envelope so a re-dispatch injects it and a restart
        rehydrates it; the item then returns to READY for a fresh attempt.
        """
        return self._contexts.sweep(
            self._router.settle_boundary_outcome(
                task_id, call_correlation, value=value, ref=ref
            )
        )

    def terminalize_boundary_invocation(
        self, task_id: str, call_correlation: str
    ) -> str | None:
        """Record a settled mediated boundary's invocation as terminal in the ledger.

        Returns the durable ``invocation_id`` so a control-plane consumer bound to it —
        the resident-capacity admission credit — can advance from this fenced ``DS``
        outcome. The transition is idempotent and never regresses an ambiguity-terminal
        outcome.
        """
        return self._router.terminalize_boundary_invocation(task_id, call_correlation)

    def boundary_invocation_completed(self, invocation_id: str) -> bool | None:
        """Whether a terminal boundary invocation completed with an outcome; None while
        it is unknown or not terminal."""
        return self._boundaries.boundary_invocation_completed(invocation_id)

    def terminalize_unsettled_invocations(
        self, task_ids: Iterable[str] | None = None
    ) -> list[str]:
        """Terminalize the unsettled mediated boundary invocations of the given tasks'
        activations, or of every activation; one already terminal is left as it is.

        Returns the ``invocation_id``s it terminalized so a control-plane consumer bound
        to them — a resident-capacity admission credit — releases from this fenced
        terminal.
        """
        return self._router.terminalize_unsettled_invocations(task_ids)

    def suspended_boundary_tasks(self) -> list[str]:
        """Tasks suspended at an unsettled mediated boundary.

        Such a task's worker released the lane, so it holds no dispatch and returns no
        terminal; a task mid-step is not among them.
        """
        return self._boundaries.suspended_boundary_tasks()

    def episode_context(
        self, task_id: str
    ) -> tuple[str | None, tuple[DeliveredOutcome, ...]]:
        """The durable capsule and pending injected outcome for an agent's next step.

        Rebuilt from the ledger, never in-memory episode state: the capsule is the work
        item's continuation, and the one pending outcome is reconstructed from its
        settled boundary envelope, so a re-dispatch after a restart carries it again.
        """
        return self._boundaries.episode_context(task_id)

    # ------------------------------------------------------------------ #
    # Structured-region API (control settlement is internal; dynamic
    # cardinality is controller-driven over generic regions)
    # ------------------------------------------------------------------ #

    def spawn_child(self, spawn: str, *, operator_id: str | None = None) -> str:
        """Materialize one child activation under a spawn/agent's open child scope.

        ``spawn`` is a region handle — an operator id for the non-recursive case, or an
        opener activation id for a specific recursion level. Rejects a child after a
        definitive denial, or after the child-init capability is sealed or revoked
        (late-child prevention). When the child body is itself a scope opener, opens a
        nested scope owned by the child activation, so grandchildren (and recursive re-
        entries) attenuate from the child grant at ``depth+1``. Returns the child
        activation id, which addresses that nested scope.
        """
        return self._spawns.spawn_child(spawn, operator_id=operator_id)

    def materialize_child(
        self,
        spawn: str,
        *,
        operator_id: str | None = None,
        value_ref: ValueRef | None = None,
    ) -> Advance:
        """Materialize one dispatchable child leaf and admit it as ready work.

        Unlike :meth:`spawn_child`, the child is given a stable dispatchable identity,
        carries its child-init input as ``value_ref``, and is readied for a physical
        attempt. The child body must be a leaf, or an agent that itself owns a child-
        init scope; a spawn/loop child body is not live-dispatchable and stays a trace-
        level :meth:`spawn_child`.
        """
        return self._contexts.sweep(
            self._spawns.materialize_child(
                spawn, operator_id=operator_id, value_ref=value_ref
            )
        )

    def create_fanout_child(self, spawn: str, value_ref: ValueRef) -> str:
        """Create one producer-fanout child (unadmitted) and return its task id.

        The child carries ``value_ref``, a frozen reference to one element of the
        producer's collection, as its child-init input. It stays blocked on its input
        manifest until the runtime records the child-entry accepted input.
        """
        return self._spawns.create_fanout_child(spawn, value_ref)

    def agent_entry_port(self, operator_id: str) -> str | None:
        """The single declared input port of an agent child body, or None.

        A spawn child agent declares exactly one input port (compile-enforced); a leaf
        child or an agent with no declared input has no entry port.
        """
        return self._topology.agent_entry_port(operator_id)

    def record_accepted_input(self, accepted: AcceptedInput) -> None:
        """Record a durable accepted input on an agent's target port (idempotent)."""
        self._inputs.record_accepted_input(accepted)

    def accepted_inputs_for(self, activation_id: str) -> tuple[AcceptedInput, ...]:
        """The recorded accepted inputs for one activation, ordered by ordinal."""
        return self._inputs.accepted_inputs_for(activation_id)

    def accepted_inputs_for_task(self, task_id: str) -> tuple[AcceptedInput, ...]:
        """The recorded accepted inputs for a task's activation, ordered by ordinal."""
        return self._inputs.accepted_inputs_for_task(task_id)

    def blocked_input_agents(self) -> list[str]:
        """Task ids of agents blocked on an unsatisfied declared-input manifest."""
        return self._inputs.blocked_input_agents()

    def reconsider_admission(self, task_id: str) -> Advance:
        """Re-attempt admission of a work item after its input manifest changed."""
        return self._contexts.sweep(self._flow.reconsider_admission(task_id))

    def agent_input_plan(self, task_id: str) -> AgentInputPlan | None:
        """The engine's per-port input membership for an agent, resolved by the runtime.

        Covers edge-bound ports (a direct producer or a join/merge aggregate) whose
        sources have settled and are not yet recorded. Membership and ordering are the
        engine's decision — declared by the edge and the join's child order, never
        arrival. A fan-out child's inline entry port is minted at materialization and is
        not returned here.
        """
        return self._inputs.agent_input_plan(task_id)

    def seal_spawn(self, spawn: str) -> Advance:
        """Seal a spawn's child-init capability; no further children may be created."""
        return self._contexts.sweep(self._spawns.seal_spawn(spawn))

    def revoke_spawn(self, spawn: str) -> None:
        """Revoke a spawn's child-init capability as a progress transition.

        Distinct from sealing: revocation withdraws the capability, while sealing marks
        a producer done. Both close the child-init axis once outstanding children drain.
        """
        self._spawns.revoke_spawn(spawn)

    def settle_child(
        self,
        child_activation_id: str,
        *,
        outcome: PublicationOutcome = PublicationOutcome.SUCCESS,
        value_ref: ValueRef | None = None,
    ) -> Advance:
        """Record a child activation's terminal outcome and drain its capability."""
        return self._contexts.sweep(
            self._spawns.settle_child(
                child_activation_id, outcome=outcome, value_ref=value_ref
            )
        )

    def pending_branch_reads(self) -> list[tuple[str, ValueRef]]:
        """Each branch occurrence awaiting a selector read, with the value it reads."""
        return self._flow.pending_branch_reads()

    def selection_rule(self, branch: str) -> SelectionRule | None:
        """The selection rule a branch occurrence routes by."""
        op = self._topology.operators.get(self._ledger.occurrence(branch).operator_id)
        return op.rule if isinstance(op, BranchRegion) else None

    @_ds_drive(ControlPlaneWindow.POST_START)
    def accept_branch_selection(
        self, branch: str, value: Any, *, error: str | None = None
    ) -> Advance:
        """Route a branch occurrence by the selector value read from its input.

        The value selects one declared output port; every other port resolves dead.
        A value selecting nothing, or an ``error`` reading the input, fails the branch.
        A decision is recorded once and never revised.
        """
        return self._contexts.sweep(
            self._flow.accept_branch_selection(branch, value, error=error)
        )

    def branch_decision(self, branch: str) -> BranchDecision | None:
        return self._ledger.branch_decisions.get(branch)

    def spawn_input(self, spawn: str) -> ValueRef | None:
        """The value a live spawn occurrence fans out over."""
        return self._flow.spawn_input(spawn)

    def awaiting_fanouts(self) -> list[tuple[str, ValueRef]]:
        """Each live spawn occurrence still to fan out over its input, with that
        input."""
        return self._flow.awaiting_fanouts()

    def awaits_control_reads(self) -> bool:
        """Whether the instance still waits on a value read to route a branch or fan
        out a spawn, which no task of its own holds open."""
        return bool(self.pending_branch_reads() or self.awaiting_fanouts())

    def has_unsettled_tasks(self) -> bool:
        """Whether any work item a task runs as is still to settle."""
        return any(
            wi.legacy_task_id and wi.status not in TERMINAL_WORK_ITEM_STATUSES
            for wi in self._ledger.work_items.values()
        )

    def task_work_items(self) -> list[WorkItem]:
        """Every work item a task runs as."""
        return [wi for wi in self._ledger.work_items.values() if wi.legacy_task_id]

    def control_failure(self) -> str | None:
        """Why the instance failed outside any task, if it has: the whole instance, or
        a control occurrence, failed."""
        if (reason := self._failures.instance_failure) is not None:
            return reason
        failed = sorted(
            (key, state.reason or "control failed")
            for key, state in self._ledger.control_states.items()
            if state.status is ControlStatus.FAILED
        )
        if failed:
            return failed[0][1]
        if self._failures.failed_regions:
            return f"region {min(self._failures.failed_regions)} failed"
        return None

    def spawn_handle(self, spawn: str) -> str:
        """The handle a spawn occurrence's children are created and sealed under."""
        occurrence = self._ledger.occurrence(spawn)
        return occurrence.activation_id or occurrence.operator_id

    def spawn_region(self, spawn: str) -> SpawnRegion | None:
        """The spawn region a spawn occurrence runs."""
        op = self._topology.operators.get(self._ledger.occurrence(spawn).operator_id)
        return op if isinstance(op, SpawnRegion) else None

    @_ds_drive(ControlPlaneWindow.POST_START)
    def fail_control(self, occurrence: str, reason: str) -> Advance:
        """Settle a pending control occurrence as a declared failure, as an input it
        could not read fails it."""
        advance = Advance()
        self._flow.fail_control(occurrence, reason, advance)
        return self._contexts.sweep(advance)

    @_ds_drive(ControlPlaneWindow.POST_START)
    def enter_definition_child(
        self, spawn: str, index: int, element: ValueRef
    ) -> Advance:
        """Create the child of a spawn that enters a region definition for the element
        at ``index`` of its fan-out, once."""
        return self._contexts.sweep(
            self._spawns.enter_definition_child(spawn, index, element)
        )

    def loop_instance(self, loop: str) -> LoopInstance | None:
        """The loop instance a loop occurrence entered, if it entered."""
        scope_id = self._ledger.loop_by_occurrence.get(loop)
        return self._ledger.loop_instances.get(scope_id) if scope_id else None

    def iteration(self, loop: str, time: int) -> IterationResolution | None:
        """How a loop occurrence's time resolved, if it has."""
        scope_id = self._ledger.loop_by_occurrence.get(loop)
        return self._ledger.iterations.get((scope_id, time)) if scope_id else None

    def control_state(self, occurrence: str) -> ControlState | None:
        return self._ledger.control_states.get(occurrence)

    def occurrences(self, operator_id: str) -> list[Occurrence]:
        """Every occurrence of an operator inside a region definition so far."""
        return [
            o for o in self._ledger.occurrences.values() if o.operator_id == operator_id
        ]

    def occurrence_of(self, task_id: str) -> Occurrence | None:
        """The region-definition occurrence a task runs, or None for a root task."""
        wi = self._ledger.work_item_for_task(task_id)
        key = (
            self._ledger.occurrence_by_activation.get(wi.activation_id) if wi else None
        )
        return self._ledger.occurrences.get(key) if key else None

    def occurrence_place(self, task_id: str) -> OccurrencePlace | None:
        """Where a task runs inside a region definition, by authored names: the member
        it runs, the child context it runs in, and its loop times; None at the root."""
        occurrence = self.occurrence_of(task_id)
        if occurrence is None:
            return None
        names = {e.logical_ref: e.source_id for e in self.template.source_map}
        time: list[tuple[str, int]] = []
        for frame in occurrence.time:
            instance = self._ledger.loop_instances.get(frame.loop)
            loop_op = (
                self._ledger.occurrence(instance.occurrence).operator_id
                if instance is not None
                else frame.loop
            )
            time.append((names.get(loop_op, loop_op), frame.iteration))
        return OccurrencePlace(
            member=names.get(occurrence.operator_id, occurrence.operator_id),
            context=occurrence.context_id or None,
            time=tuple(time),
        )

    def legacy_control_regions(self) -> list[str]:
        """Branch and loop operators stored before they had a runnable contract."""
        return [
            op.operator_id
            for op in self._topology.operators.values()
            if (isinstance(op, BranchRegion) and op.rule is None)
            or (isinstance(op, LoopContextRegion) and op.body_ref is None)
        ]

    def deny_spawn(
        self, spawn_op: str, interface: str, *, kind: DenialKind = DenialKind.AUTHORITY
    ) -> None:
        """Record a definitive dynamic authorization denial at a spawn site.

        A denial creates no child activation and no resident claim, and is separate from
        quota/rate/capacity/transport outcomes. It does not seal the child-init
        capability: grant denial and cardinality sealing stay distinct.
        """
        self._authority.deny_spawn(spawn_op, interface, kind=kind)

    def can_delegate(self, region_op: str, interface: str) -> bool:
        """Whether a child of ``region_op`` may itself delegate ``interface``."""
        return self._authority.can_delegate(region_op, interface)

    # ------------------------------------------------------------------ #
    # Cancellation (a durable semantic event)
    # ------------------------------------------------------------------ #

    def cancel_instance(self) -> Advance:
        """Cancel the whole workflow instance: the root scope and every descendant."""
        self._failures.instance_cancelled = True
        return self.cancel_scope(self._ledger.root_scope.scope_id)

    @property
    def template(self) -> LogicalWorkflowTemplate:
        """The logical template the instance runs."""
        return self._topology.bundle.template

    def instance_cancelled(self) -> bool:
        """Whether the whole workflow instance was cancelled."""
        return self._failures.instance_cancelled

    def fail_instance(self, reason: str) -> Advance:
        """Fail the whole workflow instance as a recorded terminal event.

        No scope admits another child, every unsettled leaf or agent settles as a
        declared failure, and every unpublished declared output resolves to one.
        """
        self._failures.instance_failure = self._failures.instance_failure or reason
        return self._fail_scope_tree(self._ledger.root_scope.scope_id, reason)

    @_ds_drive(ControlPlaneWindow.POST_START)
    def _fail_scope_tree(self, scope_id: str, reason: str) -> Advance:
        return self._contexts.sweep(
            self._cancellation.fail_scope_tree(scope_id, reason)
        )

    @_ds_drive(ControlPlaneWindow.POST_START)
    def cancel_scope(self, scope_id: str) -> Advance:
        """Cancel a scope subtree as a durable, recorded-before-terminal event.

        Over the scope and each descendant, in order: record the cancellation; revoke
        the child-init (and loop-time) capability — a transition distinct from sealing;
        apply the residual-child policy to materialized children; transition the
        remaining in-flight work items to ``CANCELLED``; revoke the scope's authority
        grant — distinct from the child-init revoke; and resolve declared outputs to
        their cancellation / no-winner outcome.
        """
        return self._contexts.sweep(self._cancellation.cancel_scope(scope_id))

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def effective_invoke_face(self, task_id: str) -> tuple[str, ...]:
        """The interfaces this agent activation may invoke.

        The same effective face the engine authorizes ordinary boundaries against: the
        activation's scope grant under its operator ceiling and the policy envelope. A
        spawned activation reads its own delegated grant, which its parent already
        attenuated, so an interface an ancestor withheld is absent here even where the
        operator's own declared ceiling names it. A task that is not an agent invokes
        nothing through this face.
        """
        return self._authority.effective_invoke_face(task_id)

    def output_publication(
        self,
        output_id: str,
        scope_id: str | None = None,
        logical_key: str | None = None,
        sequence: int | None = None,
    ) -> ResultPublication | None:
        """The terminal publication of exactly one slot, if it has one."""
        return self._publication.output_publication(
            output_id, scope_id, logical_key, sequence
        )

    def output_slots(self, output_id: str) -> list[ResultSlot]:
        """Every slot a declared output holds so far, pending or published."""
        return self._publication.output_slots(output_id)

    def output_slot(
        self,
        output_id: str,
        scope_id: str | None = None,
        logical_key: str | None = None,
        sequence: int | None = None,
    ) -> ResultSlot | None:
        """Exactly one slot of a declared output, if it holds one."""
        return self._publication.output_slot(output_id, scope_id, logical_key, sequence)

    def published_outputs(self) -> list[tuple[str, ResultDeclaration]]:
        """Each published declaration with the public name it was authored under."""
        return self._topology.published_outputs()

    def resolve_legacy_task(self, task_id: str) -> ResultPublication | None:
        """Resolve a legacy task id's induced output slot (compatibility adapter)."""
        return self._publication.resolve_legacy_task(task_id)

    def legacy_task_value(
        self, task_id: str
    ) -> tuple[PublicationOutcome, ValueRef | None] | None:
        """The settled value a legacy task id reads as, or None while it is unsettled.

        A task compiled from the source resolves its induced output slot. A task the
        engine materialized at run time — a spawned child, a later loop iteration — has
        no slot of its own, so it reads as the value its work item settled with. Either
        way the value is the one bound at settlement and never re-pointed.
        """
        return self._publication.legacy_task_value(task_id)

    def failure_reason(self, task_id: str) -> str | None:
        """Why a task settled as a declared failure, or None for one that has not."""
        return self._failures.failure_reason(task_id)

    def declared_failures(self) -> dict[str, str]:
        """Every task settled as a declared failure, with why."""
        return self._failures.declared_failures()

    def recovery_disposition(self, task_id: str) -> RecoveryDisposition | None:
        """Whether the task's operation may be recomputed or must be restored."""
        return self._ledger.recovery_disposition(task_id)

    def boundary_envelope(
        self, activation_id: str, call_correlation: str
    ) -> BoundaryEvent | None:
        """The durable envelope recorded for one mediated facade call, if any.

        Carries the fabric-assigned idempotency key, the causal invocation id, and the
        outcome (or denial) the continuation resumes with.
        """
        return self._boundaries.boundary_envelope(activation_id, call_correlation)

    def contract_trace(self) -> list[tuple[str, str]]:
        """A compact (kind, subject) projection of the trace for test inspection."""
        return self._ledger.contract_trace()

    def capability(
        self, scope_id: str | None, axis: ProgressAxis
    ) -> ProgressCapability | None:
        return self._scope_progress.capability(scope_id, axis)

    def scope_for(self, region_op: str) -> str | None:
        return self._ledger.scope_id_for(region_op)

    def region_scope_for(self, agent_activation: str, role: str) -> str | None:
        """The child-init scope an agent's declared role region opened, if entered."""
        return self._ledger.region_scope_for(agent_activation, role)

    def grant_for(self, region_op: str) -> DelegatedAuthorityGrant | None:
        return self._authority.grant_for(region_op)

    def region_closed(self, region_op: str) -> bool:
        return self._ledger.region_closed(region_op)

    def fanout_spawn(self, operator_id: str) -> str | None:
        """The root spawn that fans out over an operator's whole result, if any."""
        return self._topology.fanout_spawn(operator_id)

    def spawn_awaits_children(self, spawn_op: str) -> bool:
        """Whether a spawn has yet to fan out: unopened, or open and not sealed.

        A failed spawn never fans out.
        """
        return self._scope_progress.spawn_awaits_children(spawn_op)

    def spawn_is_open(self, spawn_op: str) -> bool:
        """Whether a spawn's child-init capability still admits new children.

        False once the spawn has sealed or revoked, or before its child-init scope
        opens, so a re-driven fan-out over an already-closed spawn is a clean no-op. A
        read-only query: it never opens a scope.
        """
        return self._scope_progress.spawn_is_open(spawn_op)

    def embodiment_menu(self, task_id: str) -> InferenceEmbodimentMenu | None:
        """The finite set of embodiments a task's plan node offers, if it offers one."""
        return self._ledger.embodiment_menu(task_id)

    def embodiment_selection(self, task_id: str) -> EmbodimentSelection | None:
        """The embodiment a task is already bound to, if one was resolved."""
        return self._embodiments.embodiment_selection(task_id)

    def embodiment_pinned(self, task_id: str) -> bool:
        """Whether a resolved embodiment is committed to the run that carries it.

        An embodiment changes only before its candidate-specific issue or delivery. A
        resident candidate commits at its invocation, after which reconciliation reuses
        that invocation and its idempotency and credit path rather than running the
        other embodiment; a local candidate carries no invocation and commits when its
        attempt is issued, which is where it was delivered to a worker.
        """
        return self._embodiments.embodiment_pinned(task_id)

    def record_embodiment_selection(
        self, task_id: str, alternative_id: str, selector: str, evidence: str
    ) -> EmbodimentSelection | None:
        """Bind a task to one embodiment durably, before its worker message goes out.

        A pinned selection is kept: the caller receives the standing one rather than a
        replacement.
        """
        return self._embodiments.record_embodiment_selection(
            task_id, alternative_id, selector, evidence
        )

    def input_resolution(self, task_id: str) -> InputResolution | None:
        """The resolution a task's inputs were materialized under, if one exists."""
        return self._embodiments.input_resolution(task_id)

    def input_preparation(self, task_id: str) -> InputPreparation | None:
        """The preparation dispatch a task's inputs are being resolved by, if any."""
        return self._embodiments.input_preparation(task_id)

    def on_input_preparation_dispatched(
        self, task_id: str, worker_id: str | None
    ) -> None:
        """Record that a work item's inputs are being resolved on a worker.

        This deliberately mints neither an invocation nor an attempt: both are
        candidate-specific commitments, and a work item whose inputs are still being
        resolved has not chosen an embodiment to commit to.
        """
        self._embodiments.on_input_preparation_dispatched(task_id, worker_id)

    def record_input_resolution(
        self,
        task_id: str,
        binding: InputResolutionBinding,
        reference: ContentReference | None = None,
    ) -> InputResolution | None:
        """Record how a work item's inputs resolved, before its embodiment runs.

        A standing resolution is kept: a re-drive that reaches the same request records
        nothing new, and one that reaches a different request leaves the recorded
        binding in place for the reconciliation that compares against it.

        A resolution carrying its request's reference commits both together, so the
        request a later run hydrates is durable exactly when the binding proving what it
        is becomes durable.
        """
        return self._embodiments.record_input_resolution(task_id, binding, reference)

    def episode_spec(self, task_id: str) -> EpisodeSpec | None:
        """The run-to-yield episode a task's operator lowers to, if the plan cut it."""
        return self._ledger.episode_spec(task_id)

    def work_item(self, task_id: str) -> WorkItem | None:
        return self._ledger.work_item_for_task(task_id)

    def child_input(self, task_id: str) -> ValueRef | None:
        """The child-init input a spawned child task runs on, if it has one."""
        return self._inputs.child_input(task_id)

    def agent_operator(self, task_id: str) -> AgentOperator | None:
        """The agent operator a dispatched task realizes, resolving its work item."""
        return self._ledger.agent_operator(task_id)

    def private_state_owner(self, task_id: str) -> OwnerFence | None:
        """The single holder that can supply an agent task's bound generation."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or self._ledger.agent_operator(task_id) is None:
            return None
        return self._ledger.private_state.owner(wi.activation_id)

    def private_state_holders(self) -> list[OwnerFence]:
        """The holders whose sealed private state an unsettled activation resumes on."""
        holders: list[OwnerFence] = []
        for lineage in self._ledger.private_state.lineages():
            if (owner := lineage.binding.owner) is None:
                continue
            wi_id = self._ledger.wi_by_activation.get(
                lineage.binding.reference.activation_id
            )
            wi = self._ledger.work_items.get(wi_id) if wi_id is not None else None
            if wi is not None and wi.status not in TERMINAL_WORK_ITEM_STATUSES:
                holders.append(owner)
        return holders

    def grant_private_state(
        self, task_id: str, worker_id: str, incarnation: int
    ) -> tuple[PrivateStateBinding, PrivateStateAttachment] | None:
        """Bind a holder to an agent task's private state for one dispatch.

        A settled or cancelled work item is granted nothing: a dispatch still in flight
        when its activation ended cannot take back the write authority the terminal
        released.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return None
        if self._ledger.agent_operator(task_id) is None:
            return None
        binding = self._ledger.private_state.ensure(
            wi.activation_id, self._ledger.workflow_instance.instance_id
        )
        attachment = self._ledger.private_state.attach(
            wi.activation_id, worker_id, incarnation
        )
        self._ledger.emit(
            "private_state_attached",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={
                "reference": binding.reference.reference_id,
                "generation": str(binding.generation),
                "write_epoch": str(attachment.write_epoch),
            },
        )
        return binding, attachment

    def seal_private_state(
        self, task_id: str, manifest: StateBundleManifest, write_epoch: int
    ) -> None:
        """Record the generation a holder sealed at its quiescence fence."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return
        self._ledger.private_state.seal(wi.activation_id, manifest, write_epoch)
        self._ledger.emit(
            "private_state_sealed",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={
                "reference": manifest.reference_id,
                "generation": str(manifest.generation),
            },
        )

    def service_dependency(self, task_id: str) -> ServiceDependency | None:
        """The normalized resident dependency a dispatched task consumes, or None."""
        return self._ledger.service_dependency(task_id)

    def resident_admission_binding(
        self, workflow_id: str, task_id: str
    ) -> ResidentAdmissionBinding | None:
        """The dependency a task consumes joined with its own plan node's annotations.

        The node is the one the task's operator lowered to; an unresolved embodiment
        menu carries its resident annotations per candidate, so it contributes none.
        """
        return self._ledger.resident_admission_binding(workflow_id, task_id)

    def invocation_for_task(self, task_id: str) -> Invocation | None:
        return self._ledger.invocation_for_task(task_id)

    @property
    def instance(self) -> WorkflowInstance:
        return self._ledger.workflow_instance

    def work_item_id_for_task(self, task_id: str) -> str | None:
        """The episode (work item) id backing a legacy task id, or None."""
        return self._ledger.work_item_id_for_task(task_id)

    def latest_attempt_open(self, task_id: str) -> bool:
        """Whether the work item's latest attempt still expects a terminal report.

        A reroute that re-enqueues an episode closes the attempt that produced the turn,
        so a completion whose attempt is already closed is a superseded replay —
        applying it would preempt the live turn. A genuine terminal report lands while
        its attempt is still issued or running.
        """
        return self._attempt_lifecycle.latest_attempt_open(task_id)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_snapshot(self) -> LedgerSnapshot:
        return self._codec.to_snapshot()

    def reconcile_failure(self, task_id: str) -> list[str]:
        """Fail what a task's settled failure left standing downstream of it.

        Returns the legacy task ids newly failed; empty for a task that has not failed,
        for a spawned child, whose failure drains its scope instead, and once the
        downstream has already failed.
        """
        return self._flow.reconcile_failure(task_id)

    def fail_undeliverable_region_inputs(self) -> list[str]:
        """Fail each task reading a region of an agent that runs only as a spawned
        child, and everything downstream of it.

        Every scope of such a region is nested, so its join never delivers at the root
        and the reader would wait forever. Returns the legacy task ids newly failed.
        """
        return self._flow.fail_undeliverable_region_inputs()

    def reconcile_pending(self, task_id: str) -> bool:
        """Re-derive readiness for a task whose durable record shows PENDING.

        Returns whether the work item is ready to admit. A work item the snapshot still
        shows in flight — a crash after a retry persisted the PENDING record but before
        the ledger caught up — is reset to ready with its lost attempt marked, so the
        retry is not orphaned; a work item whose predecessors have not all settled, or
        whose declared inputs have not all been accepted, stays blocked.
        """
        return self._attempt_lifecycle.reconcile_pending(task_id)
