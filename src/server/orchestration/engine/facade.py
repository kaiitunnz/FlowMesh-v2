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
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from enum import Enum, auto
from typing import Any

from server.telemetry.tracing import NULL_CONTROL_TRACER, ControlPlaneTracer
from shared.content import ContentReference
from shared.harness import DeliveredOutcome, OutcomeKind
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
    new_attempt_id,
    new_authority_grant_id,
    new_idempotency_key,
    new_invocation_id,
    new_mediated_permit_id,
    new_scope_id,
    new_work_item_id,
)

from ...task.v2.representations.admission import ResidentAdmissionBinding
from ...task.v2.representations.bundle import PersistedV2Workflow
from ...task.v2.representations.operators import (
    AgentOperator,
    BoundaryEventKind,
    BranchRegion,
    EffectClass,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    OperatorKind,
    RecoveryClass,
    ResidualPolicy,
    ServiceDependency,
    SpawnRegion,
    spawned_only_region_owners,
)
from ...task.v2.representations.plan import EpisodeSpec, InferenceEmbodimentMenu
from ...task.v2.representations.results import CardinalityKind, ResultDeclaration
from ...utils.time import now_iso
from ..guardrails import ScopeBudget
from ..outcomes import (
    attenuate,
    check_admissible,
    is_compensable,
    is_replayable,
    next_on_acknowledge,
    next_on_reissue,
    next_on_terminal,
    next_on_uncertain,
)
from ..state import (
    TERMINAL_INVOCATION_STATES,
    TERMINAL_WORK_ITEM_STATUSES,
    AcceptedInput,
    AcceptedInputMember,
    Activation,
    Attempt,
    AttemptStatus,
    AuthorityDecision,
    AuthorityDecisionKind,
    AuthorityGrant,
    BoundaryEvent,
    CapabilityStatus,
    Continuation,
    DelegatedAuthorityGrant,
    DenialKind,
    EffectReceipt,
    EmbodimentSelection,
    InputPreparation,
    InputResolution,
    Invocation,
    InvocationState,
    LedgerSnapshot,
    ProgressAxis,
    ProgressCapability,
    PublicationOutcome,
    Record,
    RecoveryDisposition,
    RegionAggregateMember,
    RegionJoinAggregate,
    ResultPublication,
    ResultSlot,
    Scope,
    ValueRef,
    WorkflowInstance,
    WorkItem,
    WorkItemStatus,
    slot_identity,
)
from ..telemetry import NULL_SPAN_EMITTER, TelemetrySpanEmitter
from ..tool_dispatch import (
    MODEL_INTERFACE,
    AgentInputPlan,
    FacadeCallMember,
    FacadeCompletionMode,
    FacadeTurnGroup,
    GrantSnapshot,
    InputMemberPlan,
    InputPortPlan,
    ToolInvocationEnvelope,
    ToolOutcomeStatus,
)
from .advance import Advance, RegionError, dependency_failed
from .failures import FailureLedger
from .ledger import OrchestrationLedger
from .topology import _CONTROL_KINDS, PlanTopology

_CHILD_INIT_OPENERS = frozenset({OperatorKind.SPAWN, OperatorKind.AGENT})
# Boundary kinds whose exactly-once rests on the durable correlation key, so a mediated
# one must carry a call correlation or it could duplicate a target effect on re-drive.
_DEDUP_CAPABLE = frozenset(
    {
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.INVOCATION,
        BoundaryEventKind.EXTERNAL_EFFECT,
    }
)
# Boundary kinds an off-lane handler settles while their episode is suspended.
_MEDIATED_BOUNDARY_KINDS = frozenset(
    {BoundaryEventKind.INVOCATION, BoundaryEventKind.EXTERNAL_EFFECT}
)
_EARLY_JOINS = frozenset(
    {JoinCompletion.ANY, JoinCompletion.FIRST_K, JoinCompletion.PREDICATE}
)
_AMBIGUITY_TERMINAL_REASON = "ambiguity-terminal effect"
_DECLARED_FAILURE_REASON = "declared-failure obligation"


def _ambiguity_terminal_reason(error: str | None) -> str:
    """Why a work item that cannot run again failed: its executor's own message, when
    it reported one, beside the reason."""
    if error is None:
        return _AMBIGUITY_TERMINAL_REASON
    return f"{error} ({_AMBIGUITY_TERMINAL_REASON})"


_OPEN_ATTEMPT_STATUSES = frozenset({AttemptStatus.ISSUED, AttemptStatus.RUNNING})


class _LossResolution(Enum):
    """How the loss of a work item's worker resolves it."""

    NOTHING = auto()
    PREPARE_AGAIN = auto()
    BOUNDARY_FAILS = auto()
    RUN_AGAIN = auto()
    FAILS = auto()


_RERUN_ON_LOSS = frozenset({_LossResolution.PREPARE_AGAIN, _LossResolution.RUN_AGAIN})


def _control_key(operator_id: str) -> str:
    return f"control:{operator_id}"


def _effect_recovery(op: LogicalOperator | None) -> tuple[EffectClass, RecoveryClass]:
    """A dispatchable operator's effect/recovery: a leaf's profile, else pure/recompute.

    An agent episode is itself pure and recomputable; its mediated effects flow through
    boundary events rather than the episode's own effect class.
    """
    if isinstance(op, LeafOperator):
        return op.profile.effect, op.profile.recovery
    return EffectClass.PURE, RecoveryClass.RECOMPUTE


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


def _rekeyed_publications(
    slots: Iterable[ResultSlot], publications: Iterable[ResultPublication]
) -> dict[str, ResultPublication]:
    """Index publications by slot identity, re-keying any stored under the unscoped
    key format."""
    current = {slot.legacy_slot_key: slot.slot_key for slot in slots}
    identities = set(current.values())
    indexed: dict[str, ResultPublication] = {}
    for publication in publications:
        key = publication.slot_key
        if key not in identities and (rekeyed := current.get(key)) is not None:
            publication = publication.model_copy(update={"slot_key": rekeyed})
        indexed[publication.slot_key] = publication
    return indexed


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
        self._budget = budget or ScopeBudget()
        self._initial = Advance()
        self._control = control if control is not None else NULL_CONTROL_TRACER

        self._ledger.scopes = {s.scope_id: s for s in snapshot.scopes}
        self._ledger.scopes.setdefault(
            self._ledger.root_scope.scope_id, self._ledger.root_scope
        )
        self._ledger.activations = {}
        # Per-scope and dynamic activation counts that number and budget each child.
        self._ledger.scope_population = Counter()
        self._ledger.scope_children = Counter()
        self._ledger.dynamic_activations = 0
        for activation in snapshot.activations:
            self._ledger.add_activation(activation)
        self._ledger.work_items = {w.work_item_id: w for w in snapshot.work_items}
        self._ledger.continuations = {c.work_item_id: c for c in snapshot.continuations}
        self._ledger.records = list(snapshot.records)
        self._accepted_inputs = list(snapshot.accepted_inputs)
        self._accepted_by_activation: dict[str, list[AcceptedInput]] = {}
        for accepted in self._accepted_inputs:
            self._accepted_by_activation.setdefault(accepted.activation_id, []).append(
                accepted
            )
        self._ledger.region_aggregates = list(snapshot.region_aggregates)
        self._ledger.aggregate_by_join = {}
        # A stored ledger may hold a nested level's aggregate after its root level's;
        # the root level's is the one delivered downstream.
        for aggregate in self._ledger.region_aggregates:
            join_op = aggregate.join_operator_id
            if join_op not in self._ledger.aggregate_by_join or not any(
                (act := self._ledger.activations.get(member.child_activation_id))
                is not None
                and not self._ledger.root_level(act.scope_id)
                for member in aggregate.members
            ):
                self._ledger.aggregate_by_join[join_op] = aggregate
        self._ledger.invocations = {i.invocation_id: i for i in snapshot.invocations}
        self._ledger.attempts = {a.attempt_id: a for a in snapshot.attempts}
        self._embodiment_selections = {
            sel.work_item_id: sel for sel in snapshot.embodiment_selections
        }
        self._input_resolutions = {
            res.work_item_id: res for res in snapshot.input_resolutions
        }
        self._input_preparations = {
            prep.work_item_id: prep for prep in snapshot.input_preparations
        }
        self._receipts = {r.invocation_id: r for r in snapshot.effect_receipts}
        self._decisions = list(snapshot.authority_decisions)
        self._grants = {g.grant_id: g for g in snapshot.delegated_grants}
        self._ledger.capabilities = {
            (c.scope_id, c.axis): c for c in snapshot.progress_capabilities
        }
        self._slots = {s.slot_key: s for s in snapshot.result_slots}
        self._publications = _rekeyed_publications(
            self._slots.values(), snapshot.result_publications
        )
        self._ledger.trace = list(snapshot.trace)

        # (agent activation, region operator) -> the synthetic opener activation that
        # owns that region's child-init scope, rebuilt from the persisted openers.
        self._ledger.region_openers = {
            (a.parent_activation_id, a.operator_id): a.activation_id
            for a in self._ledger.activations.values()
            if a.kind == "region" and a.parent_activation_id
        }
        # Mediated boundaries, keyed by (activation, adapter-local call correlation):
        # the correlation rule that maps a re-driven facade call to its recorded key.
        self._boundary_events = {
            (b.activation, b.call_correlation): b
            for b in snapshot.boundary_events
            if b.activation and b.call_correlation
        }
        self._ledger.wi_by_task = {
            w.legacy_task_id: w.work_item_id
            for w in self._ledger.work_items.values()
            if w.legacy_task_id
        }
        # The operator index resolves a static leaf's forward-record successor; a
        # dispatched child or iteration shares its body operator across instances, so it
        # is addressed by task or activation, never by operator.
        self._ledger.wi_by_operator = {
            w.operator_id: w.work_item_id
            for w in self._ledger.work_items.values()
            if w.legacy_task_id
            and not self._ledger.is_dynamic_activation(w.activation_id)
        }
        self._ledger.wi_by_activation = {
            w.activation_id: w.work_item_id for w in self._ledger.work_items.values()
        }
        self._slots_by_operator: dict[str, list[str]] = {}
        self._slots_by_output: dict[str, list[str]] = {}
        for slot in self._slots.values():
            self._slots_by_operator.setdefault(slot.source_operator_id, []).append(
                slot.slot_key
            )
            self._slots_by_output.setdefault(slot.output_id, []).append(slot.slot_key)

        # Scope ownership is keyed on the opener activation, so one operator can own a
        # scope per recursion level; an operator handle resolves through the index.
        self._ledger.scope_by_activation = {
            s.owner_activation_id: s.scope_id
            for s in self._ledger.scopes.values()
            if s.owner_activation_id
        }
        self._ledger.owner_acts_by_operator = {}
        for s in self._ledger.scopes.values():
            if s.owner_operator_id and s.owner_activation_id:
                self._ledger.owner_acts_by_operator.setdefault(
                    s.owner_operator_id, []
                ).append(s.owner_activation_id)
        self._ledger.loop_time = {}
        for scope in self._ledger.scopes.values():
            owner = scope.owner_operator_id
            if owner and self._topology.kind(owner) is OperatorKind.LOOP_CONTEXT:
                self._ledger.loop_time[scope.scope_id] = max(
                    (
                        a.loop_time
                        for a in self._ledger.activations.values()
                        if a.scope_id == scope.scope_id
                    ),
                    default=0,
                )
        # Released scopes are authoritative scope-level state, restored directly rather
        # than re-derived from records: a recursive region's levels share one join/loop
        # operator, so a record could not attribute a release to the right level.
        self._ledger.released_scopes = set(snapshot.released_scopes)
        # Control operators settled as a declared failure; a late record from another
        # input never fires one.
        self._failures.failed_regions = set(snapshot.failed_regions)
        # Child-init scopes a failed agent opened and that had not released: each
        # one's join never releases.
        self._failures.failed_scopes = set(snapshot.failed_scopes)
        # Why each task settled as a declared failure: its own reason, or the failure
        # it depends on. A ledger stored without them names each failed work item's own.
        self._failures.failure_reasons = dict(snapshot.failure_reasons)
        for wi in self._ledger.work_items.values():
            if wi.outcome is PublicationOutcome.DECLARED_FAILURE and wi.legacy_task_id:
                self._failures.failure_reasons.setdefault(
                    wi.legacy_task_id, wi.failure_reason or _DECLARED_FAILURE_REASON
                )
        # A spawn-site denial names no work item; an agent's denied boundary names one
        # and never refuses a later spawn.
        self._denied_spawns = {
            d.operator_id
            for d in self._decisions
            if d.kind is AuthorityDecisionKind.DENIED
            and d.operator_id
            and d.work_item_id is None
        }

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
        # The agent that declares each spawn region as one of its child regions.
        region_owner = {
            ref.spawn_ref: op.operator_id
            for op in template.operators
            if isinstance(op, AgentOperator)
            for ref in op.child_region_refs
        }
        # A region's entry body is materialized dynamically per spawn, never dispatched
        # eagerly. A region whose entry is its enclosing agent is explicit recursion:
        # that agent stays a normal dispatchable entry rather than a materialized body.
        child_body_refs = {
            op.child_template_ref
            for op in template.operators
            if isinstance(op, SpawnRegion)
            and op.child_template_ref
            and op.child_template_ref != op.operator_id
            and region_owner.get(op.operator_id) != op.child_template_ref
        }
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
            if edge.feedback or edge.to_op not in preds:
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
        for op in template.operators:
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
                if op.kind in _CONTROL_KINDS:
                    continuations.append(
                        Continuation(
                            work_item_id=_control_key(op.operator_id),
                            waiting_on=set(preds[op.operator_id]),
                        )
                    )
                continue
            effect, recovery = _effect_recovery(op)
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
        )
        engine = cls(snapshot, bundle, budget=budget, control=control, emitter=emitter)
        engine._initial = engine._open_roots()
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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        if wi.invocation_id is None:
            invocation = Invocation(
                invocation_id=new_invocation_id(),
                work_item_id=wi.work_item_id,
                state=InvocationState.ISSUED,
                replayable=is_replayable(wi.effect_class, wi.replay_contract),
                compensable=is_compensable(wi.effect_class, wi.replay_contract),
            )
            wi.invocation_id = invocation.invocation_id
            self._ledger.invocations[invocation.invocation_id] = invocation
        else:
            invocation = self._ledger.invocations[wi.invocation_id]
            invocation.state = next_on_reissue(invocation.state)
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
            attempt_no=len(wi.attempt_ids) + 1,
            worker_id=worker_id,
            started_at=now_iso(),
            alternative_id=(
                selection.alternative_id
                if (selection := self._embodiment_selections.get(wi.work_item_id))
                else None
            ),
        )
        wi.attempt_ids.append(attempt.attempt_id)
        self._ledger.attempts[attempt.attempt_id] = attempt
        wi.status = WorkItemStatus.DISPATCHED
        self._ledger.emit(
            "attempt_issued",
            work_item_id=wi.work_item_id,
            attempt_id=attempt.attempt_id,
            invocation_id=wi.invocation_id or "",
            operator_id=wi.operator_id,
        )

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_started(self, task_id: str) -> None:
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.invocation_id is None:
            return
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.RUNNING
        invocation = self._ledger.invocations[wi.invocation_id]
        invocation.state = next_on_acknowledge(invocation.state)
        self._ledger.emit(
            "invocation_acknowledged",
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )

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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        outcome = (
            PublicationOutcome.EXPLICIT_EMPTY if empty else PublicationOutcome.SUCCESS
        )
        value_ref = (
            ValueRef(kind="empty")
            if empty
            else ValueRef(
                kind="legacy_task_result",
                legacy_task_id=wi.legacy_task_id,
                content=content,
            )
        )
        self._settle_attempt_terminal(wi, outcome)
        activation = self._ledger.activations[wi.activation_id]
        # An agent's terminal completion settles every declared child region, so a
        # spawn_agent scope closes even without an explicit SpawnSeal.
        released = self._agent_terminal_regions(wi.operator_id, wi.activation_id)
        if activation.kind == "child":
            # A dispatched spawn child settles through its scope's child-init account,
            # never as a static forward record: the join closes on capability drain.
            return self._settle_child_wi(wi, activation, outcome, value_ref).extend(
                released
            )
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = outcome
        wi.value_ref = value_ref
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)
        self._ledger.private_state.release(wi.activation_id)
        self._publish(wi.operator_id, outcome, value_ref)
        return self._deliver_record(wi.operator_id, wi.activation_id, value_ref).extend(
            released
        )

    def _settle_attempt_terminal(
        self, wi: WorkItem, outcome: PublicationOutcome
    ) -> None:
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.finished_at = now_iso()
            self._ledger.emitter.emit_attempt(attempt)
        if wi.invocation_id is not None:
            invocation = self._ledger.invocations[wi.invocation_id]
            invocation.state = next_on_terminal(invocation.state)
            self._ledger.emitter.emit_boundary(invocation)
            self._record_receipt(wi, outcome)

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_failed(self, task_id: str, error: str, *, retryable: bool) -> Advance:
        """Retry a work item as a fresh attempt, or settle it and cascade failure."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        self._fail_open_attempt(wi, error)
        if retryable:
            wi.status = WorkItemStatus.READY
            self._ledger.emit(
                "attempt_retry",
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
            )
            return Advance(retry=[wi.legacy_task_id])
        wi.failure_reason = error
        return self._settle_failed_wi(wi)

    def _fail_open_attempt(self, wi: WorkItem, error: str) -> None:
        """Close the work item's attempt as failed while it is still in flight; one
        already closed keeps its outcome."""
        attempt = self._ledger.latest_attempt(wi)
        if attempt is None or attempt.status not in _OPEN_ATTEMPT_STATUSES:
            return
        attempt.status = AttemptStatus.FAILED
        attempt.finished_at = now_iso()
        attempt.error = error
        self._ledger.emitter.emit_attempt(attempt)

    def _settle_failed_wi(self, wi: WorkItem) -> Advance:
        """Settle a work item as a declared failure: a child drains its scope, anything
        else cascades over its static successors. A failed agent fails the regions it
        declares either way."""
        activation = self._ledger.activations[wi.activation_id]
        if activation.kind != "child":
            return self._settle_failure(wi.work_item_id)
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        # A child's terminal failure drains its scope account and lets an all-succeed
        # join fail; it does not cascade over static successors.
        advance = self._settle_child_wi(
            wi, activation, PublicationOutcome.DECLARED_FAILURE, None
        )
        cascade = Advance(failed=[wi.legacy_task_id])
        self._fail_agent_regions(wi, cascade, set())
        self._declare_failures(wi, cascade.failed)
        advance.failed[:0] = cascade.failed
        advance.cancelled.extend(cascade.cancelled)
        return advance

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_returned(self, task_id: str) -> bool:
        """Close an in-flight attempt handed back without an outcome, and re-ready it;
        returns whether there was one.

        The attempt is not charged: the work item runs again under its invocation.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.DISPATCHED:
            return False
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.RETURNED
            attempt.finished_at = now_iso()
            self._ledger.emitter.emit_attempt(attempt)
        wi.status = WorkItemStatus.READY
        self._ledger.emit(
            "attempt_returned", work_item_id=wi.work_item_id, operator_id=wi.operator_id
        )
        return True

    @_ds_drive(ControlPlaneWindow.POST_START)
    def on_uncertain(self, task_id: str, error: str | None = None) -> Advance:
        """Resolve a lost acknowledgement, route loss, or failure that may follow the
        work item's external effect.

        ``error`` is the executor's message for a reported failure; the attempt keeps
        it, and a work item that cannot run again fails with it beside the reason.
        """
        wi = self._ledger.work_item_for_task(task_id)
        resolution = self._resolve_loss(wi)
        if wi is None or resolution is _LossResolution.NOTHING:
            return Advance()
        if resolution is _LossResolution.PREPARE_AGAIN:
            # An input preparation commits to no invocation and reserves nothing, so
            # the task resolves its inputs again on another worker.
            self._ledger.emit(
                "input_preparation_lost",
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
            )
            return Advance(retry=[wi.legacy_task_id])
        assert wi.invocation_id is not None
        if resolution is _LossResolution.BOUNDARY_FAILS:
            # The worker that captured this boundary's request is lost, and the
            # worker-private request cannot be recovered here (a fresh permit would need
            # a fresh proposal on a new worker). Fail the boundary clean so the workflow
            # errors rather than resuming the agent past a boundary with no outcome.
            self._ledger.emit(
                "invocation_ambiguity_terminal",
                work_item_id=wi.work_item_id,
                invocation_id=wi.invocation_id,
            )
            self._ledger.emitter.emit_boundary(
                self._ledger.invocations[wi.invocation_id]
            )
            wi.failure_reason = _ambiguity_terminal_reason(error)
            return self._settle_failed_wi(wi)
        invocation = self._ledger.invocations[wi.invocation_id]
        invocation.state = next_on_uncertain(
            invocation.state,
            replayable=invocation.replayable,
            compensable=invocation.compensable,
        )
        self._ledger.emitter.emit_boundary(invocation)
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.LOST
            attempt.finished_at = now_iso()
            if error is not None:
                attempt.error = error
            self._ledger.emitter.emit_attempt(attempt)
        if resolution is _LossResolution.RUN_AGAIN:
            wi.status = WorkItemStatus.READY
            self._ledger.emit(
                "invocation_uncertain_retry",
                work_item_id=wi.work_item_id,
                invocation_id=wi.invocation_id,
                error=error,
            )
            return Advance(retry=[wi.legacy_task_id])
        self._ledger.emit(
            (
                "invocation_compensation_required"
                if invocation.compensable
                else "invocation_ambiguity_terminal"
            ),
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
            error=error,
        )
        wi.failure_reason = _ambiguity_terminal_reason(error)
        return self._settle_failed_wi(wi)

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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        # A recorded call is a re-drive whether it was granted or denied: it maps to its
        # key and creates no new work, so re-validation and duplicate records are cut.
        if self._is_boundary_redrive(wi, event):
            return Advance()
        op = self._topology.operators.get(wi.operator_id)
        # An agent boundary is authority-checked against the operator's declared face; a
        # leaf boundary carries no authority ceiling and defers its invocation directly.
        if isinstance(op, AgentOperator):
            if (denial := self._validate_agent_boundary(op, wi, event)) is not None:
                return self._record_boundary_denial(wi, event, denial)
            if event.kind in _DEDUP_CAPABLE and event.call_correlation is None:
                raise RegionError(
                    f"mediated {event.kind.value} boundary requires a call correlation "
                    "for durable dedup"
                )
        match event.kind:
            case BoundaryEventKind.SPAWN:
                opener = self._agent_spawn_opener(op, wi, event)
                # Record only after the child materializes: a budget/depth rejection
                # raises and must leave no phantom-accepted envelope to re-drive. The
                # region selects the entry body; a raw child_ref never names a body. The
                # spawn payload becomes the child's inline child-init input.
                child_input = event.value_ref or (
                    ValueRef(kind="inline", literal=event.request_payload)
                    if event.request_payload is not None
                    else None
                )
                advance = self.materialize_child(opener, value_ref=child_input)
                self._record_boundary(wi, event)
                return advance
            case BoundaryEventKind.SPAWN_SEAL:
                opener = self._agent_spawn_opener(op, wi, event)
                advance = self.seal_spawn(opener)
                self._record_boundary(wi, event)
                return advance
            case BoundaryEventKind.INVOCATION:
                return self._suspend_on_request(wi, event, effect=False)
            case BoundaryEventKind.EXTERNAL_EFFECT:
                return self._suspend_on_request(wi, event, effect=True)
            case BoundaryEventKind.YIELD:
                wi.continuation_ref = event.continuation
                self._record_boundary(wi, event)
                self._suspend_work_item(wi, "episode_yielded")
                return Advance()
            case BoundaryEventKind.STATE_ACCESS:
                self._ledger.emit(
                    "state_access",
                    work_item_id=wi.work_item_id,
                    operator_id=wi.operator_id,
                    detail={"state_ref": event.state_ref or ""},
                )
                return Advance()
        return Advance()

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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        op = self._topology.operators.get(wi.operator_id)
        advance = Advance()
        per_region: dict[str, int] = {}
        for member in group.members:
            if (wi.activation_id, member.call_correlation) in self._boundary_events:
                continue  # a re-driven group reuses its recorded members
            event = self._facade_member_event(group.group_id, member)
            if member.kind is BoundaryEventKind.SPAWN:
                self._route_group_spawn(op, wi, event, per_region, advance)
            else:
                self._route_group_invocation(op, wi, event)
        members = self._group_members(wi.activation_id, group.group_id)
        if self._group_awaits_unresolved(members):
            self._suspend_work_item(wi, "episode_group_suspended")
            return advance
        # No await-outcome member is pending: the group settled at admission. Stage its
        # ordered acceptance vector for the next step; the runtime re-enqueues the lane
        # at once (closing this turn's attempt) rather than holding a worker.
        wi.pending_outcome_group = group.group_id
        wi.pending_outcome_call = None
        return advance

    def _facade_member_event(
        self, group_id: str, member: FacadeCallMember
    ) -> BoundaryEvent:
        is_spawn = member.kind is BoundaryEventKind.SPAWN
        return BoundaryEvent(
            kind=member.kind,
            interface=None if is_spawn else member.interface_or_region,
            child_region_ref=member.interface_or_region if is_spawn else None,
            call_correlation=member.call_correlation,
            request_payload=member.request_payload,
            request_digest=member.request_digest,
            injection_target=member.harness_call_id,
            injection_tool=member.tool_name,
            group_id=group_id,
            group_ordinal=member.ordinal,
            completion_mode=member.completion_mode.value,
        )

    def _route_group_invocation(
        self, op: LogicalOperator | None, wi: WorkItem, event: BoundaryEvent
    ) -> None:
        # Only an agent operator originates a facade; a non-agent operator is a fabric
        # misconfiguration, denied fail-closed rather than minting an unvalidated call.
        denial = (
            self._validate_agent_boundary(op, wi, event)
            if isinstance(op, AgentOperator)
            else DenialKind.AUTHORITY
        )
        if denial is not None:
            self._record_boundary(wi, event, denial=denial)
            return
        invocation = Invocation(
            invocation_id=new_invocation_id(),
            work_item_id=wi.work_item_id,
            state=InvocationState.ISSUED,
            replayable=True,
        )
        self._ledger.invocations[invocation.invocation_id] = invocation
        self._record_boundary(wi, event, invocation_id=invocation.invocation_id)

    def _route_group_spawn(
        self,
        op: LogicalOperator | None,
        wi: WorkItem,
        event: BoundaryEvent,
        per_region: dict[str, int],
        advance: Advance,
    ) -> None:
        if not isinstance(op, AgentOperator):
            self._record_boundary(wi, event, denial=DenialKind.AUTHORITY)
            return
        if (denial := self._validate_agent_boundary(op, wi, event)) is not None:
            self._record_boundary(wi, event, denial=denial)
            return
        region = event.child_region_ref or ""
        # This turn's admitted spawns already register in the child-init scope, so the
        # region count reflects them; per_region only sums the per-turn budget across
        # regions and must not be re-added here (that would trip the region cap at 2x).
        turn_total = sum(per_region.values())
        region_total = self._region_child_count(wi.activation_id, region)
        if (
            turn_total >= self._budget.max_spawns_per_turn
            or region_total >= self._budget.max_spawns_per_region
        ):
            # A budget overflow is a typed quota outcome, never a denial and never a
            # sibling-poisoning failure: the member acks quota and creates no child.
            self._record_boundary(wi, event)
            self._set_member_outcome(
                wi,
                event.call_correlation,
                f"{ToolOutcomeStatus.QUOTA.value}: the {region!r} spawn budget was "
                "reached this turn; no child was created",
            )
            return
        child_input = (
            ValueRef(kind="inline", literal=event.request_payload)
            if event.request_payload is not None
            else None
        )
        try:
            opener = self._agent_spawn_opener(op, wi, event)
            child_advance = self.materialize_child(opener, value_ref=child_input)
        except RegionError as exc:
            self._record_boundary(wi, event)
            self._set_member_outcome(
                wi,
                event.call_correlation,
                f"{ToolOutcomeStatus.UNAVAILABLE.value}: the {region!r} region cannot "
                f"accept a child ({exc})",
            )
            return
        advance.extend(child_advance)
        self._record_boundary(wi, event)
        self._set_member_outcome(
            wi,
            event.call_correlation,
            f"{ToolOutcomeStatus.SUCCESS.value}: spawned a {region!r} reviewer child",
        )
        per_region[region] = per_region.get(region, 0) + 1

    def _set_member_outcome(
        self, wi: WorkItem, call_correlation: str | None, value: str
    ) -> None:
        """Settle a group member's outcome in place without re-readying the lane."""
        if call_correlation is None:
            return
        corr = (wi.activation_id, call_correlation)
        if (env := self._boundary_events.get(corr)) is not None:
            self._boundary_events[corr] = env.model_copy(
                update={"outcome_value": value}
            )

    def _region_child_count(self, agent_activation: str, region: str) -> int:
        """The children already materialized under an agent's named spawn region."""
        act = self._ledger.activations.get(agent_activation)
        op = self._topology.operators.get(act.operator_id) if act else None
        if not isinstance(op, AgentOperator):
            return 0
        region_op = self._topology.agent_region_op(op, region)
        if region_op is None:
            return 0
        opener = self._ledger.region_openers.get((agent_activation, region_op))
        if opener is None or (scope_id := self._ledger.scope_id_for(opener)) is None:
            return 0
        return self._ledger.scope_children[scope_id]

    def _group_members(self, activation_id: str, group_id: str) -> list[BoundaryEvent]:
        members = [
            env
            for (act, _), env in self._boundary_events.items()
            if act == activation_id and env.group_id == group_id
        ]
        members.sort(key=lambda e: e.group_ordinal or 0)
        return members

    @staticmethod
    def _boundary_resolved(env: BoundaryEvent) -> bool:
        """Whether a boundary is settled: an inline value, a reference, or a denial."""
        return (
            env.outcome_value is not None
            or env.outcome_ref is not None
            or env.denial is not None
        )

    def retries_on_loss(self, task_id: str) -> bool:
        """Whether the loss of the task's worker runs its work item again, as
        ``on_uncertain`` resolves it, rather than failing it."""
        return (
            self._resolve_loss(self._ledger.work_item_for_task(task_id))
            in _RERUN_ON_LOSS
        )

    def _resolve_loss(self, wi: WorkItem | None) -> _LossResolution:
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return _LossResolution.NOTHING
        if wi.invocation_id is None:
            if wi.work_item_id in self._input_preparations:
                return _LossResolution.PREPARE_AGAIN
            return _LossResolution.NOTHING
        if wi.status is WorkItemStatus.BLOCKED and self._has_pending_local_boundary(wi):
            return _LossResolution.BOUNDARY_FAILS
        if self._ledger.invocations[wi.invocation_id].replayable:
            return _LossResolution.RUN_AGAIN
        return _LossResolution.FAILS

    def awaits_worker_held_boundary(self, task_id: str) -> bool:
        """Whether the task is suspended on an unsettled boundary whose raw request
        only its capturing worker holds."""
        wi = self._ledger.work_item_for_task(task_id)
        return (
            wi is not None
            and wi.status is WorkItemStatus.BLOCKED
            and self._has_pending_local_boundary(wi)
        )

    def suspending_worker(self, task_id: str) -> str | None:
        """The worker whose step suspended the task on a mediated boundary that awaits
        its outcome, or None when the task is not suspended on one."""
        wi = self._ledger.work_item_for_task(task_id)
        if (
            wi is None
            or wi.status is not WorkItemStatus.BLOCKED
            or not self._awaits_mediated_outcome(wi)
            or (attempt := self._ledger.latest_attempt(wi)) is None
        ):
            return None
        return attempt.worker_id

    def _has_pending_local_boundary(self, wi: WorkItem) -> bool:
        """Whether the work item awaits an unsettled worker-originated boundary.

        A recorded request digest marks a boundary whose raw request lives only on the
        capturing worker; if that worker is lost the boundary cannot be recovered here.
        """
        return any(
            act == wi.activation_id
            and env.request_digest is not None
            and not self._boundary_resolved(env)
            for (act, _), env in self._boundary_events.items()
        )

    @classmethod
    def _group_awaits_unresolved(cls, members: Sequence[BoundaryEvent]) -> bool:
        """Whether the group holds an await-outcome member with no settled outcome."""
        return any(
            e.completion_mode == FacadeCompletionMode.AWAIT_OUTCOME.value
            and not cls._boundary_resolved(e)
            for e in members
        )

    def has_open_facade_group(self, task_id: str) -> bool:
        """Whether a recorded facade group for this episode still holds the resume gate.

        A group is open only while an await-outcome member is unsettled; a spawn-only
        group closes at admission, so the fence lets the next turn issue another group.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return False
        return self._group_awaits_unresolved(
            [
                env
                for (act, _), env in self._boundary_events.items()
                if act == wi.activation_id and env.group_id is not None
            ]
        )

    def group_dispatch_envelopes(
        self, task_id: str, group_id: str
    ) -> list[ToolInvocationEnvelope]:
        """The dispatch envelopes for a group's still-unresolved invocation members."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return []
        out: list[ToolInvocationEnvelope] = []
        for env in self._group_members(wi.activation_id, group_id):
            if env.kind is not BoundaryEventKind.INVOCATION:
                continue
            if self._boundary_resolved(env):
                continue
            if (envelope := self._envelope_from(wi, env)) is not None:
                out.append(envelope)
        return out

    def deliver_boundary_outcome(self, task_id: str, call_correlation: str) -> Advance:
        """Re-ready a boundary-suspended work item once its outcome is durable.

        A mediated request's durable outcome — a model or tool result, or a denial —
        lets the episode resume: the work item returns to READY for a fresh attempt that
        injects the outcome at its originating call. Only a boundary-suspended work item
        with a recorded envelope for the call is resumed — a delivery to a running or
        settled item, or for an unrecorded call, is a no-op. An episode has one
        outstanding boundary at a time, so the recorded call is the one it awaits.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.BLOCKED:
            return Advance()
        if (wi.activation_id, call_correlation) not in self._boundary_events:
            return Advance()
        wi.status = WorkItemStatus.READY
        self._ledger.emit(
            "episode_resumed",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={"call": call_correlation},
        )
        return Advance(ready=[wi.legacy_task_id])

    def mark_pending_outcome(self, task_id: str, call_correlation: str | None) -> None:
        """Record (or clear) the settled boundary whose outcome the next resume injects.

        Cleared as each step is processed, so a step never re-injects a prior step's
        already-consumed outcome.
        """
        if (wi := self._ledger.work_item_for_task(task_id)) is not None:
            wi.pending_outcome_call = call_correlation
            if call_correlation is None:
                wi.pending_outcome_group = None

    def close_latest_attempt(self, task_id: str) -> None:
        """Settle a still-running attempt of a continuing episode, bounding its history.

        A continue-boundary re-dispatches without suspending, so its finished attempt is
        marked succeeded here rather than left perpetually running.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is not None and (attempt := self._ledger.latest_attempt(wi)) is not None:
            if attempt.status in (AttemptStatus.ISSUED, AttemptStatus.RUNNING):
                attempt.status = AttemptStatus.SUCCEEDED
                attempt.finished_at = now_iso()
                self._ledger.emitter.emit_attempt(attempt)

    def _awaits_mediated_outcome(self, wi: WorkItem) -> bool:
        """Whether the work item's activation awaits a mediated boundary with no
        outcome."""
        return any(
            act == wi.activation_id
            and env.kind in _MEDIATED_BOUNDARY_KINDS
            and not self._boundary_resolved(env)
            for (act, _), env in self._boundary_events.items()
        )

    def pending_tool_dispatches(self) -> list[ToolInvocationEnvelope]:
        """Mediated boundaries suspended with no durable outcome, for a restart.

        A model or tool boundary suspends off-lane while its handler settles it; a crash
        before that settle leaves the work item blocked with an issued invocation and no
        recorded outcome. Returns each as a full dispatch envelope so the runtime routes
        it back to its handler by (kind, interface) — a search to the broker, a model to
        the gateway — never misrouting on the recovered kind.
        """
        pending: list[ToolInvocationEnvelope] = []
        for (activation, corr), env in self._boundary_events.items():
            if env.kind not in _MEDIATED_BOUNDARY_KINDS:
                continue
            if self._boundary_resolved(env):
                continue
            wi = self._ledger.wi_by_activation.get(activation)
            work_item = self._ledger.work_items.get(wi) if wi else None
            if work_item is None or work_item.status is not WorkItemStatus.BLOCKED:
                continue
            if (envelope := self._envelope_from(work_item, env)) is not None:
                pending.append(envelope)
        return pending

    def tool_dispatch_envelope(
        self, task_id: str, call_correlation: str
    ) -> ToolInvocationEnvelope | None:
        """The dispatch envelope for a recorded mediated boundary, or None if absent."""
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        env = self._boundary_events.get((wi.activation_id, call_correlation))
        if env is None:
            return None
        return self._envelope_from(wi, env)

    def pending_tool_dispatch(
        self, task_id: str, call_correlation: str
    ) -> ToolInvocationEnvelope | None:
        """The dispatch envelope for a still-pending mediated boundary, or None.

        Returns an envelope only while the boundary is suspended with no durable outcome
        and its work item is blocked, so a held re-drive re-issues exactly the off-lane
        dispatch a restart would; a settled, terminalized, or cancelled boundary yields
        nothing.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.BLOCKED:
            return None
        env = self._boundary_events.get((wi.activation_id, call_correlation))
        if env is None or self._boundary_resolved(env):
            return None
        return self._envelope_from(wi, env)

    def _envelope_from(
        self, wi: WorkItem, env: BoundaryEvent
    ) -> ToolInvocationEnvelope | None:
        if env.invocation_id is None or env.call_correlation is None:
            return None
        return ToolInvocationEnvelope(
            kind=env.kind,
            interface=env.interface or MODEL_INTERFACE,
            invocation_id=env.invocation_id,
            task_id=wi.legacy_task_id,
            activation_id=wi.activation_id,
            call_correlation=env.call_correlation,
            idempotency_key=env.idempotency_key,
            request_payload=env.request_payload,
            request_digest=env.request_digest,
            grant_snapshot=self._grant_snapshot_for(wi),
        )

    def _grant_snapshot_for(self, wi: WorkItem) -> GrantSnapshot:
        grant_id = self._ledger.workflow_instance.root_grant_id
        if (act := self._ledger.activations.get(wi.activation_id)) is not None:
            if (scope := self._ledger.scopes.get(act.scope_id)) is not None:
                grant_id = scope.grant_id or grant_id
        return GrantSnapshot(
            grant_id=grant_id,
            policy_envelope=self._ledger.workflow_instance.policy_envelope,
        )

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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        env = self._boundary_events.get((wi.activation_id, call_correlation))
        if env is None or env.invocation_id is None or env.request_digest is None:
            return None
        interface = env.interface or ""
        epoch = 0
        if (act := self._ledger.activations.get(wi.activation_id)) is not None:
            epoch = self._grant_for_scope(act.scope_id).epoch
        return MediatedOperationPermit(
            permit_id=new_mediated_permit_id(),
            agent_task_id=wi.legacy_task_id,
            call_correlation=call_correlation,
            interface=interface,
            subject=interface,
            invocation_id=env.invocation_id,
            idempotency_key=env.idempotency_key,
            request_digest=env.request_digest,
            target_id=target_id,
            target_generation=target_generation,
            policy_epoch=epoch,
            deadline_epoch=deadline_epoch,
            max_results=max_results,
            timeout_sec=timeout_sec,
            result_char_cap=result_char_cap,
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
        wi = self._ledger.work_item_for_task(task_id)
        act = self._ledger.activations.get(wi.activation_id) if wi is not None else None
        op = self._topology.operators.get(act.operator_id) if act is not None else None
        if wi is None or act is None or not isinstance(op, AgentOperator):
            return None
        invoke, _ = self._agent_faces(op, wi)
        if MODEL_INTERFACE not in invoke:
            return None
        return MediatedOperationPermit(
            permit_id=new_mediated_permit_id(),
            agent_task_id=wi.legacy_task_id,
            call_correlation=call_correlation,
            interface=MODEL_INTERFACE,
            subject=MODEL_INTERFACE,
            invocation_id=new_invocation_id(),
            idempotency_key=new_idempotency_key(),
            request_digest=request_digest,
            target_id=target_id,
            target_generation=target_generation,
            policy_epoch=self._grant_for_scope(act.scope_id).epoch,
            deadline_epoch=deadline_epoch,
            max_results=1,
            timeout_sec=timeout_sec,
            result_char_cap=result_char_cap,
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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.BLOCKED:
            return False
        env = self._boundary_events.get((wi.activation_id, call_correlation))
        return env is not None and not self._boundary_resolved(env)

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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return Advance()
        corr = (wi.activation_id, call_correlation)
        env = self._boundary_events.get(corr)
        resolved = wi.status is not WorkItemStatus.BLOCKED or (
            env is not None and self._boundary_resolved(env)
        )
        if resolved:
            # A duplicate/late settle of an already-resolved member, or any settle for a
            # no-longer-BLOCKED (cancelled or terminal) work item, is an idempotent
            # no-op: it never re-runs the deliver path, so it cannot re-ready or stamp a
            # stale outcome on a resolved or cancelled boundary.
            return Advance()
        if env is not None and (value is not None or ref is not None):
            env = env.model_copy(update={"outcome_value": value, "outcome_ref": ref})
            self._boundary_events[corr] = env
        if env is not None and env.group_id is not None:
            # A group member settled: hold the resume until every await-outcome member
            # is resolved, then re-ready exactly once with the full ordered vector.
            members = self._group_members(wi.activation_id, env.group_id)
            if self._group_awaits_unresolved(members):
                return Advance()
            wi.pending_outcome_group = env.group_id
            wi.pending_outcome_call = None
            return self.deliver_boundary_outcome(task_id, call_correlation)
        self.mark_pending_outcome(task_id, call_correlation)
        return self.deliver_boundary_outcome(task_id, call_correlation)

    def terminalize_boundary_invocation(
        self, task_id: str, call_correlation: str
    ) -> str | None:
        """Record a settled mediated boundary's invocation as terminal in the ledger.

        Returns the durable ``invocation_id`` so a control-plane consumer bound to it —
        the resident-capacity admission credit — can advance from this fenced ``DS``
        outcome. The transition is idempotent and never regresses an ambiguity-terminal
        outcome.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        env = self._boundary_events.get((wi.activation_id, call_correlation))
        if env is None or env.invocation_id is None:
            return None
        invocation = self._ledger.invocations.get(env.invocation_id)
        if invocation is not None:
            invocation.state = next_on_terminal(invocation.state)
            self._ledger.emitter.emit_boundary(invocation)
        return env.invocation_id

    def boundary_invocation_completed(self, invocation_id: str) -> bool | None:
        """Whether a terminal boundary invocation completed with an outcome; None while
        it is unknown or not terminal."""
        invocation = self._ledger.invocations.get(invocation_id)
        if invocation is None or invocation.state not in TERMINAL_INVOCATION_STATES:
            return None
        return any(
            env.invocation_id == invocation_id
            and (env.outcome_value is not None or env.outcome_ref is not None)
            for env in self._boundary_events.values()
        )

    def terminalize_unsettled_invocations(
        self, task_ids: Iterable[str] | None = None
    ) -> list[str]:
        """Terminalize the unsettled mediated boundary invocations of the given tasks'
        activations, or of every activation; one already terminal is left as it is.

        Returns the ``invocation_id``s it terminalized so a control-plane consumer bound
        to them — a resident-capacity admission credit — releases from this fenced
        terminal.
        """
        activations = (
            None
            if task_ids is None
            else {
                wi.activation_id
                for task_id in task_ids
                if (wi := self._ledger.work_item_for_task(task_id)) is not None
            }
        )
        ids: list[str] = []
        for activation, invocation_id in self._unsettled_invocation_boundaries():
            if activations is not None and activation not in activations:
                continue
            invocation = self._ledger.invocations.get(invocation_id)
            if invocation is not None:
                if invocation.state in TERMINAL_INVOCATION_STATES:
                    continue
                invocation.state = next_on_terminal(invocation.state)
                self._ledger.emitter.emit_boundary(invocation)
            ids.append(invocation_id)
        return ids

    def suspended_boundary_tasks(self) -> list[str]:
        """Tasks suspended at an unsettled mediated boundary.

        Such a task's worker released the lane, so it holds no dispatch and returns no
        terminal; a task mid-step is not among them.
        """
        tasks: list[str] = []
        for activation, _ in self._unsettled_invocation_boundaries():
            wi_id = self._ledger.wi_by_activation.get(activation)
            if (wi := self._ledger.work_items.get(wi_id) if wi_id else None) is None:
                continue
            if wi.legacy_task_id not in tasks:
                tasks.append(wi.legacy_task_id)
        return tasks

    def _unsettled_invocation_boundaries(self) -> list[tuple[str, str]]:
        """Each unsettled mediated-boundary invocation id and its owning activation."""
        return [
            (activation, env.invocation_id)
            for (activation, _), env in self._boundary_events.items()
            if env.invocation_id is not None and not self._boundary_resolved(env)
        ]

    def episode_context(
        self, task_id: str
    ) -> tuple[str | None, tuple[DeliveredOutcome, ...]]:
        """The durable capsule and pending injected outcome for an agent's next step.

        Rebuilt from the ledger, never in-memory episode state: the capsule is the work
        item's continuation, and the one pending outcome is reconstructed from its
        settled boundary envelope, so a re-dispatch after a restart carries it again.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None, ()
        outcomes: tuple[DeliveredOutcome, ...] = ()
        if wi.pending_outcome_group is not None:
            members = self._group_members(wi.activation_id, wi.pending_outcome_group)
            outcomes = tuple(self._delivered_outcome(env) for env in members)
        elif wi.pending_outcome_call is not None:
            env = self._boundary_events.get((wi.activation_id, wi.pending_outcome_call))
            if env is not None:
                outcomes = (self._delivered_outcome(env),)
        return wi.continuation_ref, outcomes

    @staticmethod
    def _delivered_outcome(env: BoundaryEvent) -> DeliveredOutcome:
        corr = env.call_correlation or ""
        if env.denial is not None:
            return DeliveredOutcome(
                call_correlation=corr,
                idempotency_key=env.idempotency_key,
                kind=OutcomeKind.DENIED,
                denial=env.denial,
                injection_target=env.injection_target,
                injection_tool=env.injection_tool,
                injection_arguments=env.request_payload,
            )
        return DeliveredOutcome(
            call_correlation=corr,
            idempotency_key=env.idempotency_key,
            kind=OutcomeKind.RESULT,
            value=env.outcome_value,
            outcome_ref=env.outcome_ref,
            injection_target=env.injection_target,
            injection_tool=env.injection_tool,
        )

    def _is_boundary_redrive(self, wi: WorkItem, event: BoundaryEvent) -> bool:
        """Whether a boundary reissues a recorded facade call under its stable id."""
        if event.call_correlation is None:
            return False
        if (wi.activation_id, event.call_correlation) not in self._boundary_events:
            return False
        self._ledger.emit(
            "boundary_redriven",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={"call": event.call_correlation},
        )
        return True

    def _validate_agent_boundary(
        self, op: AgentOperator, wi: WorkItem, event: BoundaryEvent
    ) -> DenialKind | None:
        """Check an agent boundary against its signature and effective authority.

        A kind outside the declared signature, a tool/model interface outside the
        effective invoke face, or a spawn/seal that names no declared child region — a
        raw operator id or an undeclared role — is a definitive denial: authority when
        the request is undeclared, policy when the pinned envelope blocks a declared
        one. None means admissible.
        """
        if event.kind not in op.boundary.events:
            return DenialKind.AUTHORITY
        if event.kind in (
            BoundaryEventKind.INVOCATION,
            BoundaryEventKind.EXTERNAL_EFFECT,
        ):
            invoke, _ = self._agent_faces(op, wi)
            if event.interface is not None and event.interface not in invoke:
                return (
                    DenialKind.POLICY
                    if event.interface in op.authority.invoke
                    else DenialKind.AUTHORITY
                )
        elif event.kind in (BoundaryEventKind.SPAWN, BoundaryEventKind.SPAWN_SEAL):
            if self._topology.agent_region_op(op, event.child_region_ref) is None:
                return DenialKind.AUTHORITY
        return None

    def _agent_spawn_opener(
        self, op: LogicalOperator | None, wi: WorkItem, event: BoundaryEvent
    ) -> str:
        """The opener activation for the agent's selected region, minted on first use.

        Validation admitted the role, so it resolves to a declared spawn region; the
        opener owns that region's child-init scope keyed by (agent activation, region).
        """
        if not isinstance(op, AgentOperator):
            raise RegionError(
                f"{wi.operator_id!r} is not an agent; it cannot yield a spawn boundary"
            )
        region_op = self._topology.agent_region_op(op, event.child_region_ref)
        assert region_op is not None  # admitted by _validate_agent_boundary
        return self._region_opener(wi.activation_id, region_op)

    def _region_opener(self, agent_activation: str, region_op: str) -> str:
        """Mint or reuse the (agent activation, region) opener that owns its scope.

        Reuses the recursion opener machinery: a synthetic ``region`` activation of the
        spawn region owns a child-init scope nested under the agent's own scope, so the
        scope, its delegated grant, its progress, and its matched join resolve through
        the existing region path. The grant attenuates from the agent's delegate face
        and the region's per-site ceiling, never the agent's blanket ceiling.
        """
        key = (agent_activation, region_op)
        if (opener := self._ledger.region_openers.get(key)) is not None:
            return opener
        agent = self._ledger.activations[agent_activation]
        agent_op = self._topology.operators[agent.operator_id]
        assert isinstance(agent_op, AgentOperator)
        _, agent_delegate = self._agent_face_tuples(agent_op, agent.scope_id)
        opener_act = Activation(
            activation_id=new_activation_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id=agent.scope_id,
            operator_id=region_op,
            kind="region",
            parent_activation_id=agent_activation,
        )
        self._ledger.add_activation(opener_act)
        self._ledger.region_openers[key] = opener_act.activation_id
        self._open_child_init_scope(
            opener_act.activation_id,
            parent_scope_id=agent.scope_id,
            parent_delegate=agent_delegate,
        )
        return opener_act.activation_id

    def _agent_faces(
        self, op: AgentOperator, wi: WorkItem
    ) -> tuple[frozenset[str], frozenset[str]]:
        """The agent's effective invoke/delegate faces: its ceiling under policy."""
        invoke, delegate = self._agent_face_tuples(
            op, self._ledger.activations[wi.activation_id].scope_id
        )
        return frozenset(invoke), frozenset(delegate)

    def _agent_face_tuples(
        self, op: AgentOperator, agent_scope_id: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The agent's invoke/delegate faces: the scope grant under ceiling+policy."""
        scope_grant = self._grant_for_scope(agent_scope_id)
        envelope = self._policy_interfaces()
        invoke = attenuate(scope_grant.invoke, op.authority.invoke, envelope)
        delegate = attenuate(invoke, op.authority.delegate, envelope)
        return invoke, delegate

    def _record_boundary(
        self,
        wi: WorkItem,
        event: BoundaryEvent,
        *,
        invocation_id: str | None = None,
        denial: DenialKind | None = None,
    ) -> str | None:
        """Persist the durable correlation envelope; return its idempotency key.

        Keyed on (activation, call correlation): a re-driven facade call reuses the
        recorded key rather than minting a new one, so a fresh harness call id never
        duplicates a dedupe-capable target effect. A boundary without a call correlation
        carries no durable dedupe handle and records nothing.
        """
        if event.call_correlation is None:
            return None
        corr = (wi.activation_id, event.call_correlation)
        existing = self._boundary_events.get(corr)
        key = existing.idempotency_key if existing else new_idempotency_key()
        self._boundary_events[corr] = event.model_copy(
            update={
                "activation": wi.activation_id,
                "idempotency_key": key,
                "invocation_id": invocation_id,
                "denial": denial,
            }
        )
        detail = {"idempotency_key": key or "", "call": event.call_correlation}
        if denial is not None:
            detail["denial"] = denial.value
        self._ledger.emit(
            "boundary_recorded",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            invocation_id=invocation_id,
            detail=detail,
        )
        return key

    def _record_boundary_denial(
        self, wi: WorkItem, event: BoundaryEvent, denial: DenialKind
    ) -> Advance:
        """Settle a denied agent boundary as a durable typed continuation outcome.

        The denial is recorded against the operator and injected back as the boundary's
        outcome; it creates neither an invocation nor a child, and the episode suspends
        to receive it rather than silently proceeding.
        """
        subject = event.interface or event.child_region_ref or event.child_ref or ""
        self._decisions.append(
            AuthorityDecision(
                grant_id=self._ledger.root_grant.grant_id,
                interface=subject,
                kind=AuthorityDecisionKind.DENIED,
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
                denial_kind=denial,
                reason=f"boundary {event.kind.value} denied",
            )
        )
        self._record_boundary(wi, event, denial=denial)
        self._ledger.emit(
            "authority_denied" if denial is DenialKind.AUTHORITY else "policy_denied",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
            detail={"interface": subject},
        )
        self._suspend_work_item(wi, "episode_suspended")
        return Advance()

    def _suspend_on_request(
        self, wi: WorkItem, event: BoundaryEvent, *, effect: bool
    ) -> Advance:
        invocation = Invocation(
            invocation_id=new_invocation_id(),
            work_item_id=wi.work_item_id,
            state=InvocationState.ISSUED,
            replayable=(
                is_replayable(wi.effect_class, wi.replay_contract) if effect else True
            ),
            compensable=(
                is_compensable(wi.effect_class, wi.replay_contract) if effect else False
            ),
        )
        self._ledger.invocations[invocation.invocation_id] = invocation
        self._record_boundary(wi, event, invocation_id=invocation.invocation_id)
        self._ledger.emit(
            "effect_requested" if effect else "invocation_issued",
            work_item_id=wi.work_item_id,
            invocation_id=invocation.invocation_id,
            operator_id=wi.operator_id,
            detail={"interface": event.interface or ""},
        )
        self._suspend_work_item(wi, "episode_suspended")
        return Advance()

    def _suspend_work_item(self, wi: WorkItem, kind: str) -> None:
        """Release the worker and suspend a work item awaiting a boundary reply."""
        if attempt := self._ledger.latest_attempt(wi):
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.finished_at = now_iso()
            self._ledger.emitter.emit_attempt(attempt)
        wi.status = WorkItemStatus.BLOCKED
        self._ledger.emit(
            kind, work_item_id=wi.work_item_id, operator_id=wi.operator_id
        )

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
        activation, _ = self._create_child(
            spawn, operator_id, dispatchable=False, value_ref=None
        )
        return activation.activation_id

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
        advance = Advance()
        activation, wi = self._create_child(
            spawn, operator_id, dispatchable=True, value_ref=value_ref
        )
        self._init_child_input(activation, wi, value_ref)
        self._admit(wi.work_item_id, advance)
        return advance

    def create_fanout_child(self, spawn: str, value_ref: ValueRef) -> str:
        """Create one producer-fanout child (unadmitted) and return its task id.

        The child carries ``value_ref``, a frozen reference to one element of the
        producer's collection, as its child-init input. It stays blocked on its input
        manifest until the runtime records the child-entry accepted input.
        """
        activation, wi = self._create_child(
            spawn, None, dispatchable=True, value_ref=value_ref
        )
        self._init_child_input(activation, wi, value_ref)
        return wi.legacy_task_id

    def _init_child_input(
        self, activation: Activation, wi: WorkItem, value_ref: ValueRef | None
    ) -> None:
        """Declare a child agent's input manifest and mint an inline child-init input.

        A child agent with a declared entry port gains a continuation requiring that
        port. An inline child-init value (a ``spawn_agent`` payload) is minted here; a
        producer-collection reference is left for the runtime to resolve and record.
        """
        entry_port = self._topology.agent_entry_port(wi.operator_id)
        if entry_port is None:
            return
        self._ledger.continuations[wi.work_item_id] = Continuation(
            work_item_id=wi.work_item_id, required_ports={entry_port}
        )
        if value_ref is not None and value_ref.kind == "inline":
            self.record_accepted_input(
                AcceptedInput(
                    activation_id=activation.activation_id,
                    target_port=entry_port,
                    occurrence_key=str(activation.child_index or 0),
                    provenance="spawn_element",
                    members=(
                        AcceptedInputMember(
                            source_operator_id=self._ledger.scopes[
                                activation.scope_id
                            ].owner_operator_id
                            or activation.operator_id,
                            source_activation_id=activation.activation_id,
                            child_index=activation.child_index,
                            outcome=PublicationOutcome.SUCCESS,
                            value_ref=value_ref,
                        ),
                    ),
                )
            )

    def agent_entry_port(self, operator_id: str) -> str | None:
        """The single declared input port of an agent child body, or None.

        A spawn child agent declares exactly one input port (compile-enforced); a leaf
        child or an agent with no declared input has no entry port.
        """
        return self._topology.agent_entry_port(operator_id)

    def record_accepted_input(self, accepted: AcceptedInput) -> None:
        """Record a durable accepted input on an agent's target port (idempotent)."""
        existing = self._accepted_by_activation.setdefault(accepted.activation_id, [])
        if any(a.target_port == accepted.target_port for a in existing):
            return
        self._accepted_inputs.append(accepted)
        existing.append(accepted)

    def accepted_inputs_for(self, activation_id: str) -> tuple[AcceptedInput, ...]:
        """The recorded accepted inputs for one activation, ordered by ordinal."""
        return tuple(
            sorted(
                self._accepted_by_activation.get(activation_id, ()),
                key=lambda a: (a.ordinal, a.target_port),
            )
        )

    def accepted_inputs_for_task(self, task_id: str) -> tuple[AcceptedInput, ...]:
        """The recorded accepted inputs for a task's activation, ordered by ordinal."""
        wi = self._ledger.work_item_for_task(task_id)
        return self.accepted_inputs_for(wi.activation_id) if wi else ()

    def blocked_input_agents(self) -> list[str]:
        """Task ids of agents blocked on an unsatisfied declared-input manifest."""
        pending: list[str] = []
        for wi in self._ledger.work_items.values():
            if wi.status is not WorkItemStatus.BLOCKED or not wi.legacy_task_id:
                continue
            cont = self._ledger.continuations.get(wi.work_item_id)
            if cont is None or not cont.required_ports:
                continue
            have = {a.target_port for a in self.accepted_inputs_for(wi.activation_id)}
            if not cont.required_ports <= have:
                pending.append(wi.legacy_task_id)
        return pending

    def reconsider_admission(self, task_id: str) -> Advance:
        """Re-attempt admission of a work item after its input manifest changed."""
        advance = Advance()
        wi = self._ledger.work_item_for_task(task_id)
        if wi is not None and wi.status is WorkItemStatus.BLOCKED:
            self._admit(wi.work_item_id, advance)
        return advance

    def agent_input_plan(self, task_id: str) -> AgentInputPlan | None:
        """The engine's per-port input membership for an agent, resolved by the runtime.

        Covers edge-bound ports (a direct producer or a join/merge aggregate) whose
        sources have settled and are not yet recorded. Membership and ordering are the
        engine's decision — declared by the edge and the join's child order, never
        arrival. A fan-out child's inline entry port is minted at materialization and is
        not returned here.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        cont = self._ledger.continuations.get(wi.work_item_id)
        if cont is None or not cont.required_ports:
            return None
        have = {a.target_port for a in self.accepted_inputs_for(wi.activation_id)}
        ports: list[InputPortPlan] = []
        for port in sorted(cont.required_ports):
            if port in have:
                continue
            resolved = self._port_members(wi.operator_id, port)
            if resolved is None:
                continue
            provenance, members = resolved
            ports.append(
                InputPortPlan(target_port=port, provenance=provenance, members=members)
            )
        if not ports:
            return None
        return AgentInputPlan(activation_id=wi.activation_id, ports=tuple(ports))

    def _port_members(
        self, agent_op: str, port: str
    ) -> tuple[str, tuple[InputMemberPlan, ...]] | None:
        """The ordered members feeding a port, or None if a source is unsettled."""
        sources = [
            edge.from_op
            for edge in self._topology.bundle.template.edges
            if edge.to_op == agent_op and edge.to_port == port and not edge.feedback
        ]
        if not sources:
            return None
        provenance = "producer"
        members: list[InputMemberPlan] = []
        ordinal = 0
        for source in sources:
            if self._topology.kind(source) is OperatorKind.JOIN:
                provenance = "join_aggregate"
                aggregate = self._ledger.aggregate_by_join.get(source)
                if aggregate is None:
                    return (
                        None  # the join has not released and frozen its aggregate yet
                    )
                for member in sorted(aggregate.members, key=lambda m: m.child_key):
                    child_op = (
                        self._ledger.activations[member.child_activation_id].operator_id
                        if member.child_activation_id in self._ledger.activations
                        else source
                    )
                    members.append(
                        self._member_plan(
                            child_op,
                            member.child_activation_id,
                            (
                                int(member.child_key)
                                if member.child_key.isdigit()
                                else None
                            ),
                            member.outcome,
                            member.value_ref,
                            ordinal,
                        )
                    )
                    ordinal += 1
            else:
                src_wi_id = self._ledger.wi_by_operator.get(source)
                src_wi = self._ledger.work_items.get(src_wi_id) if src_wi_id else None
                if src_wi is None or src_wi.outcome is None:
                    return None
                members.append(
                    self._member_plan(
                        source,
                        src_wi.activation_id,
                        None,
                        src_wi.outcome,
                        ValueRef(
                            kind="legacy_task_result",
                            legacy_task_id=src_wi.legacy_task_id,
                        ),
                        ordinal,
                    )
                )
                ordinal += 1
        return provenance, tuple(members)

    @staticmethod
    def _member_plan(
        source_operator_id: str,
        source_activation_id: str,
        child_index: int | None,
        outcome: PublicationOutcome,
        value_ref: ValueRef | None,
        ordinal: int,
    ) -> InputMemberPlan:
        return InputMemberPlan(
            source_operator_id=source_operator_id,
            source_activation_id=source_activation_id,
            child_index=child_index,
            outcome=outcome.value,
            value_ref_kind=value_ref.kind if value_ref else "empty",
            legacy_task_id=value_ref.legacy_task_id if value_ref else None,
            collection_key=value_ref.collection_key if value_ref else None,
            literal=value_ref.literal if value_ref else None,
            ordinal=ordinal,
        )

    def _create_child(
        self,
        spawn: str,
        operator_id: str | None,
        *,
        dispatchable: bool,
        value_ref: ValueRef | None,
    ) -> tuple[Activation, WorkItem]:
        spawn_op = self._ledger.handle_operator(spawn)
        spawn_op_obj = self._topology.operators[spawn_op]
        child_ref = (
            spawn_op_obj.child_template_ref
            if isinstance(spawn_op_obj, (SpawnRegion, AgentOperator))
            else None
        )
        body_ref = operator_id or child_ref or spawn_op
        body_op = self._topology.operators.get(body_ref)
        if spawn_op in self._denied_spawns:
            self._ledger.emit(
                "child_rejected", operator_id=spawn_op, detail={"reason": "denied"}
            )
            raise RegionError(f"spawn {spawn_op!r} was denied; no child may be created")
        scope_id = self._require_child_init_scope(spawn)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is not CapabilityStatus.OPEN:
            self._ledger.emit(
                "child_rejected",
                operator_id=spawn_op,
                detail={"reason": cap.status.value},
            )
            raise RegionError(
                f"spawn {spawn_op!r} child-init capability is {cap.status.value}; "
                "no child may be created"
            )
        body_opens_scope = self._topology.kind(body_ref) in _CHILD_INIT_OPENERS or (
            self._topology.kind(body_ref) is OperatorKind.LOOP_CONTEXT
        )
        # A leaf child dispatches directly; an agent child dispatches and owns its own
        # child-init scope (recursion). A spawn/loop child body stays trace-level.
        if dispatchable and not isinstance(body_op, (LeafOperator, AgentOperator)):
            raise RegionError(
                f"child body {body_ref!r} is not live-dispatchable; only a leaf or "
                "agent body is"
            )
        if dispatchable and body_opens_scope and not isinstance(body_op, AgentOperator):
            raise RegionError(
                f"child body {body_ref!r} opens a scope; only a leaf or agent body is "
                "live-dispatchable"
            )
        # Validate every budget before materializing, so a rejected child leaves no
        # half-open region whose join could never drain.
        self._charge_activation()
        if body_opens_scope:
            self._check_scope_depth(scope_id)
        index = self._ledger.scope_population[scope_id]
        activation = Activation(
            activation_id=new_activation_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id=scope_id,
            operator_id=body_ref,
            kind="child",
            child_index=index,
        )
        self._ledger.add_activation(activation)
        effect, recovery = _effect_recovery(body_op)
        child_wi = WorkItem(
            work_item_id=new_work_item_id(),
            activation_id=activation.activation_id,
            operator_id=body_ref,
            legacy_task_id=activation.activation_id if dispatchable else "",
            value_ref=value_ref,
            child_input=value_ref,
            effect_class=effect,
            recovery=recovery,
            replay_contract=self._topology.replay.get(body_ref),
        )
        self._ledger.work_items[child_wi.work_item_id] = child_wi
        self._ledger.wi_by_activation[activation.activation_id] = child_wi.work_item_id
        if dispatchable:
            self._ledger.wi_by_task[child_wi.legacy_task_id] = child_wi.work_item_id
        cap.outstanding += 1
        self._ledger.emit(
            "child_spawned",
            operator_id=body_ref,
            detail={"scope": scope_id, "index": str(index)},
        )
        # A spawn/loop child body opens its nested scope eagerly; a dispatchable agent
        # child opens its own child-init scope lazily, only when it first spawns.
        if self._topology.kind(body_ref) is OperatorKind.SPAWN:
            self._open_child_init_scope(
                activation.activation_id, parent_scope_id=scope_id
            )
        elif self._topology.kind(body_ref) is OperatorKind.LOOP_CONTEXT:
            self._open_loop(activation.activation_id, parent_scope_id=scope_id)
        return activation, child_wi

    def seal_spawn(self, spawn: str) -> Advance:
        """Seal a spawn's child-init capability; no further children may be created."""
        scope_id = self._require_child_init_scope(spawn)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.SEALED
            self._ledger.emit(
                "child_init_sealed",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )
        return self._maybe_release_join(scope_id)

    def _settle_agent_regions(self, activation_id: str) -> Advance:
        """Settle every declared child region of a completed agent under its residual.

        Each entered region seals its open child-init capability so its join releases on
        drain; a declared-but-never-entered region opens as a zero-child region and
        seals, so its join releases empty. A cancel residual revokes and cancels the
        region's children instead. Opening an unused region is skipped when it would
        exceed the scope-depth budget, since the agent could never have entered it. A
        non-spawning agent owns no region.
        """
        advance = Advance()
        agent = self._ledger.activations.get(activation_id)
        op = self._topology.operators.get(agent.operator_id) if agent else None
        if not isinstance(op, AgentOperator) or agent is None:
            return advance
        room = (
            self._ledger.scopes[agent.scope_id].depth + 1
            <= self._budget.max_scope_depth
        )
        for ref in op.child_region_refs:
            opener = self._ledger.region_openers.get((activation_id, ref.spawn_ref))
            if opener is None:
                if not room:
                    continue
                opener = self._region_opener(activation_id, ref.spawn_ref)
            advance.extend(self._settle_owned_region(opener))
        return advance

    def _agent_terminal_regions(self, operator_id: str, activation_id: str) -> Advance:
        """Settle an agent's declared regions when the agent completes, else no-op."""
        if isinstance(self._topology.operators.get(operator_id), AgentOperator):
            return self._settle_agent_regions(activation_id)
        return Advance()

    def _settle_owned_region(self, opener: str) -> Advance:
        scope_id = self._ledger.scope_by_activation.get(opener)
        advance = Advance()
        if scope_id is None or not self._close_owned_region(scope_id, advance):
            return advance
        return advance.extend(self._maybe_release_join(scope_id))

    def _close_owned_region(self, scope_id: str, advance: Advance) -> bool:
        """Close a terminal agent's open region under its residual policy, adding the
        children it cancels to ``advance``; returns whether it was open."""
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if cap is None or cap.status is not CapabilityStatus.OPEN:
            return False
        owner = self._ledger.scopes[scope_id].owner_operator_id
        join = self._join_of_scope(scope_id)
        if self._residual_policy(join, ResidualPolicy.DRAIN) is ResidualPolicy.CANCEL:
            cap.status = CapabilityStatus.REVOKED
            self._ledger.emit(
                "child_init_revoked",
                operator_id=owner,
                detail={"scope": scope_id, "reason": "agent_terminal"},
            )
            advance.cancelled.extend(self._cancel_residual_subtrees(scope_id))
        else:
            cap.status = CapabilityStatus.SEALED
            self._ledger.emit(
                "child_init_sealed",
                operator_id=owner,
                detail={"scope": scope_id, "reason": "agent_terminal"},
            )
        return True

    def revoke_spawn(self, spawn: str) -> None:
        """Revoke a spawn's child-init capability as a progress transition.

        Distinct from sealing: revocation withdraws the capability, while sealing marks
        a producer done. Both close the child-init axis once outstanding children drain.
        """
        scope_id = self._require_child_init_scope(spawn)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.REVOKED
            self._ledger.emit(
                "child_init_revoked",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )

    def settle_child(
        self,
        child_activation_id: str,
        *,
        outcome: PublicationOutcome = PublicationOutcome.SUCCESS,
        value_ref: ValueRef | None = None,
    ) -> Advance:
        """Record a child activation's terminal outcome and drain its capability."""
        activation = self._ledger.activations.get(child_activation_id)
        if activation is None or activation.kind != "child":
            raise RegionError(f"unknown child activation {child_activation_id!r}")
        wi = self._ledger.work_items[self._ledger.wi_by_activation[child_activation_id]]
        return self._settle_child_wi(wi, activation, outcome, value_ref)

    def _settle_child_wi(
        self,
        wi: WorkItem,
        activation: Activation,
        outcome: PublicationOutcome,
        value_ref: ValueRef | None,
    ) -> Advance:
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return Advance()
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = outcome
        wi.value_ref = value_ref
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)
        self._ledger.private_state.release(wi.activation_id)
        cap = self._capability(activation.scope_id, ProgressAxis.CHILD_INIT)
        cap.outstanding = max(0, cap.outstanding - 1)
        spawn_op = self._ledger.scopes[activation.scope_id].owner_operator_id or ""
        self._publish_keyed(spawn_op, activation, outcome, value_ref)
        self._ledger.emit(
            "child_settled",
            operator_id=activation.operator_id,
            detail={"scope": activation.scope_id, "outcome": outcome.value},
        )
        return self._maybe_release_join(activation.scope_id)

    def route_branch(self, branch_op: str, selected_port: str) -> Advance:
        """Route a branch record to the selected port; settle the other ports empty."""
        if self._topology.kind(branch_op) is not OperatorKind.BRANCH:
            raise RegionError(f"{branch_op!r} is not a branch region")
        advance = Advance()
        skipped: set[str] = set()
        for successor in sorted(self._topology.forward.get(branch_op, ())):
            from_port = self._topology.edge_from_port(branch_op, successor)
            if from_port in (selected_port, None):
                self._release_one(successor, branch_op, advance)
            else:
                self._settle_empty_successor(successor, skipped)
        self._ledger.emit(
            "branch_routed", operator_id=branch_op, detail={"port": selected_port}
        )
        return advance

    def loop_feedback(self, loop: str, *, value_ref: ValueRef | None = None) -> str:
        """Re-materialize a loop body at the next loop-time coordinate.

        Enforces well-founded logical time: loop_time strictly increases and stays under
        the iteration budget, so a finite prefix is acyclic after time unrolling.
        Returns the iteration activation id.
        """
        scope_id = self._require_loop_scope(loop)
        loop_op = self._ledger.scopes[
            scope_id
        ].owner_operator_id or self._ledger.handle_operator(loop)
        cap = self._capability(scope_id, ProgressAxis.LOOP_TIME)
        if cap.status is not CapabilityStatus.OPEN:
            raise RegionError(
                f"loop {loop_op!r} is {cap.status.value}; no feedback may arrive"
            )
        next_time = self._ledger.loop_time.get(scope_id, 0) + 1
        if next_time > self._budget.max_loop_iterations:
            self._exhaust_budget("loop_iterations", self._budget.max_loop_iterations)
        self._charge_activation()
        self._ledger.loop_time[scope_id] = next_time
        cap.coordinate = next_time
        cap.outstanding += 1
        activation = Activation(
            activation_id=new_activation_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id=scope_id,
            operator_id=loop_op,
            kind="iteration",
            loop_time=next_time,
        )
        self._ledger.add_activation(activation)
        self._ledger.records.append(
            Record(
                operator_id=loop_op,
                activation_id=activation.activation_id,
                scope_id=scope_id,
                loop_time=next_time,
                value_ref=value_ref,
            )
        )
        self._ledger.emit(
            "loop_feedback", operator_id=loop_op, detail={"loop_time": str(next_time)}
        )
        return activation.activation_id

    def settle_iteration(self, iteration_activation_id: str) -> Advance:
        """Mark a loop iteration terminal and drain the loop-time capability."""
        activation = self._ledger.activations.get(iteration_activation_id)
        if activation is None or activation.kind != "iteration":
            raise RegionError(f"unknown loop iteration {iteration_activation_id!r}")
        cap = self._capability(activation.scope_id, ProgressAxis.LOOP_TIME)
        cap.outstanding = max(0, cap.outstanding - 1)
        return self._maybe_egress_loop(activation.scope_id)

    def loop_seal(self, loop: str) -> Advance:
        """Seal a loop: no further feedback; egress once pending iterations drain."""
        scope_id = self._require_loop_scope(loop)
        cap = self._capability(scope_id, ProgressAxis.LOOP_TIME)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.SEALED
            self._ledger.emit(
                "loop_sealed",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )
        return self._maybe_egress_loop(scope_id)

    def deny_spawn(
        self, spawn_op: str, interface: str, *, kind: DenialKind = DenialKind.AUTHORITY
    ) -> None:
        """Record a definitive dynamic authorization denial at a spawn site.

        A denial creates no child activation and no resident claim, and is separate from
        quota/rate/capacity/transport outcomes. It does not seal the child-init
        capability: grant denial and cardinality sealing stay distinct.
        """
        self._denied_spawns.add(spawn_op)
        scope_id = self._ledger.scope_id_for(spawn_op)
        self._decisions.append(
            AuthorityDecision(
                grant_id=self._grant_for_scope(scope_id or "").grant_id,
                interface=interface,
                kind=AuthorityDecisionKind.DENIED,
                operator_id=spawn_op,
                scope_id=scope_id,
                denial_kind=kind,
                reason=f"interface {interface!r} outside spawn-site {kind.value} face",
            )
        )
        self._ledger.emit(
            "policy_denied" if kind is DenialKind.POLICY else "authority_denied",
            operator_id=spawn_op,
        )

    def can_delegate(self, region_op: str, interface: str) -> bool:
        """Whether a child of ``region_op`` may itself delegate ``interface``."""
        scope_id = self._ledger.scope_id_for(region_op)
        return (
            scope_id is not None
            and interface in self._grant_for_scope(scope_id).delegate
        )

    # ------------------------------------------------------------------ #
    # Cancellation (a durable semantic event)
    # ------------------------------------------------------------------ #

    def cancel_instance(self) -> Advance:
        """Cancel the whole workflow instance: the root scope and every descendant."""
        return self.cancel_scope(self._ledger.root_scope.scope_id)

    def fail_instance(self, reason: str) -> Advance:
        """Fail the whole workflow instance as a recorded terminal event.

        No scope admits another child, every unsettled leaf or agent settles as a
        declared failure, and every unpublished declared output resolves to one.
        """
        return self._fail_scope_tree(self._ledger.root_scope.scope_id, reason)

    @_ds_drive(ControlPlaneWindow.POST_START)
    def _fail_scope_tree(self, scope_id: str, reason: str) -> Advance:
        self._ledger.emit("instance_failed", detail={"reason": reason})
        for sid in self._ledger.scope_subtree(scope_id):
            self._revoke_progress(sid)
        advance = Advance()
        for wi in list(self._ledger.work_items.values()):
            if wi.status in TERMINAL_WORK_ITEM_STATUSES or self._topology.kind(
                wi.operator_id
            ) not in (OperatorKind.LEAF, OperatorKind.AGENT):
                continue
            self._fail_open_attempt(wi, reason)
            advance.extend(self._settle_failed_wi(wi))
        for slot in list(self._slots.values()):
            self._write_publication(slot, PublicationOutcome.DECLARED_FAILURE, None)
        return advance

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
        if scope_id not in self._ledger.scopes:
            raise RegionError(f"{scope_id!r} is no cancellable scope")
        advance = Advance()
        for sid in self._ledger.scope_subtree(scope_id):
            advance.extend(self._cancel_scope(sid))
        return advance

    def _cancel_scope(self, scope_id: str) -> Advance:
        self._ledger.emit("scope_cancelled", detail={"scope": scope_id})
        self._revoke_progress(scope_id)
        scope = self._ledger.scopes[scope_id]
        cancelled = self._apply_cancellation_residual(scope_id)
        for wi in self._ledger.scope_work_items(scope_id, kinds=("leaf", "agent")):
            if wi.status not in TERMINAL_WORK_ITEM_STATUSES:
                self._cancel_work_item(wi)
                cancelled.append(wi.legacy_task_id)
        if scope.grant_id and scope.grant_id in self._grants:
            grant = self._grants[scope.grant_id]
            if not grant.revoked:
                self._grants[scope.grant_id] = grant.model_copy(
                    update={"revoked": True}
                )
                self._ledger.emit(
                    "grant_revoked",
                    operator_id=scope.owner_operator_id,
                    detail={"scope": scope_id},
                )
        advance = self._resolve_cancelled_outputs(scope_id)
        advance.cancelled.extend(cancelled)
        return advance

    def _revoke_progress(self, scope_id: str) -> None:
        """Revoke a scope's open child-init and loop-time capabilities."""
        scope = self._ledger.scopes[scope_id]
        for axis in (ProgressAxis.CHILD_INIT, ProgressAxis.LOOP_TIME):
            cap = self._ledger.capabilities.get((scope_id, axis))
            if cap is not None and cap.status is CapabilityStatus.OPEN:
                cap.status = CapabilityStatus.REVOKED
                self._ledger.emit(
                    (
                        "child_init_revoked"
                        if axis is ProgressAxis.CHILD_INIT
                        else "loop_revoked"
                    ),
                    operator_id=scope.owner_operator_id,
                    detail={"scope": scope_id},
                )

    def _apply_cancellation_residual(self, scope_id: str) -> list[str]:
        """Apply a cancelled scope's join residual policy to its materialized children;
        returns the task ids it cancelled.

        A declared ``drain``/``continue`` leaves materialized children to settle; the
        default (``cancel``, and any scope without a declared policy) cancels every not-
        yet-settled child.
        """
        join = self._join_of_scope(scope_id)
        if self._residual_policy(join, ResidualPolicy.CANCEL) is ResidualPolicy.CANCEL:
            return [
                wi.legacy_task_id for wi in self._cancel_residual_children(scope_id)
            ]
        return []

    def _resolve_cancelled_outputs(self, scope_id: str) -> Advance:
        """Release a cancelled scope's join as its cancellation outcome.

        Only a root-level scope publishes it and delivers its record.
        """
        if scope_id in self._ledger.released_scopes:
            return Advance()
        owner_op = self._ledger.scopes[scope_id].owner_operator_id or ""
        join = self._join_of_scope(scope_id)
        if join is not None:
            outcome = self._no_winner_outcome(join)
            release_op = join.operator_id
        elif self._topology.kind(owner_op) is OperatorKind.LOOP_CONTEXT:
            outcome = PublicationOutcome.EXPLICIT_EMPTY
            release_op = owner_op
        else:
            return Advance()
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        self._ledger.emit(
            "join_released" if join is not None else "loop_egress",
            operator_id=release_op,
            detail={"outcome": outcome.value, "cancelled": "true"},
        )
        self._frontier_closed(scope_id)
        if not self._ledger.root_level(scope_id):
            return Advance()
        self._publish(release_op, outcome, ValueRef(kind="empty"))
        return self._deliver_record(
            release_op,
            self._ledger.control_activation(release_op),
            ValueRef(kind="empty"),
        )

    def _cancel_work_item(self, wi: WorkItem) -> None:
        # A cancelled in-flight external effect is not compensated here; compensation on
        # cancel rides with the deferred effect-commit machinery.
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        wi.status = WorkItemStatus.CANCELLED
        self._ledger.private_state.release(wi.activation_id)
        self._publish(
            wi.operator_id, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
        )
        # Recorded before the emitter reads the trace: a cancelled item with no attempt
        # has no other event carrying its work_item_id, so this is the sole source the
        # work-item span's "or latest matching event" end-time rule can find.
        self._ledger.emit(
            "work_item_cancelled",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
        )
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)

    # ------------------------------------------------------------------ #
    # Record delivery, control firing, and closure
    # ------------------------------------------------------------------ #

    def _deliver_record(
        self, operator_id: str, activation_id: str, value_ref: ValueRef | None
    ) -> Advance:
        self._ledger.records.append(
            Record(
                operator_id=operator_id,
                activation_id=activation_id,
                scope_id=self._ledger.root_scope.scope_id,
                value_ref=value_ref,
            )
        )
        self._ledger.emit("record_delivered", operator_id=operator_id)
        advance = Advance()
        for successor in sorted(self._topology.forward.get(operator_id, ())):
            self._release_one(successor, operator_id, advance)
        return advance

    def _release_one(self, successor: str, from_op: str, advance: Advance) -> None:
        """Deliver a record to one successor: fire a control op, or admit a leaf."""
        if self._topology.is_control(successor):
            cont = self._ledger.continuations.get(_control_key(successor))
            if cont is None or self._failures.region_failed(successor):
                return
            cont.waiting_on.discard(from_op)
            if not cont.waiting_on:
                self._fire_control(successor, advance)
            return
        wi_id = self._ledger.wi_by_operator.get(successor)
        cont = self._ledger.continuations.get(wi_id) if wi_id else None
        if cont is None:
            return
        cont.waiting_on.discard(from_op)
        if not cont.waiting_on:
            self._admit(cont.work_item_id, advance)

    def _fire_control(self, operator_id: str, advance: Advance) -> None:
        kind = self._topology.kind(operator_id)
        if kind in _CHILD_INIT_OPENERS:
            self._open_child_init_scope(self._ledger.control_activation(operator_id))
        elif kind is OperatorKind.LOOP_CONTEXT:
            self._open_loop(self._ledger.control_activation(operator_id))
        elif kind is OperatorKind.MERGE:
            self._ledger.emit("merge_combined", operator_id=operator_id)
            advance.extend(
                self._deliver_record(
                    operator_id, self._ledger.control_activation(operator_id), None
                )
            )
        elif kind is OperatorKind.BRANCH:
            branch = self._topology.operators[operator_id]
            selection = branch.selection if isinstance(branch, BranchRegion) else None
            if selection and any(
                self._topology.edge_from_port(operator_id, s) == selection
                for s in self._topology.forward.get(operator_id, ())
            ):
                advance.extend(self.route_branch(operator_id, selection))
        # JOIN is released by scope closure, never by an input record.

    def _open_child_init_scope(
        self,
        opener_activation: str,
        *,
        parent_scope_id: str | None = None,
        parent_delegate: tuple[str, ...] | None = None,
    ) -> str:
        if opener_activation in self._ledger.scope_by_activation:
            return self._ledger.scope_by_activation[opener_activation]
        scope = self._new_child_scope(
            opener_activation, parent_scope_id, parent_delegate=parent_delegate
        )
        self._register_scope_owner(scope)
        self._acquire_capability(scope.scope_id, ProgressAxis.CHILD_INIT)
        self._ledger.emit(
            "child_init_acquired",
            operator_id=scope.owner_operator_id,
            detail={"scope": scope.scope_id},
        )
        return scope.scope_id

    def _open_loop(
        self, opener_activation: str, *, parent_scope_id: str | None = None
    ) -> str:
        if opener_activation in self._ledger.scope_by_activation:
            return self._ledger.scope_by_activation[opener_activation]
        scope = self._new_child_scope(opener_activation, parent_scope_id)
        self._register_scope_owner(scope)
        self._ledger.loop_time[scope.scope_id] = 0
        self._acquire_capability(scope.scope_id, ProgressAxis.LOOP_TIME, coordinate=0)
        self._ledger.emit(
            "loop_ingress",
            operator_id=scope.owner_operator_id,
            detail={"scope": scope.scope_id},
        )
        return scope.scope_id

    def _maybe_release_join(self, scope_id: str) -> Advance:
        scope = self._ledger.scopes.get(scope_id)
        if scope is None or scope.owner_operator_id is None:
            return Advance()
        join_op = self._topology.join_for_spawn(scope.owner_operator_id)
        if self._failures.scope_failed(scope_id):
            self._ledger.emit_scope_owner(scope_id)
            return Advance()
        # A join failed at the root blocks only the root-level scope: a nested level
        # still releases, delivering nothing.
        if (
            join_op is None
            or scope_id in self._ledger.released_scopes
            or (
                self._failures.region_failed(join_op)
                and self._ledger.root_level(scope_id)
            )
        ):
            return Advance()
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if cap is None:
            return Advance()
        if (early := self._maybe_early_release(join_op, scope_id, cap)) is not None:
            return early
        if not cap.closed:
            return Advance()
        return self._release_join(join_op, scope_id)

    def _maybe_early_release(
        self, join_op: str, scope_id: str, cap: ProgressCapability
    ) -> Advance | None:
        """Release an early join once its qualifier threshold is met, per its rule.

        A monotone rule (``any``/``first_k``/a monotone predicate) releases on the first
        witness; a non-monotone predicate never releases early, waiting for closure.
        """
        join = self._topology.operators[join_op]
        if not isinstance(join, JoinRegion) or join.completion not in _EARLY_JOINS:
            return None
        threshold, monotone = self._early_rule(join)
        if not monotone or len(self._qualifiers(scope_id)) < threshold:
            return None
        return self._release_join(join_op, scope_id)

    def _release_join(self, join_op: str, scope_id: str) -> Advance:
        """Release a scope's join by its completion rule.

        Only a root-level scope freezes the join's aggregate, publishes it, and
        delivers its record downstream. A scope nested under a spawned child shares the
        join operator with its sibling levels, so its release is local to its level.
        """
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        join = self._topology.operators[join_op]
        assert isinstance(join, JoinRegion)
        outcome, value_ref = self._join_result(join, scope_id)
        children = self._materialized_children(scope_id)
        nested = not self._ledger.root_level(scope_id)
        if not nested:
            self._freeze_region_aggregate(join, join_op, scope_id)
        if outcome is PublicationOutcome.DECLARED_FAILURE:
            self._frontier_closed(scope_id)
            cancelled = self._apply_residual_policy(join, scope_id)
            advance = self._fail_resolved_join(join_op, scope_id, children)
            advance.cancelled.extend(cancelled)
            return advance
        self._ledger.emit(
            "join_released",
            operator_id=join_op,
            detail={"outcome": outcome.value, "children": str(len(children))},
        )
        self._frontier_closed(scope_id)
        cancelled = self._apply_residual_policy(join, scope_id)
        if nested:
            return Advance(cancelled=cancelled)
        self._publish(join_op, outcome, value_ref)
        advance = self._deliver_record(
            join_op, self._ledger.control_activation(join_op), value_ref
        )
        advance.cancelled.extend(cancelled)
        return advance

    def _fail_resolved_join(
        self, join_op: str, scope_id: str, children: list[Activation]
    ) -> Advance:
        """Settle a join that resolved as a declared failure as a failed region.

        It delivers no record, and everything downstream of it fails as the dependent
        of its first failed child, or of the join itself when no child failed. A join
        of a scope nested under a spawned child fails only that scope, since sibling
        scopes share its operator.
        """
        if not self._ledger.root_level(scope_id):
            self._failures.mark_scope_failed(scope_id)
            self._ledger.emit(
                "region_failed", operator_id=join_op, detail={"scope": scope_id}
            )
            return Advance()
        cascade = Advance()
        self._settle_region_failed(join_op)
        self._fail_downstream(join_op, cascade, {join_op})
        failed_child = next(
            (
                wi
                for child in children
                if (
                    wi := self._ledger.work_items[
                        self._ledger.wi_by_activation[child.activation_id]
                    ]
                ).outcome
                is PublicationOutcome.DECLARED_FAILURE
            ),
            None,
        )
        if failed_child is not None and failed_child.legacy_task_id:
            self._failures.name_failures(
                cascade.failed, dependency_failed(failed_child.legacy_task_id)
            )
            cascade.failed.insert(0, failed_child.legacy_task_id)
            return cascade
        self._failures.name_failures(
            cascade.failed, f"join {join_op} resolved no winner"
        )
        return cascade

    def _freeze_region_aggregate(
        self, join: JoinRegion, join_op: str, scope_id: str
    ) -> None:
        """Capture an immutable region-join aggregate at release, ordered by child key.

        A full-closure join freezes every settled child; an early join freezes only its
        selected qualifiers. Membership is fixed here so a residual child never mutates
        the emitted aggregate and a restart replays the same members.
        """
        selected = (
            self._qualifiers(scope_id)
            if join.completion in _EARLY_JOINS
            else self._materialized_children(scope_id)
        )
        members = tuple(
            RegionAggregateMember(
                child_activation_id=child.activation_id,
                child_key=str(
                    child.child_index if child.child_index is not None else 0
                ),
                source_port="out",
                outcome=self._ledger.work_items[
                    self._ledger.wi_by_activation[child.activation_id]
                ].outcome
                or PublicationOutcome.SUCCESS,
                value_ref=self._ledger.work_items[
                    self._ledger.wi_by_activation[child.activation_id]
                ].value_ref,
            )
            for child in selected
        )
        aggregate = RegionJoinAggregate(
            join_operator_id=join_op,
            activation_id=self._ledger.control_activation(join_op),
            members=members,
        )
        self._ledger.region_aggregates.append(aggregate)
        self._ledger.aggregate_by_join[join_op] = aggregate

    def _join_result(
        self, join: JoinRegion, scope_id: str
    ) -> tuple[PublicationOutcome, ValueRef | None]:
        """A join's published outcome and value: a winner's value, or a no-winner mark.

        An early join publishes its lowest-``child_index`` qualifier's value, or its
        declared no-winner outcome; a full-closure join aggregates over its children.
        """
        if join.completion in _EARLY_JOINS:
            qualifiers = self._qualifiers(scope_id)
            threshold, _ = self._early_rule(join)
            if len(qualifiers) >= threshold:
                winner = self._ledger.work_items[
                    self._ledger.wi_by_activation[qualifiers[0].activation_id]
                ]
                return PublicationOutcome.SUCCESS, (
                    winner.value_ref or ValueRef(kind="join_result")
                )
            return self._no_winner_outcome(join), ValueRef(kind="empty")
        outcomes = [
            self._ledger.work_items[
                self._ledger.wi_by_activation[c.activation_id]
            ].outcome
            for c in self._materialized_children(scope_id)
        ]
        return (
            self._join_outcome(join, [o for o in outcomes if o is not None]),
            ValueRef(kind="join_result"),
        )

    def _no_winner_outcome(self, join: JoinRegion) -> PublicationOutcome:
        return (
            PublicationOutcome.DECLARED_FAILURE
            if join.no_winner_failure
            else PublicationOutcome.EXPLICIT_EMPTY
        )

    def _join_outcome(
        self, join: JoinRegion, outcomes: list[PublicationOutcome]
    ) -> PublicationOutcome:
        if not outcomes:
            return PublicationOutcome.EXPLICIT_EMPTY
        if (
            join.completion is JoinCompletion.ALL_SUCCEED
            and PublicationOutcome.DECLARED_FAILURE in outcomes
        ):
            return PublicationOutcome.DECLARED_FAILURE
        return PublicationOutcome.SUCCESS

    def _early_rule(self, join: JoinRegion) -> tuple[int, bool]:
        """The (qualifier threshold, monotone) of an early join's release rule."""
        if join.completion is JoinCompletion.ANY:
            return 1, True
        if join.completion is JoinCompletion.FIRST_K:
            return join.first_k or 1, True
        pred = join.predicate
        return (pred.min_qualifiers if pred else 1), (pred.monotone if pred else True)

    def _qualifiers(self, scope_id: str) -> list[Activation]:
        """A scope's settled children that succeeded, ordered by ``child_index``."""
        return [
            child
            for child in self._materialized_children(scope_id)
            if self._ledger.work_items[
                self._ledger.wi_by_activation[child.activation_id]
            ].outcome
            is PublicationOutcome.SUCCESS
        ]

    def _materialized_children(self, scope_id: str) -> list[Activation]:
        return sorted(
            (
                a
                for a in self._ledger.activations.values()
                if a.scope_id == scope_id and a.kind == "child"
            ),
            key=lambda a: a.child_index if a.child_index is not None else 0,
        )

    def _residual_policy(
        self, join: JoinRegion | None, default: ResidualPolicy
    ) -> ResidualPolicy:
        if join is not None and join.residual_policy:
            return ResidualPolicy(join.residual_policy)
        return default

    def _join_of_scope(self, scope_id: str) -> JoinRegion | None:
        join_op = self._topology.join_for_spawn(
            self._ledger.scopes[scope_id].owner_operator_id or ""
        )
        join = self._topology.operators.get(join_op) if join_op else None
        return join if isinstance(join, JoinRegion) else None

    def _cancel_residual_children(self, scope_id: str) -> list[WorkItem]:
        """Cancel a scope's unsettled children; returns their work items."""
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        cancelled: list[WorkItem] = []
        for child in self._materialized_children(scope_id):
            wi = self._ledger.work_items[
                self._ledger.wi_by_activation[child.activation_id]
            ]
            if wi.status not in TERMINAL_WORK_ITEM_STATUSES:
                self._cancel_work_item(wi)
                cancelled.append(wi)
                if cap is not None:
                    cap.outstanding = max(0, cap.outstanding - 1)
        return cancelled

    def _cancel_residual_subtrees(self, scope_id: str) -> list[str]:
        """Cancel a scope's unsettled children, each with the regions it entered and
        everything under them, as a cancel does; returns the task ids cancelled."""
        cancelled: list[str] = []
        for wi in self._cancel_residual_children(scope_id):
            cancelled.append(wi.legacy_task_id)
            for opener in self._entered_region_openers(wi.activation_id):
                for sid in self._ledger.scope_subtree(
                    self._ledger.scope_by_activation[opener]
                ):
                    cancelled.extend(self._cancel_scope(sid).cancelled)
        return cancelled

    def _entered_region_openers(self, agent_activation: str) -> list[str]:
        return [
            opener
            for (activation_id, _), opener in self._ledger.region_openers.items()
            if activation_id == agent_activation
            and opener in self._ledger.scope_by_activation
        ]

    def _apply_residual_policy(self, join: JoinRegion, scope_id: str) -> list[str]:
        """Govern a released early join's not-yet-settled materialized children;
        returns the ones it cancelled.

        ``continue`` leaves child-init open; ``drain`` seals it; ``cancel`` revokes it
        and cancels the residual children. Residual settlements stay ledger-visible but
        never become the join's implicit winner output.
        """
        if join.completion not in _EARLY_JOINS:
            return []
        policy = self._residual_policy(join, ResidualPolicy.CONTINUE)
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if policy is ResidualPolicy.CONTINUE or cap is None:
            return []
        cancel = policy is ResidualPolicy.CANCEL
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.REVOKED if cancel else CapabilityStatus.SEALED
            self._ledger.emit(
                "child_init_revoked" if cancel else "child_init_sealed",
                operator_id=self._ledger.scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id, "residual": policy.value},
            )
        return self._cancel_residual_subtrees(scope_id) if cancel else []

    def _maybe_egress_loop(self, scope_id: str) -> Advance:
        if not scope_id or scope_id in self._ledger.released_scopes:
            return Advance()
        cap = self._ledger.capabilities.get((scope_id, ProgressAxis.LOOP_TIME))
        if cap is None or not cap.closed:
            return Advance()
        self._ledger.released_scopes.add(scope_id)
        if (owner_act := self._ledger.scopes[scope_id].owner_activation_id) is not None:
            self._ledger.emitter.emit_activation(owner_act)
        self._frontier_closed(scope_id)
        loop_op = self._ledger.scopes[scope_id].owner_operator_id or ""
        self._ledger.emit(
            "loop_egress", operator_id=loop_op, detail={"scope": scope_id}
        )
        carried = self._ledger.latest_carried(scope_id)
        self._publish(loop_op, PublicationOutcome.SUCCESS, carried)
        return self._deliver_record(
            loop_op, self._ledger.control_activation(loop_op), carried
        )

    # ------------------------------------------------------------------ #
    # Progress capabilities and scopes
    # ------------------------------------------------------------------ #

    def _acquire_capability(
        self, scope_id: str, axis: ProgressAxis, *, coordinate: int | None = None
    ) -> ProgressCapability:
        cap = ProgressCapability(scope_id=scope_id, axis=axis, coordinate=coordinate)
        self._ledger.capabilities[(scope_id, axis)] = cap
        return cap

    def _capability(self, scope_id: str, axis: ProgressAxis) -> ProgressCapability:
        cap = self._ledger.capabilities.get((scope_id, axis))
        if cap is None:
            raise RegionError(f"scope {scope_id!r} holds no {axis.value} capability")
        return cap

    def _check_scope_depth(self, parent_scope_id: str) -> None:
        if (
            self._ledger.scopes[parent_scope_id].depth + 1
            > self._budget.max_scope_depth
        ):
            self._exhaust_budget("scope_depth", self._budget.max_scope_depth)

    def _new_child_scope(
        self,
        opener_activation: str,
        parent_scope_id: str | None,
        *,
        parent_delegate: tuple[str, ...] | None = None,
    ) -> Scope:
        opener_op = self._ledger.activations[opener_activation].operator_id
        parent = self._ledger.scopes[
            parent_scope_id or self._ledger.root_scope.scope_id
        ]
        self._check_scope_depth(parent.scope_id)
        grant = self._mint_delegated_grant(
            opener_op, parent.scope_id, parent_delegate=parent_delegate
        )
        scope = Scope(
            scope_id=new_scope_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            parent_scope_id=parent.scope_id,
            owner_operator_id=opener_op,
            owner_activation_id=opener_activation,
            grant_id=grant.grant_id,
            depth=parent.depth + 1,
        )
        self._ledger.scopes[scope.scope_id] = scope
        self._grants[grant.grant_id] = grant.model_copy(
            update={"scope_id": scope.scope_id}
        )
        return scope

    def _register_scope_owner(self, scope: Scope) -> None:
        if scope.owner_activation_id:
            self._ledger.scope_by_activation[scope.owner_activation_id] = scope.scope_id
        if scope.owner_operator_id and scope.owner_activation_id:
            self._ledger.owner_acts_by_operator.setdefault(
                scope.owner_operator_id, []
            ).append(scope.owner_activation_id)

    def _frontier_closed(self, scope_id: str) -> None:
        self._ledger.emit("frontier_closed", detail={"scope": scope_id})

    def _charge_activation(self) -> None:
        if self._ledger.dynamic_activations >= self._budget.max_activations:
            self._exhaust_budget("activations", self._budget.max_activations)

    def _exhaust_budget(self, budget: str, limit: int) -> None:
        """Record a durable scope-budget breach, distinct from an authority denial."""
        self._ledger.emit(
            "scope_budget_exhausted", detail={"budget": budget, "limit": str(limit)}
        )
        raise RegionError(f"{budget} budget {limit} exhausted")

    # ------------------------------------------------------------------ #
    # Authority: delegated-grant minting with monotone attenuation
    # ------------------------------------------------------------------ #

    def _mint_delegated_grant(
        self,
        opener_op: str,
        parent_scope_id: str,
        *,
        parent_delegate: tuple[str, ...] | None = None,
    ) -> DelegatedAuthorityGrant:
        parent = self._grant_for_scope(parent_scope_id)
        # An agent-selected region attenuates from the agent's delegate face, supplied
        # here, rather than the enclosing scope's raw delegate face.
        base = parent.delegate if parent_delegate is None else parent_delegate
        opener = self._topology.operators[opener_op]
        ceiling = (
            opener.authority
            if isinstance(opener, (SpawnRegion, AgentOperator))
            else None
        )
        ceiling_invoke = ceiling.invoke if ceiling else base
        ceiling_delegate = ceiling.delegate if ceiling else base
        envelope = self._policy_interfaces()
        invoke = attenuate(base, ceiling_invoke, envelope)
        delegate = attenuate(invoke, ceiling_delegate, envelope)
        grant = DelegatedAuthorityGrant(
            grant_id=new_authority_grant_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id="",
            parent_grant_id=parent.grant_id,
            policy_id=parent.policy_id,
            invoke=invoke,
            delegate=delegate,
            epoch=parent.epoch + 1,
        )
        self._ledger.emit(
            "grant_delegated",
            operator_id=opener_op,
            detail={"invoke": ",".join(invoke), "delegate": ",".join(delegate)},
        )
        return grant

    def effective_invoke_face(self, task_id: str) -> tuple[str, ...]:
        """The interfaces this agent activation may invoke.

        The same effective face the engine authorizes ordinary boundaries against: the
        activation's scope grant under its operator ceiling and the policy envelope. A
        spawned activation reads its own delegated grant, which its parent already
        attenuated, so an interface an ancestor withheld is absent here even where the
        operator's own declared ceiling names it. A task that is not an agent invokes
        nothing through this face.
        """
        wi = self._ledger.work_item_for_task(task_id)
        act = self._ledger.activations.get(wi.activation_id) if wi is not None else None
        op = self._topology.operators.get(wi.operator_id) if wi is not None else None
        if act is None or not isinstance(op, AgentOperator):
            return ()
        invoke, _delegate = self._agent_face_tuples(op, act.scope_id)
        return invoke

    def _grant_for_scope(
        self, scope_id: str
    ) -> AuthorityGrant | DelegatedAuthorityGrant:
        scope = self._ledger.scopes.get(scope_id)
        if scope and scope.grant_id and scope.grant_id in self._grants:
            return self._grants[scope.grant_id]
        return self._ledger.root_grant

    def _policy_interfaces(self) -> tuple[str, ...]:
        # The pinned policy envelope caps every face; the root grant projects it.
        return self._ledger.root_grant.delegate or self._ledger.root_grant.invoke

    # ------------------------------------------------------------------ #
    # Readiness, settlement, publication (static path)
    # ------------------------------------------------------------------ #

    def _open_roots(self) -> Advance:
        advance = Advance()
        for wi in list(self._ledger.work_items.values()):
            cont = self._ledger.continuations.get(wi.work_item_id)
            if cont is not None and not cont.waiting_on and wi.legacy_task_id:
                self._admit(wi.work_item_id, advance)
        for op_id in self._topology.operators:
            # A child-template region opens only when its parent spawn materializes it,
            # and an agent-selected region only when the agent requests it, so neither
            # fires at the root.
            if (
                not self._topology.is_control(op_id)
                or op_id in self._topology.child_templates
                or op_id in self._topology.agent_region_spawns
            ):
                continue
            cont = self._ledger.continuations.get(_control_key(op_id))
            if cont is not None and not cont.waiting_on:
                self._fire_control(op_id, advance)
        return advance

    def _admit(self, work_item_id: str, advance: Advance) -> None:
        """Move a work item whose predecessors settled to READY, gating on authority.

        A declared-input agent is admissible only once its accepted-input manifest is
        satisfied — every required port carries a recorded accepted input — so a fan-out
        child or a downstream merge agent never dispatches before its input is durable.
        """
        wi = self._ledger.work_items[work_item_id]
        if wi.status is not WorkItemStatus.BLOCKED:
            return
        cont = self._ledger.continuations.get(work_item_id)
        if cont is not None and cont.required_ports:
            have = {a.target_port for a in self.accepted_inputs_for(wi.activation_id)}
            if not cont.required_ports <= have:
                return
        interface = self._topology.requested_interface(wi.operator_id)
        if interface is not None and interface not in self._ledger.root_grant.invoke:
            reason = f"interface {interface!r} outside root grant invoke face"
            self._decisions.append(
                AuthorityDecision(
                    work_item_id=work_item_id,
                    grant_id=self._ledger.root_grant.grant_id,
                    interface=interface,
                    kind=AuthorityDecisionKind.DENIED,
                    denial_kind=DenialKind.AUTHORITY,
                    reason=reason,
                )
            )
            wi.failure_reason = f"authority denied: {reason}"
            self._ledger.emit(
                "authority_denied",
                work_item_id=work_item_id,
                operator_id=wi.operator_id,
            )
            advance.extend(self._settle_failed_wi(wi))
            return
        if interface is not None:
            self._decisions.append(
                AuthorityDecision(
                    work_item_id=work_item_id,
                    grant_id=self._ledger.root_grant.grant_id,
                    interface=interface,
                    kind=AuthorityDecisionKind.GRANTED,
                )
            )
        wi.status = WorkItemStatus.READY
        self._ledger.emit(
            "work_item_ready", work_item_id=work_item_id, operator_id=wi.operator_id
        )
        advance.ready.append(wi.legacy_task_id)

    def _settle_empty_successor(
        self, operator_id: str, visited: set[str] | None = None
    ) -> None:
        """Skip a non-selected branch successor and its whole subtree.

        A leaf resolves empty; a control successor clears its pending inputs and is
        skipped so a downstream join never waits on an untaken path. A join itself is
        left to its own scope closure. Two non-selected ports can share a downstream
        operator, so the walk is idempotent per operator.
        """
        visited = visited if visited is not None else set()
        if operator_id in visited:
            return
        visited.add(operator_id)
        if self._topology.is_control(operator_id):
            if self._topology.kind(operator_id) is OperatorKind.JOIN:
                return
            if (
                cont := self._ledger.continuations.get(_control_key(operator_id))
            ) is not None:
                cont.waiting_on.clear()
            self._ledger.emit("region_skipped", operator_id=operator_id)
        else:
            wi_id = self._ledger.wi_by_operator.get(operator_id)
            if wi_id is None:
                return
            wi = self._ledger.work_items[wi_id]
            if wi.status in TERMINAL_WORK_ITEM_STATUSES:
                return
            wi.status = WorkItemStatus.SETTLED
            wi.outcome = PublicationOutcome.EXPLICIT_EMPTY
            self._ledger.emitter.emit_work_item(wi)
            self._ledger.emitter.emit_activation(wi.activation_id)
            self._ledger.private_state.release(wi.activation_id)
            self._publish(
                operator_id, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
            )
        for successor in sorted(self._topology.forward.get(operator_id, ())):
            self._settle_empty_successor(successor, visited)

    def _settle_failure(self, work_item_id: str) -> Advance:
        """Settle a work item as a declared failure and fail everything downstream of
        it."""
        cascade = Advance()
        self._fail_work_item(work_item_id, cascade, set())
        self._declare_failures(self._ledger.work_items[work_item_id], cascade.failed)
        return cascade

    def _declare_failures(self, primary: WorkItem, failed: list[str]) -> None:
        """Name why a failed work item and each task its failure cascaded into failed:
        the work item by its own reason, the rest as its dependents. A task an earlier
        failure named keeps that reason."""
        if not failed:
            return
        self._failures.name_failure(
            primary.legacy_task_id, primary.failure_reason or _DECLARED_FAILURE_REASON
        )
        self._failures.name_failures(failed, dependency_failed(primary.legacy_task_id))

    def _fail_work_item(
        self, work_item_id: str, cascade: Advance, visited: set[str]
    ) -> None:
        wi = self._ledger.work_items[work_item_id]
        if wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = PublicationOutcome.DECLARED_FAILURE
        self._ledger.emitter.emit_work_item(wi)
        self._ledger.emitter.emit_activation(wi.activation_id)
        self._ledger.private_state.release(wi.activation_id)
        self._publish(wi.operator_id, PublicationOutcome.DECLARED_FAILURE, None)
        cascade.failed.append(wi.legacy_task_id)
        self._fail_agent_regions(wi, cascade, visited)
        self._fail_downstream(wi.operator_id, cascade, visited)

    def _fail_agent_regions(
        self, wi: WorkItem, cascade: Advance, visited: set[str]
    ) -> None:
        """Fail every child region a failed agent declares.

        A region the agent never entered fails as a spawn whose input failed does, and
        opens no scope. An entered region's children follow its residual policy, and
        its join fails unless it already released. A spawned agent instance shares its
        regions' operators with its siblings, so only its own scopes fail.
        """
        op = self._topology.operators.get(wi.operator_id)
        if not isinstance(op, AgentOperator):
            return
        instance = self._ledger.activations[wi.activation_id].kind == "child"
        for ref in op.child_region_refs:
            opener = self._ledger.region_openers.get((wi.activation_id, ref.spawn_ref))
            scope_id = self._ledger.scope_by_activation.get(opener) if opener else None
            if scope_id is None:
                if not instance:
                    self._fail_region(ref.spawn_ref, cascade, visited)
                continue
            if (
                scope_id not in self._ledger.released_scopes
                and not self._failures.scope_failed(scope_id)
            ):
                self._fail_entered_region(
                    ref.spawn_ref, scope_id, instance, cascade, visited
                )
            self._close_owned_region(scope_id, cascade)
            self._ledger.emit_scope_owner(scope_id)

    def _fail_entered_region(
        self,
        spawn_op: str,
        scope_id: str,
        instance: bool,
        cascade: Advance,
        visited: set[str],
    ) -> None:
        """Fail the join of one scope a failed agent opened, which has not released.

        The scope is named directly: a recursive agent's levels share the join
        operator, and another level's scope may already have released.
        """
        self._failures.mark_scope_failed(scope_id)
        join_op = self._topology.join_for_spawn(spawn_op)
        if instance or join_op is None:
            self._ledger.emit(
                "region_failed", operator_id=join_op, detail={"scope": scope_id}
            )
        elif join_op not in visited and not self._failures.region_failed(join_op):
            visited.add(join_op)
            self._settle_region_failed(join_op)
            self._fail_downstream(join_op, cascade, visited)

    def _fail_downstream(
        self, operator_id: str, cascade: Advance, visited: set[str]
    ) -> None:
        for successor in sorted(self._topology.forward.get(operator_id, ())):
            if self._topology.is_control(successor):
                self._fail_region(successor, cascade, visited)
            elif succ_wi := self._ledger.wi_by_operator.get(successor):
                self._fail_work_item(succ_wi, cascade, visited)

    def _fail_region(
        self, operator_id: str, cascade: Advance, visited: set[str]
    ) -> None:
        """Settle a control operator a failed input feeds as a declared failure.

        The region never fires. A failed spawn opens no scope and creates no child: it
        fails its child template and its join, and each collection it publishes holds
        one failed member. A join fails whatever its completion rule, unless its scope
        already released.
        """
        if operator_id in visited or self._failures.region_failed(operator_id):
            return
        visited.add(operator_id)
        kind = self._topology.kind(operator_id)
        if kind is OperatorKind.JOIN and self._ledger.region_closed(operator_id):
            return
        self._settle_region_failed(operator_id)
        if kind is OperatorKind.SPAWN:
            self._fail_spawn_template(operator_id, cascade)
            self._publish_keyed(
                operator_id, None, PublicationOutcome.DECLARED_FAILURE, None
            )
            if (join_op := self._topology.join_for_spawn(operator_id)) is not None:
                self._fail_region(join_op, cascade, visited)
        self._fail_downstream(operator_id, cascade, visited)

    def _settle_region_failed(self, operator_id: str) -> None:
        self._failures.mark_region_failed(operator_id)
        self._ledger.emit("region_failed", operator_id=operator_id)
        self._publish(operator_id, PublicationOutcome.DECLARED_FAILURE, None)

    def _fail_spawn_template(self, spawn_op: str, cascade: Advance) -> None:
        """Fail a failed spawn's child template, and the templates nested under it,
        once no live spawn instantiates them."""
        template = self._topology.child_template_of(spawn_op)
        if template is None:
            return
        for failed in self._ledger.template_closure(
            template, self._instantiable_by_live
        ):
            self._publish(failed, PublicationOutcome.DECLARED_FAILURE, None)
            cascade.failed.append(failed)

    def _instantiable_by_live(self, template: str, dead: list[str]) -> bool:
        """Whether a spawn that neither failed nor belongs to a dead template can still
        instantiate ``template``."""
        return any(
            isinstance(op, SpawnRegion)
            and op.child_template_ref == template
            and not self._failures.region_failed(op.operator_id)
            and self._topology.region_owner(op.operator_id) not in dead
            for op in self._topology.operators.values()
        )

    def template_closure(
        self,
        template: str,
        excluded: Callable[[str, list[str]], bool] | None = None,
    ) -> list[str]:
        """A child template and, under an agent template, the child templates of every
        region it declares, however deep; a template ``excluded`` rejects is left out
        together with what is nested under it."""
        return self._ledger.template_closure(template, excluded)

    def _publish(
        self, operator_id: str, outcome: PublicationOutcome, value_ref: ValueRef | None
    ) -> None:
        for slot_key in self._slots_by_operator.get(operator_id, ()):
            self._write_publication(self._slots[slot_key], outcome, value_ref)

    def _publish_keyed(
        self,
        spawn_op: str,
        activation: Activation | None,
        outcome: PublicationOutcome,
        value_ref: ValueRef | None,
    ) -> None:
        """Publish a spawn child's member of each collection the spawn declares, or,
        with no child, the collection's one member."""
        for decl in self._topology.bundle.template.result_declarations:
            if (
                decl.source_ref != spawn_op
                or decl.cardinality is not CardinalityKind.KEYED_COLLECTION
            ):
                continue
            self._write_publication(
                ResultSlot(
                    instance_id=self._ledger.workflow_instance.instance_id,
                    output_id=decl.output_id,
                    source_operator_id=spawn_op,
                    scope_id=activation.scope_id if activation else None,
                    logical_key=str(activation.child_index) if activation else None,
                ),
                outcome,
                value_ref,
            )

    def _write_publication(
        self, slot: ResultSlot, outcome: PublicationOutcome, value_ref: ValueRef | None
    ) -> None:
        if slot.slot_key in self._publications:
            return
        if slot.slot_key not in self._slots:
            self._slots_by_output.setdefault(slot.output_id, []).append(slot.slot_key)
        self._slots[slot.slot_key] = slot.model_copy(update={"published": True})
        self._publications[slot.slot_key] = ResultPublication(
            slot_key=slot.slot_key,
            output_id=slot.output_id,
            outcome=outcome,
            value_ref=value_ref,
        )
        self._ledger.emit(
            "result_published",
            operator_id=slot.source_operator_id,
            slot_key=slot.slot_key,
            outcome=outcome.value,
        )

    def _record_receipt(self, wi: WorkItem, outcome: PublicationOutcome) -> None:
        if wi.invocation_id is None or wi.invocation_id in self._receipts:
            return
        self._receipts[wi.invocation_id] = EffectReceipt(
            invocation_id=wi.invocation_id,
            work_item_id=wi.work_item_id,
            outcome=outcome,
        )
        self._ledger.emit(
            "effect_receipt",
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def output_publication(
        self,
        output_id: str,
        scope_id: str | None = None,
        logical_key: str | None = None,
        sequence: int | None = None,
    ) -> ResultPublication | None:
        """The terminal publication of exactly one slot, if it has one."""
        return self._publications.get(
            slot_identity(
                self._ledger.workflow_instance.instance_id,
                output_id,
                scope_id,
                logical_key,
                sequence,
            )
        )

    def output_slots(self, output_id: str) -> list[ResultSlot]:
        """Every slot a declared output holds so far, pending or published."""
        return [self._slots[key] for key in self._slots_by_output.get(output_id, ())]

    def output_slot(
        self,
        output_id: str,
        scope_id: str | None = None,
        logical_key: str | None = None,
        sequence: int | None = None,
    ) -> ResultSlot | None:
        """Exactly one slot of a declared output, if it holds one."""
        return self._slots.get(
            slot_identity(
                self._ledger.workflow_instance.instance_id,
                output_id,
                scope_id,
                logical_key,
                sequence,
            )
        )

    def published_outputs(self) -> list[tuple[str, ResultDeclaration]]:
        """Each published declaration with the public name it was authored under."""
        return self._topology.published_outputs()

    def resolve_legacy_task(self, task_id: str) -> ResultPublication | None:
        """Resolve a legacy task id's induced output slot (compatibility adapter)."""
        return self.output_publication(f"legacy:{task_id}")

    def legacy_task_value(
        self, task_id: str
    ) -> tuple[PublicationOutcome, ValueRef | None] | None:
        """The settled value a legacy task id reads as, or None while it is unsettled.

        A task compiled from the source resolves its induced output slot. A task the
        engine materialized at run time — a spawned child, a later loop iteration — has
        no slot of its own, so it reads as the value its work item settled with. Either
        way the value is the one bound at settlement and never re-pointed.
        """
        if (publication := self.resolve_legacy_task(task_id)) is not None:
            return publication.outcome, publication.value_ref
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.SETTLED or wi.outcome is None:
            return None
        return wi.outcome, wi.value_ref

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
        return self._boundary_events.get((activation_id, call_correlation))

    def contract_trace(self) -> list[tuple[str, str]]:
        """A compact (kind, subject) projection of the trace for test inspection."""
        return self._ledger.contract_trace()

    def capability(
        self, scope_id: str | None, axis: ProgressAxis
    ) -> ProgressCapability | None:
        return self._ledger.capabilities.get((scope_id, axis)) if scope_id else None

    def scope_for(self, region_op: str) -> str | None:
        return self._ledger.scope_for(region_op)

    def region_scope_for(self, agent_activation: str, role: str) -> str | None:
        """The child-init scope an agent's declared role region opened, if entered."""
        return self._ledger.region_scope_for(agent_activation, role)

    def grant_for(self, region_op: str) -> DelegatedAuthorityGrant | None:
        scope_id = self._ledger.scope_id_for(region_op)
        if scope_id is None:
            return None
        grant_id = self._ledger.scopes[scope_id].grant_id
        return self._grants.get(grant_id) if grant_id else None

    def region_closed(self, region_op: str) -> bool:
        return self._ledger.region_closed(region_op)

    def spawn_successor(self, operator_id: str) -> str | None:
        """The spawn region an operator feeds via a forward edge, if any."""
        return self._topology.spawn_successor(operator_id)

    def child_template_of(self, spawn_op: str) -> str | None:
        """The operator id of a spawn's child template, if it declares one."""
        return self._topology.child_template_of(spawn_op)

    def sealed_region_child_templates(self) -> frozenset[str]:
        """Child templates of agent-region spawns whose child-init sealed or revoked.

        A child template holds the workflow open until its spawn seals; a producer
        fanout retires it on materialization, and an agent's dynamic spawn region
        retires it once the region seals (on ``spawn_agent`` seal or the parent's
        completion), so the template — never dispatched as a task — stops holding it.
        """
        return self._ledger.sealed_region_child_templates()

    def spawn_awaits_children(self, spawn_op: str) -> bool:
        """Whether a spawn has yet to fan out: unopened, or open and not sealed.

        A failed spawn never fans out.
        """
        if self._failures.region_failed(spawn_op):
            return False
        scope_id = self._ledger.scope_id_for(spawn_op)
        if scope_id is None:
            return True
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        return cap.status is CapabilityStatus.OPEN

    def spawn_is_open(self, spawn_op: str) -> bool:
        """Whether a spawn's child-init capability still admits new children.

        False once the spawn has sealed or revoked, or before its child-init scope
        opens, so a re-driven fan-out over an already-closed spawn is a clean no-op. A
        read-only query: it never opens a scope.
        """
        scope_id = self._ledger.scope_id_for(spawn_op)
        if scope_id is None:
            return False
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        return cap.status is CapabilityStatus.OPEN

    def embodiment_menu(self, task_id: str) -> InferenceEmbodimentMenu | None:
        """The finite set of embodiments a task's plan node offers, if it offers one."""
        return self._ledger.embodiment_menu(task_id)

    def embodiment_selection(self, task_id: str) -> EmbodimentSelection | None:
        """The embodiment a task is already bound to, if one was resolved."""
        wi = self._ledger.work_item_for_task(task_id)
        return self._embodiment_selections.get(wi.work_item_id) if wi else None

    def embodiment_pinned(self, task_id: str) -> bool:
        """Whether a resolved embodiment is committed to the run that carries it.

        An embodiment changes only before its candidate-specific issue or delivery. A
        resident candidate commits at its invocation, after which reconciliation reuses
        that invocation and its idempotency and credit path rather than running the
        other embodiment; a local candidate carries no invocation and commits when its
        attempt is issued, which is where it was delivered to a worker.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.work_item_id not in self._embodiment_selections:
            return False
        return wi.invocation_id is not None or bool(wi.attempt_ids)

    def record_embodiment_selection(
        self, task_id: str, alternative_id: str, selector: str, evidence: str
    ) -> EmbodimentSelection | None:
        """Bind a task to one embodiment durably, before its worker message goes out.

        A pinned selection is kept: the caller receives the standing one rather than a
        replacement.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return None
        if (standing := self._embodiment_selections.get(wi.work_item_id)) is not None:
            if self.embodiment_pinned(task_id):
                return standing
        selection = EmbodimentSelection(
            work_item_id=wi.work_item_id,
            alternative_id=alternative_id,
            plan_version=self._topology.bundle.plan.plan_version.content_digest,
            selector=selector,
            evidence=evidence,
        )
        self._embodiment_selections[wi.work_item_id] = selection
        self._ledger.emit(
            "embodiment_selected",
            work_item_id=wi.work_item_id,
            detail={"alternative_id": alternative_id, "selector": selector},
        )
        return selection

    def input_resolution(self, task_id: str) -> InputResolution | None:
        """The resolution a task's inputs were materialized under, if one exists."""
        wi = self._ledger.work_item_for_task(task_id)
        return self._input_resolutions.get(wi.work_item_id) if wi else None

    def input_preparation(self, task_id: str) -> InputPreparation | None:
        """The preparation dispatch a task's inputs are being resolved by, if any."""
        wi = self._ledger.work_item_for_task(task_id)
        return self._input_preparations.get(wi.work_item_id) if wi else None

    def on_input_preparation_dispatched(
        self, task_id: str, worker_id: str | None
    ) -> None:
        """Record that a work item's inputs are being resolved on a worker.

        This deliberately mints neither an invocation nor an attempt: both are
        candidate-specific commitments, and a work item whose inputs are still being
        resolved has not chosen an embodiment to commit to.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return
        self._input_preparations[wi.work_item_id] = InputPreparation(
            work_item_id=wi.work_item_id, worker_id=worker_id
        )
        self._ledger.emit(
            "input_preparation_dispatched",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
        )

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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return None
        if (standing := self._input_resolutions.get(wi.work_item_id)) is not None:
            return standing
        resolution = InputResolution(
            work_item_id=wi.work_item_id, binding=binding, reference=reference
        )
        self._input_resolutions[wi.work_item_id] = resolution
        self._ledger.emit(
            "input_resolved",
            work_item_id=wi.work_item_id,
            detail={
                "request_digest": binding.request_digest,
                "cardinality": str(binding.cardinality),
            },
        )
        return resolution

    def episode_spec(self, task_id: str) -> EpisodeSpec | None:
        """The run-to-yield episode a task's operator lowers to, if the plan cut it."""
        return self._ledger.episode_spec(task_id)

    def work_item(self, task_id: str) -> WorkItem | None:
        return self._ledger.work_item(task_id)

    def child_input(self, task_id: str) -> ValueRef | None:
        """The child-init input a spawned child task runs on, if it has one."""
        wi = self._ledger.work_item_for_task(task_id)
        return wi.child_input if wi is not None else None

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
        return self._ledger.instance

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            instance=self._ledger.workflow_instance,
            root_scope=self._ledger.root_scope,
            root_grant=self._ledger.root_grant,
            scopes=list(self._ledger.scopes.values()),
            activations=list(self._ledger.activations.values()),
            work_items=list(self._ledger.work_items.values()),
            continuations=list(self._ledger.continuations.values()),
            records=list(self._ledger.records),
            accepted_inputs=list(self._accepted_inputs),
            region_aggregates=list(self._ledger.region_aggregates),
            invocations=list(self._ledger.invocations.values()),
            attempts=list(self._ledger.attempts.values()),
            embodiment_selections=list(self._embodiment_selections.values()),
            input_resolutions=list(self._input_resolutions.values()),
            input_preparations=list(self._input_preparations.values()),
            boundary_events=list(self._boundary_events.values()),
            effect_receipts=list(self._receipts.values()),
            authority_decisions=list(self._decisions),
            delegated_grants=list(self._grants.values()),
            progress_capabilities=list(self._ledger.capabilities.values()),
            result_slots=list(self._slots.values()),
            result_publications=list(self._publications.values()),
            trace=list(self._ledger.trace),
            private_state=self._ledger.private_state.lineages(),
            released_scopes=sorted(self._ledger.released_scopes),
            failed_regions=sorted(self._failures.failed_regions),
            failed_scopes=sorted(self._failures.failed_scopes),
            failure_reasons=dict(self._failures.failure_reasons),
            next_seq=self._ledger.next_seq,
        )

    def reconcile_failure(self, task_id: str) -> list[str]:
        """Fail what a task's settled failure left standing downstream of it.

        Returns the legacy task ids newly failed; empty for a task that has not failed,
        for a spawned child, whose failure drains its scope instead, and once the
        downstream has already failed.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if (
            wi is None
            or wi.outcome is not PublicationOutcome.DECLARED_FAILURE
            or self._ledger.is_dynamic_activation(wi.activation_id)
        ):
            return []
        cascade = Advance()
        visited: set[str] = set()
        self._fail_agent_regions(wi, cascade, visited)
        self._fail_downstream(wi.operator_id, cascade, visited)
        self._failures.name_failures(cascade.failed, dependency_failed(task_id))
        return cascade.failed

    def fail_undeliverable_region_inputs(self) -> list[str]:
        """Fail each task reading a region of an agent that runs only as a spawned
        child, and everything downstream of it.

        Every scope of such a region is nested, so its join never delivers at the root
        and the reader would wait forever. Returns the legacy task ids newly failed.
        """
        cascade = Advance()
        visited: set[str] = set()
        owners = spawned_only_region_owners(self._topology.operators.values())
        for spawn_op, owner in sorted(owners.items()):
            join_op = self._topology.join_for_spawn(spawn_op)
            for successor in sorted(self._topology.forward.get(join_op or "", ())):
                if (wi_id := self._ledger.wi_by_operator.get(successor)) is None:
                    continue
                start = len(cascade.failed)
                self._fail_work_item(wi_id, cascade, visited)
                if failed := cascade.failed[start:]:
                    self._failures.name_failures(
                        failed[:1],
                        f"region of spawned-only agent {owner} delivers nothing",
                    )
                    self._failures.name_failures(
                        failed[1:], dependency_failed(failed[0])
                    )
        return cascade.failed

    def reconcile_pending(self, task_id: str) -> bool:
        """Re-derive readiness for a task whose durable record shows PENDING.

        Returns whether the work item is ready to admit. A work item the snapshot still
        shows in flight — a crash after a retry persisted the PENDING record but before
        the ledger caught up — is reset to ready with its lost attempt marked, so the
        retry is not orphaned; a work item whose predecessors have not all settled, or
        whose declared inputs have not all been accepted, stays blocked.
        """
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status in TERMINAL_WORK_ITEM_STATUSES:
            return False
        cont = self._ledger.continuations.get(wi.work_item_id)
        if cont is not None and (
            cont.waiting_on
            or not cont.required_ports
            <= {a.target_port for a in self.accepted_inputs_for(wi.activation_id)}
        ):
            wi.status = WorkItemStatus.BLOCKED
            return False
        if wi.status is WorkItemStatus.BLOCKED and self._awaits_mediated_outcome(wi):
            # A crash beat the ledger save of the boundary's settle; the boundary is
            # re-issued, and the episode resumes only with its outcome.
            return False
        if wi.status is WorkItemStatus.DISPATCHED:
            if attempt := self._ledger.latest_attempt(wi):
                attempt.status = AttemptStatus.LOST
                attempt.finished_at = now_iso()
                self._ledger.emitter.emit_attempt(attempt)
            self._ledger.emit(
                "attempt_lost_on_restart",
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
            )
        wi.status = WorkItemStatus.READY
        return True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _require_child_init_scope(self, handle: str) -> str:
        if (scope_id := self._ledger.scope_id_for(handle)) is not None:
            return scope_id
        opener = self._ledger.resolve_opener_activation(handle)
        if opener is None:
            raise RegionError(f"{handle!r} has no opener activation")
        # A lazily opened scope (an agent's first spawn) nests under the opener's own
        # enclosing scope, so a nested agent's recursion depth is counted correctly.
        parent = (
            self._ledger.activations[opener].scope_id
            if opener in self._ledger.activations
            else None
        )
        return self._open_child_init_scope(opener, parent_scope_id=parent)

    def _require_loop_scope(self, handle: str) -> str:
        if (scope_id := self._ledger.scope_id_for(handle)) is not None:
            return scope_id
        opener = self._ledger.resolve_opener_activation(handle)
        if opener is None:
            raise RegionError(f"{handle!r} has no opener activation")
        return self._open_loop(opener)

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
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None:
            return False
        attempt = self._ledger.latest_attempt(wi)
        return attempt is not None and attempt.status in (
            AttemptStatus.ISSUED,
            AttemptStatus.RUNNING,
        )
