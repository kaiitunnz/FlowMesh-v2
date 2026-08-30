"""The orchestration engine over the transparent structured-region physical plan.

The engine owns semantic readiness: it turns settled records into ready
work items, incrementally materializing the activation graph rather than precreating
attempts. Static top-level leaf and agent operators materialize eagerly and dispatch
through the runtime; control operators (branch, merge, spawn, join, loop) settle inside
the ledger and never dispatch; spawn children and loop iterations materialize as records
flow. An agent is both dispatchable (a run-to-yield episode) and a scope owner: it
yields validated boundary requests the engine turns into durable work, and its
spawn_agent facade opens a child-init scope per selected child region, keyed by
(agent activation, region), lazily on that region's first child. Progress
capabilities (child-init and loop-time) are the closure authority — a
region closes only when its combined account seals and drains, never on an observed
empty set. Scheduler/worker placement stays a physical decision that never changes what
the engine considers ready.
"""

from dataclasses import dataclass, field
from typing import Self

from shared.harness import DeliveredOutcome, OutcomeKind
from shared.utils import (
    new_activation_id,
    new_attempt_id,
    new_authority_grant_id,
    new_idempotency_key,
    new_invocation_id,
    new_scope_id,
    new_work_item_id,
)

from ..task.v2.representations.bundle import PersistedV2Workflow
from ..task.v2.representations.operators import (
    AgentOperator,
    BoundaryEventKind,
    BranchRegion,
    EffectClass,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LeafProfile,
    LogicalOperator,
    OperatorKind,
    RecoveryClass,
    ResidualPolicy,
    SpawnRegion,
)
from ..task.v2.representations.plan import EpisodeSpec
from ..task.v2.representations.results import CardinalityKind
from ..utils.time import now_iso
from .guardrails import ScopeBudget
from .outcomes import (
    attenuate,
    check_admissible,
    classify_recovery,
    is_compensable,
    is_replayable,
    next_on_acknowledge,
    next_on_reissue,
    next_on_terminal,
    next_on_uncertain,
)
from .state import (
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
    Invocation,
    InvocationState,
    LedgerSnapshot,
    OrchestrationEvent,
    ProgressAxis,
    ProgressCapability,
    PublicationOutcome,
    Record,
    RecoveryDisposition,
    ResultPublication,
    ResultSlot,
    Scope,
    ValueRef,
    WorkflowInstance,
    WorkItem,
    WorkItemStatus,
)

_EVENT_FIELDS = frozenset(
    {"operator_id", "work_item_id", "attempt_id", "invocation_id", "slot_key"}
)
_REGION_KINDS = frozenset(
    {
        OperatorKind.BRANCH,
        OperatorKind.MERGE,
        OperatorKind.SPAWN,
        OperatorKind.JOIN,
        OperatorKind.LOOP_CONTEXT,
    }
)
# Control operators settle in-ledger and never dispatch. An agent is not control: it is
# a dispatchable run-to-yield episode that also owns a child-init scope for its
# spawn_agent children.
_CONTROL_KINDS = _REGION_KINDS
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
_EARLY_JOINS = frozenset(
    {JoinCompletion.ANY, JoinCompletion.FIRST_K, JoinCompletion.PREDICATE}
)
_TERMINAL_WI = frozenset({WorkItemStatus.SETTLED, WorkItemStatus.CANCELLED})


class RegionError(ValueError):
    """Raised when a structured-region operation is invalid, e.g. a child after seal."""


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


@dataclass
class Advance:
    """Runtime-visible effect of an engine transition, in legacy task ids.

    ``ready`` work items become admissible for a new attempt, ``failed`` ones settle
    terminally and cascade, and ``retry`` reissues an existing work item as a fresh
    attempt under its stable identity. Control settlement and dynamic child
    materialization are internal and never appear here.
    """

    ready: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    retry: list[str] = field(default_factory=list)

    def extend(self, other: "Advance") -> Self:
        self.ready.extend(other.ready)
        self.failed.extend(other.failed)
        self.retry.extend(other.retry)
        return self


class OrchestrationEngine:
    """Drives one workflow instance's semantic readiness over its durable ledger."""

    def __init__(
        self,
        snapshot: LedgerSnapshot,
        bundle: PersistedV2Workflow,
        *,
        budget: ScopeBudget | None = None,
    ) -> None:
        self._instance = snapshot.instance
        self._root_scope = snapshot.root_scope
        self._root_grant = snapshot.root_grant
        self._bundle = bundle
        self._budget = budget or ScopeBudget()
        self._next_seq = snapshot.next_seq
        self._initial = Advance()

        self._scopes = {s.scope_id: s for s in snapshot.scopes}
        self._scopes.setdefault(self._root_scope.scope_id, self._root_scope)
        self._activations = {a.activation_id: a for a in snapshot.activations}
        self._work_items = {w.work_item_id: w for w in snapshot.work_items}
        self._continuations = {c.work_item_id: c for c in snapshot.continuations}
        self._records = list(snapshot.records)
        self._invocations = {i.invocation_id: i for i in snapshot.invocations}
        self._attempts = {a.attempt_id: a for a in snapshot.attempts}
        self._receipts = {r.invocation_id: r for r in snapshot.effect_receipts}
        self._decisions = list(snapshot.authority_decisions)
        self._grants = {g.grant_id: g for g in snapshot.delegated_grants}
        self._capabilities = {
            (c.scope_id, c.axis): c for c in snapshot.progress_capabilities
        }
        self._slots = {s.slot_key: s for s in snapshot.result_slots}
        self._publications = {p.slot_key: p for p in snapshot.result_publications}
        self._trace = list(snapshot.trace)

        self._operators: dict[str, LogicalOperator] = {
            op.operator_id: op for op in bundle.template.operators
        }
        self._profiles: dict[str, LeafProfile] = {
            op.operator_id: op.profile
            for op in bundle.template.operators
            if isinstance(op, LeafOperator)
        }
        self._replay = {
            b.source_ref: b.replay_contract
            for b in bundle.template.effect_boundaries
            if b.source_ref
        }
        self._child_templates = {
            op.child_template_ref
            for op in bundle.template.operators
            if isinstance(op, SpawnRegion) and op.child_template_ref
        }
        # Spawn regions an agent selects by role: entered through the agent-request
        # path, not auto-fired at the root like a producer-driven region.
        self._agent_region_spawns = {
            ref.spawn_ref
            for op in bundle.template.operators
            if isinstance(op, AgentOperator)
            for ref in op.child_region_refs
        }
        # (agent activation, region operator) -> the synthetic opener activation that
        # owns that region's child-init scope, rebuilt from the persisted openers.
        self._region_openers: dict[tuple[str, str], str] = {
            (a.parent_activation_id, a.operator_id): a.activation_id
            for a in self._activations.values()
            if a.kind == "region" and a.parent_activation_id
        }
        # Mediated boundaries, keyed by (activation, adapter-local call correlation):
        # the correlation rule that maps a re-driven facade call to its recorded key.
        self._boundary_events = {
            (b.activation, b.call_correlation): b
            for b in snapshot.boundary_events
            if b.activation and b.call_correlation
        }
        self._wi_by_task = {
            w.legacy_task_id: w.work_item_id
            for w in self._work_items.values()
            if w.legacy_task_id
        }
        # The operator index resolves a static leaf's forward-record successor; a
        # dispatched child or iteration shares its body operator across instances, so it
        # is addressed by task or activation, never by operator.
        self._wi_by_operator = {
            w.operator_id: w.work_item_id
            for w in self._work_items.values()
            if w.legacy_task_id and not self._is_dynamic_activation(w.activation_id)
        }
        self._wi_by_activation = {
            w.activation_id: w.work_item_id for w in self._work_items.values()
        }
        self._slots_by_operator: dict[str, list[str]] = {}
        for slot in self._slots.values():
            self._slots_by_operator.setdefault(slot.source_operator_id, []).append(
                slot.slot_key
            )

        self._forward = self._build_topology()
        # Scope ownership is keyed on the opener activation, so one operator can own a
        # scope per recursion level; an operator handle resolves through the index.
        self._scope_by_activation = {
            s.owner_activation_id: s.scope_id
            for s in self._scopes.values()
            if s.owner_activation_id
        }
        self._owner_acts_by_operator: dict[str, list[str]] = {}
        for s in self._scopes.values():
            if s.owner_operator_id and s.owner_activation_id:
                self._owner_acts_by_operator.setdefault(s.owner_operator_id, []).append(
                    s.owner_activation_id
                )
        self._loop_time: dict[str, int] = {}
        for scope in self._scopes.values():
            owner = scope.owner_operator_id
            if owner and self._kind(owner) is OperatorKind.LOOP_CONTEXT:
                self._loop_time[scope.scope_id] = max(
                    (
                        a.loop_time
                        for a in self._activations.values()
                        if a.scope_id == scope.scope_id
                    ),
                    default=0,
                )
        # Released scopes are authoritative scope-level state, restored directly rather
        # than re-derived from records: a recursive region's levels share one join/loop
        # operator, so a record could not attribute a release to the right level.
        self._released_scopes: set[str] = set(snapshot.released_scopes)
        self._denied_spawns = {
            d.operator_id
            for d in self._decisions
            if d.kind is AuthorityDecisionKind.DENIED and d.operator_id
        }

    def _build_topology(self) -> dict[str, list[str]]:
        """Forward successor edges, excluding feedback and spawn->join binding edges."""
        forward: dict[str, list[str]] = {op: [] for op in self._operators}
        for edge in self._bundle.template.edges:
            if edge.feedback or edge.from_op not in forward:
                continue
            if self._is_spawn_join_edge(edge.from_op, edge.to_op):
                continue
            forward[edge.from_op].append(edge.to_op)
        return forward

    def _is_spawn_join_edge(self, from_op: str, to_op: str) -> bool:
        return (
            self._kind(from_op) is OperatorKind.SPAWN
            and self._kind(to_op) is OperatorKind.JOIN
        )

    def _kind(self, operator_id: str) -> OperatorKind | None:
        op = self._operators.get(operator_id)
        return op.kind if op else None

    def _is_control(self, operator_id: str) -> bool:
        return self._kind(operator_id) in _CONTROL_KINDS

    def _is_dynamic_activation(self, activation_id: str) -> bool:
        act = self._activations.get(activation_id)
        return act is not None and act.kind in ("child", "iteration")

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
            continuations.append(
                Continuation(
                    work_item_id=work_item.work_item_id,
                    waiting_on=set(preds[op.operator_id]),
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
        engine = cls(snapshot, bundle, budget=budget)
        engine._initial = engine._open_roots()
        return engine

    def initial_advance(self) -> Advance:
        """Ready/failed roots admitted at submission time."""
        return self._initial

    # ------------------------------------------------------------------ #
    # Physical attempt lifecycle (dispatchable leaves)
    # ------------------------------------------------------------------ #

    def on_dispatched(self, task_id: str, worker_id: str | None) -> None:
        """Record a physical attempt and issue or reissue the work item's invocation."""
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status in _TERMINAL_WI:
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
            self._invocations[invocation.invocation_id] = invocation
        else:
            invocation = self._invocations[wi.invocation_id]
            invocation.state = next_on_reissue(invocation.state)
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
            attempt_no=len(wi.attempt_ids) + 1,
            worker_id=worker_id,
            started_at=now_iso(),
        )
        wi.attempt_ids.append(attempt.attempt_id)
        self._attempts[attempt.attempt_id] = attempt
        wi.status = WorkItemStatus.DISPATCHED
        self._emit(
            "attempt_issued",
            work_item_id=wi.work_item_id,
            attempt_id=attempt.attempt_id,
            invocation_id=wi.invocation_id or "",
            operator_id=wi.operator_id,
        )

    def on_started(self, task_id: str) -> None:
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.invocation_id is None:
            return
        if attempt := self._latest_attempt(wi):
            attempt.status = AttemptStatus.RUNNING
        invocation = self._invocations[wi.invocation_id]
        invocation.state = next_on_acknowledge(invocation.state)
        self._emit(
            "invocation_acknowledged",
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )

    def on_succeeded(self, task_id: str, *, empty: bool = False) -> Advance:
        """Settle a work item on success and release its successors.

        ``empty`` marks a conditional-skip settlement, resolving the declared output to
        an explicit-empty publication rather than a value.
        """
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status in _TERMINAL_WI:
            return Advance()
        outcome = (
            PublicationOutcome.EXPLICIT_EMPTY if empty else PublicationOutcome.SUCCESS
        )
        value_ref = (
            ValueRef(kind="empty")
            if empty
            else ValueRef(kind="legacy_task_result", legacy_task_id=wi.legacy_task_id)
        )
        self._settle_attempt_terminal(wi, outcome)
        activation = self._activations[wi.activation_id]
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
        self._publish(wi.operator_id, outcome, value_ref)
        return self._deliver_record(wi.operator_id, wi.activation_id, value_ref).extend(
            released
        )

    def _settle_attempt_terminal(
        self, wi: WorkItem, outcome: PublicationOutcome
    ) -> None:
        if attempt := self._latest_attempt(wi):
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.finished_at = now_iso()
        if wi.invocation_id is not None:
            invocation = self._invocations[wi.invocation_id]
            invocation.state = next_on_terminal(invocation.state)
            self._record_receipt(wi, outcome)

    def on_failed(self, task_id: str, error: str, *, retryable: bool) -> Advance:
        """Retry a work item as a fresh attempt, or settle it and cascade failure."""
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status in _TERMINAL_WI:
            return Advance()
        if attempt := self._latest_attempt(wi):
            attempt.status = AttemptStatus.FAILED
            attempt.finished_at = now_iso()
            attempt.error = error
        if retryable:
            wi.status = WorkItemStatus.READY
            self._emit(
                "attempt_retry",
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
            )
            return Advance(retry=[wi.legacy_task_id])
        activation = self._activations[wi.activation_id]
        released = self._agent_terminal_regions(wi.operator_id, wi.activation_id)
        if activation.kind == "child":
            # A child's terminal failure drains its scope account and lets an
            # all-succeed join fail; it does not cascade over static successors.
            advance = self._settle_child_wi(
                wi, activation, PublicationOutcome.DECLARED_FAILURE, None
            )
            advance.failed.append(wi.legacy_task_id)
            return advance.extend(released)
        return Advance(failed=self._settle_failure(wi.work_item_id)).extend(released)

    def on_uncertain(self, task_id: str) -> Advance:
        """Resolve a lost acknowledgement or route loss for an in-flight work item."""
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status in _TERMINAL_WI or wi.invocation_id is None:
            return Advance()
        invocation = self._invocations[wi.invocation_id]
        invocation.state = next_on_uncertain(
            invocation.state,
            replayable=invocation.replayable,
            compensable=invocation.compensable,
        )
        if attempt := self._latest_attempt(wi):
            attempt.status = AttemptStatus.LOST
            attempt.finished_at = now_iso()
        if invocation.replayable:
            wi.status = WorkItemStatus.READY
            self._emit(
                "invocation_uncertain_retry",
                work_item_id=wi.work_item_id,
                invocation_id=wi.invocation_id,
            )
            return Advance(retry=[wi.legacy_task_id])
        self._emit(
            (
                "invocation_compensation_required"
                if invocation.compensable
                else "invocation_ambiguity_terminal"
            ),
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )
        released = self._agent_terminal_regions(wi.operator_id, wi.activation_id)
        return Advance(failed=self._settle_failure(wi.work_item_id)).extend(released)

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
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status in _TERMINAL_WI:
            return Advance()
        # A recorded call is a re-drive whether it was granted or denied: it maps to its
        # key and creates no new work, so re-validation and duplicate records are cut.
        if self._is_boundary_redrive(wi, event):
            return Advance()
        op = self._operators.get(wi.operator_id)
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
                # region selects the entry body; a raw child_ref never names a body.
                advance = self.materialize_child(opener, value_ref=event.value_ref)
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
                self._emit(
                    "state_access",
                    work_item_id=wi.work_item_id,
                    operator_id=wi.operator_id,
                    detail={"state_ref": event.state_ref or ""},
                )
                return Advance()
        return Advance()

    def deliver_boundary_outcome(self, task_id: str, call_correlation: str) -> Advance:
        """Re-ready a boundary-suspended work item once its outcome is durable.

        A mediated request's durable outcome — a model or tool result, or a denial —
        lets the episode resume: the work item returns to READY for a fresh attempt that
        injects the outcome at its originating call. Only a boundary-suspended work item
        with a recorded envelope for the call is resumed — a delivery to a running or
        settled item, or for an unrecorded call, is a no-op. An episode has one
        outstanding boundary at a time, so the recorded call is the one it awaits.
        """
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.BLOCKED:
            return Advance()
        if (wi.activation_id, call_correlation) not in self._boundary_events:
            return Advance()
        wi.status = WorkItemStatus.READY
        self._emit(
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
        if (wi := self._work_item_for_task(task_id)) is not None:
            wi.pending_outcome_call = call_correlation

    def close_latest_attempt(self, task_id: str) -> None:
        """Settle a still-running attempt of a continuing episode, bounding its history.

        A continue-boundary re-dispatches without suspending, so its finished attempt is
        marked succeeded here rather than left perpetually running.
        """
        wi = self._work_item_for_task(task_id)
        if wi is not None and (attempt := self._latest_attempt(wi)) is not None:
            if attempt.status in (AttemptStatus.ISSUED, AttemptStatus.RUNNING):
                attempt.status = AttemptStatus.SUCCEEDED
                attempt.finished_at = now_iso()

    def stranded_model_settlements(self) -> list[tuple[str, str, str | None]]:
        """Model/effect boundaries suspended with no durable outcome, for a restart.

        A model or effect boundary suspends off-lane while the gateway settles it; a
        crash before that settle leaves the work item blocked with an issued
        invocation and no recorded outcome. Returns each such boundary's (task, call,
        payload) so the settle can be re-issued from the durable envelope.
        """
        stranded: list[tuple[str, str, str | None]] = []
        for (activation, corr), env in self._boundary_events.items():
            if env.kind not in (
                BoundaryEventKind.INVOCATION,
                BoundaryEventKind.EXTERNAL_EFFECT,
            ):
                continue
            if env.denial is not None or env.outcome_value is not None:
                continue
            wi = self._wi_by_activation.get(activation)
            work_item = self._work_items.get(wi) if wi else None
            if work_item is None or work_item.status is not WorkItemStatus.BLOCKED:
                continue
            stranded.append((work_item.legacy_task_id, corr, env.request_payload))
        return stranded

    def settle_boundary_outcome(
        self, task_id: str, call_correlation: str, *, value: str | None = None
    ) -> Advance:
        """Persist a mediated outcome's value and re-ready the suspended episode.

        The value lands durably on the boundary envelope so a re-dispatch injects it and
        a restart rehydrates it; the item then returns to READY for a fresh attempt.
        """
        wi = self._work_item_for_task(task_id)
        if wi is None:
            return Advance()
        corr = (wi.activation_id, call_correlation)
        if (env := self._boundary_events.get(corr)) is not None and value is not None:
            self._boundary_events[corr] = env.model_copy(
                update={"outcome_value": value}
            )
        self.mark_pending_outcome(task_id, call_correlation)
        return self.deliver_boundary_outcome(task_id, call_correlation)

    def episode_context(
        self, task_id: str
    ) -> tuple[str | None, tuple[DeliveredOutcome, ...]]:
        """The durable capsule and pending injected outcome for an agent's next step.

        Rebuilt from the ledger, never in-memory episode state: the capsule is the work
        item's continuation, and the one pending outcome is reconstructed from its
        settled boundary envelope, so a re-dispatch after a restart carries it again.
        """
        wi = self._work_item_for_task(task_id)
        if wi is None:
            return None, ()
        outcomes: tuple[DeliveredOutcome, ...] = ()
        if wi.pending_outcome_call is not None:
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
            )
        return DeliveredOutcome(
            call_correlation=corr,
            idempotency_key=env.idempotency_key,
            kind=OutcomeKind.RESULT,
            value=env.outcome_value,
        )

    def _is_boundary_redrive(self, wi: WorkItem, event: BoundaryEvent) -> bool:
        """Whether a boundary reissues a recorded facade call under its stable id."""
        if event.call_correlation is None:
            return False
        if (wi.activation_id, event.call_correlation) not in self._boundary_events:
            return False
        self._emit(
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
        one.
        None means admissible.
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
            if self._agent_region_op(op, event.child_region_ref) is None:
                return DenialKind.AUTHORITY
        return None

    def _agent_region_op(self, op: AgentOperator, role: str | None) -> str | None:
        """The spawn region operator a declared role selects, or None if undeclared."""
        if role is None:
            return None
        return next(
            (ref.spawn_ref for ref in op.child_region_refs if ref.name == role), None
        )

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
        region_op = self._agent_region_op(op, event.child_region_ref)
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
        if (opener := self._region_openers.get(key)) is not None:
            return opener
        agent = self._activations[agent_activation]
        agent_op = self._operators[agent.operator_id]
        assert isinstance(agent_op, AgentOperator)
        _, agent_delegate = self._agent_face_tuples(agent_op, agent.scope_id)
        opener_act = Activation(
            activation_id=new_activation_id(),
            instance_id=self._instance.instance_id,
            scope_id=agent.scope_id,
            operator_id=region_op,
            kind="region",
            parent_activation_id=agent_activation,
        )
        self._activations[opener_act.activation_id] = opener_act
        self._region_openers[key] = opener_act.activation_id
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
            op, self._activations[wi.activation_id].scope_id
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
        self._emit(
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
                grant_id=self._root_grant.grant_id,
                interface=subject,
                kind=AuthorityDecisionKind.DENIED,
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
                denial_kind=denial,
                reason=f"boundary {event.kind.value} denied",
            )
        )
        self._record_boundary(wi, event, denial=denial)
        self._emit(
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
        self._invocations[invocation.invocation_id] = invocation
        self._record_boundary(wi, event, invocation_id=invocation.invocation_id)
        self._emit(
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
        if attempt := self._latest_attempt(wi):
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.finished_at = now_iso()
        wi.status = WorkItemStatus.BLOCKED
        self._emit(kind, work_item_id=wi.work_item_id, operator_id=wi.operator_id)

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
        nested scope owned by the child activation, so grandchildren (and recursive
        re-entries) attenuate from the child grant at ``depth+1``. Returns the child
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
        attempt. The child body must be a leaf, or an agent that itself owns a
        child-init scope; a spawn/loop child body is not live-dispatchable and stays a
        trace-level :meth:`spawn_child`.
        """
        advance = Advance()
        _, wi = self._create_child(
            spawn, operator_id, dispatchable=True, value_ref=value_ref
        )
        self._admit(wi.work_item_id, advance)
        return advance

    def _create_child(
        self,
        spawn: str,
        operator_id: str | None,
        *,
        dispatchable: bool,
        value_ref: ValueRef | None,
    ) -> tuple[Activation, WorkItem]:
        spawn_op = self._handle_operator(spawn)
        spawn_op_obj = self._operators[spawn_op]
        child_ref = (
            spawn_op_obj.child_template_ref
            if isinstance(spawn_op_obj, (SpawnRegion, AgentOperator))
            else None
        )
        body_ref = operator_id or child_ref or spawn_op
        body_op = self._operators.get(body_ref)
        if spawn_op in self._denied_spawns:
            self._emit(
                "child_rejected", operator_id=spawn_op, detail={"reason": "denied"}
            )
            raise RegionError(f"spawn {spawn_op!r} was denied; no child may be created")
        scope_id = self._require_child_init_scope(spawn)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is not CapabilityStatus.OPEN:
            self._emit(
                "child_rejected",
                operator_id=spawn_op,
                detail={"reason": cap.status.value},
            )
            raise RegionError(
                f"spawn {spawn_op!r} child-init capability is {cap.status.value}; "
                "no child may be created"
            )
        body_opens_scope = self._kind(body_ref) in _CHILD_INIT_OPENERS or (
            self._kind(body_ref) is OperatorKind.LOOP_CONTEXT
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
        index = sum(1 for a in self._activations.values() if a.scope_id == scope_id)
        activation = Activation(
            activation_id=new_activation_id(),
            instance_id=self._instance.instance_id,
            scope_id=scope_id,
            operator_id=body_ref,
            kind="child",
            child_index=index,
        )
        self._activations[activation.activation_id] = activation
        effect, recovery = _effect_recovery(body_op)
        child_wi = WorkItem(
            work_item_id=new_work_item_id(),
            activation_id=activation.activation_id,
            operator_id=body_ref,
            legacy_task_id=activation.activation_id if dispatchable else "",
            value_ref=value_ref,
            effect_class=effect,
            recovery=recovery,
            replay_contract=self._replay.get(body_ref),
        )
        self._work_items[child_wi.work_item_id] = child_wi
        self._wi_by_activation[activation.activation_id] = child_wi.work_item_id
        if dispatchable:
            self._wi_by_task[child_wi.legacy_task_id] = child_wi.work_item_id
        cap.outstanding += 1
        self._emit(
            "child_spawned",
            operator_id=body_ref,
            detail={"scope": scope_id, "index": str(index)},
        )
        # A spawn/loop child body opens its nested scope eagerly; a dispatchable agent
        # child opens its own child-init scope lazily, only when it first spawns.
        if self._kind(body_ref) is OperatorKind.SPAWN:
            self._open_child_init_scope(
                activation.activation_id, parent_scope_id=scope_id
            )
        elif self._kind(body_ref) is OperatorKind.LOOP_CONTEXT:
            self._open_loop(activation.activation_id, parent_scope_id=scope_id)
        return activation, child_wi

    def seal_spawn(self, spawn: str) -> Advance:
        """Seal a spawn's child-init capability; no further children may be created."""
        scope_id = self._require_child_init_scope(spawn)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.SEALED
            self._emit(
                "child_init_sealed",
                operator_id=self._scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )
        return self._maybe_release_join(scope_id)

    def _settle_agent_regions(self, activation_id: str) -> Advance:
        """Settle every declared child region of a terminal agent under its residual.

        Each entered region seals its open child-init capability so its join releases on
        drain; a declared-but-never-entered region opens as a zero-child region and
        seals, so its join releases empty. A cancel residual revokes and cancels the
        region's children instead. Opening an unused region is skipped when it would
        exceed the scope-depth budget, since the agent could never have entered it. A
        non-spawning agent owns no region.
        """
        advance = Advance()
        agent = self._activations.get(activation_id)
        op = self._operators.get(agent.operator_id) if agent else None
        if not isinstance(op, AgentOperator) or agent is None:
            return advance
        room = self._scopes[agent.scope_id].depth + 1 <= self._budget.max_scope_depth
        for ref in op.child_region_refs:
            opener = self._region_openers.get((activation_id, ref.spawn_ref))
            if opener is None:
                if not room:
                    continue
                opener = self._region_opener(activation_id, ref.spawn_ref)
            advance.extend(self._settle_owned_region(opener))
        return advance

    def _agent_terminal_regions(self, operator_id: str, activation_id: str) -> Advance:
        """Settle an agent's declared regions when the agent terminates, else no-op."""
        if isinstance(self._operators.get(operator_id), AgentOperator):
            return self._settle_agent_regions(activation_id)
        return Advance()

    def _settle_owned_region(self, opener: str) -> Advance:
        scope_id = self._scope_by_activation.get(opener)
        if scope_id is None:
            return Advance()
        cap = self._capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if cap is None or cap.status is not CapabilityStatus.OPEN:
            return Advance()
        owner = self._scopes[scope_id].owner_operator_id
        join = self._join_of_scope(scope_id)
        if self._residual_policy(join, ResidualPolicy.DRAIN) is ResidualPolicy.CANCEL:
            cap.status = CapabilityStatus.REVOKED
            self._emit(
                "child_init_revoked",
                operator_id=owner,
                detail={"scope": scope_id, "reason": "agent_terminal"},
            )
            self._cancel_residual_children(scope_id)
        else:
            cap.status = CapabilityStatus.SEALED
            self._emit(
                "child_init_sealed",
                operator_id=owner,
                detail={"scope": scope_id, "reason": "agent_terminal"},
            )
        return self._maybe_release_join(scope_id)

    def revoke_spawn(self, spawn: str) -> None:
        """Revoke a spawn's child-init capability as a progress transition.

        Distinct from sealing: revocation withdraws the capability, while sealing marks
        a producer done. Both close the child-init axis once outstanding children drain.
        """
        scope_id = self._require_child_init_scope(spawn)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.REVOKED
            self._emit(
                "child_init_revoked",
                operator_id=self._scopes[scope_id].owner_operator_id,
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
        activation = self._activations.get(child_activation_id)
        if activation is None or activation.kind != "child":
            raise RegionError(f"unknown child activation {child_activation_id!r}")
        wi = self._work_items[self._wi_by_activation[child_activation_id]]
        return self._settle_child_wi(wi, activation, outcome, value_ref)

    def _settle_child_wi(
        self,
        wi: WorkItem,
        activation: Activation,
        outcome: PublicationOutcome,
        value_ref: ValueRef | None,
    ) -> Advance:
        if wi.status in _TERMINAL_WI:
            return Advance()
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = outcome
        wi.value_ref = value_ref
        cap = self._capability(activation.scope_id, ProgressAxis.CHILD_INIT)
        cap.outstanding = max(0, cap.outstanding - 1)
        spawn_op = self._scopes[activation.scope_id].owner_operator_id or ""
        self._publish_keyed(spawn_op, activation, outcome, value_ref)
        self._emit(
            "child_settled",
            operator_id=activation.operator_id,
            detail={"scope": activation.scope_id, "outcome": outcome.value},
        )
        return self._maybe_release_join(activation.scope_id)

    def route_branch(self, branch_op: str, selected_port: str) -> Advance:
        """Route a branch record to the selected port; settle the other ports empty."""
        if self._kind(branch_op) is not OperatorKind.BRANCH:
            raise RegionError(f"{branch_op!r} is not a branch region")
        advance = Advance()
        skipped: set[str] = set()
        for successor in sorted(self._forward.get(branch_op, ())):
            from_port = self._edge_from_port(branch_op, successor)
            if from_port in (selected_port, None):
                self._release_one(successor, branch_op, advance)
            else:
                self._settle_empty_successor(successor, skipped)
        self._emit(
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
        loop_op = self._scopes[scope_id].owner_operator_id or self._handle_operator(
            loop
        )
        cap = self._capability(scope_id, ProgressAxis.LOOP_TIME)
        if cap.status is not CapabilityStatus.OPEN:
            raise RegionError(
                f"loop {loop_op!r} is {cap.status.value}; no feedback may arrive"
            )
        next_time = self._loop_time.get(scope_id, 0) + 1
        if next_time > self._budget.max_loop_iterations:
            self._exhaust_budget("loop_iterations", self._budget.max_loop_iterations)
        self._charge_activation()
        self._loop_time[scope_id] = next_time
        cap.coordinate = next_time
        cap.outstanding += 1
        activation = Activation(
            activation_id=new_activation_id(),
            instance_id=self._instance.instance_id,
            scope_id=scope_id,
            operator_id=loop_op,
            kind="iteration",
            loop_time=next_time,
        )
        self._activations[activation.activation_id] = activation
        self._records.append(
            Record(
                operator_id=loop_op,
                activation_id=activation.activation_id,
                scope_id=scope_id,
                loop_time=next_time,
                value_ref=value_ref,
            )
        )
        self._emit(
            "loop_feedback", operator_id=loop_op, detail={"loop_time": str(next_time)}
        )
        return activation.activation_id

    def settle_iteration(self, iteration_activation_id: str) -> Advance:
        """Mark a loop iteration terminal and drain the loop-time capability."""
        activation = self._activations.get(iteration_activation_id)
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
            self._emit(
                "loop_sealed",
                operator_id=self._scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id},
            )
        return self._maybe_egress_loop(scope_id)

    def deny_spawn(
        self, spawn_op: str, interface: str, *, kind: DenialKind = DenialKind.AUTHORITY
    ) -> None:
        """Record a definitive dynamic authorization denial at a spawn site.

        A denial creates no child activation and no resident claim, and is separate
        from quota/rate/capacity/transport outcomes. It does not seal the child-init
        capability: grant denial and cardinality sealing stay distinct.
        """
        self._denied_spawns.add(spawn_op)
        scope_id = self._scope_id_for(spawn_op)
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
        self._emit(
            "policy_denied" if kind is DenialKind.POLICY else "authority_denied",
            operator_id=spawn_op,
        )

    def can_delegate(self, region_op: str, interface: str) -> bool:
        """Whether a child of ``region_op`` may itself delegate ``interface``."""
        scope_id = self._scope_id_for(region_op)
        return (
            scope_id is not None
            and interface in self._grant_for_scope(scope_id).delegate
        )

    # ------------------------------------------------------------------ #
    # Cancellation (a durable semantic event)
    # ------------------------------------------------------------------ #

    def cancel_instance(self) -> Advance:
        """Cancel the whole workflow instance: the root scope and every descendant."""
        return self.on_cancelled(self._root_scope.scope_id)

    def on_cancelled(self, scope_or_task: str) -> Advance:
        """Cancel a scope subtree as a durable, recorded-before-terminal event.

        ``scope_or_task`` resolves to a scope by scope id, opener activation, region
        handle, or a settled task's owning scope. Over that scope and each descendant,
        in order: record the cancellation; revoke the child-init (and loop-time)
        capability — a transition distinct from sealing; apply the residual-child policy
        to materialized children; transition the remaining in-flight work items to
        ``CANCELLED``; revoke the scope's authority grant — distinct from the child-init
        revoke; and resolve declared outputs to their cancellation / no-winner outcome.
        """
        scope_id = self._resolve_scope(scope_or_task)
        if scope_id is None:
            raise RegionError(f"{scope_or_task!r} resolves to no cancellable scope")
        advance = Advance()
        for sid in self._scope_subtree(scope_id):
            advance.extend(self._cancel_scope(sid))
        return advance

    def _cancel_scope(self, scope_id: str) -> Advance:
        scope = self._scopes[scope_id]
        self._emit("scope_cancelled", detail={"scope": scope_id})
        for axis in (ProgressAxis.CHILD_INIT, ProgressAxis.LOOP_TIME):
            cap = self._capabilities.get((scope_id, axis))
            if cap is not None and cap.status is CapabilityStatus.OPEN:
                cap.status = CapabilityStatus.REVOKED
                self._emit(
                    (
                        "child_init_revoked"
                        if axis is ProgressAxis.CHILD_INIT
                        else "loop_revoked"
                    ),
                    operator_id=scope.owner_operator_id,
                    detail={"scope": scope_id},
                )
        self._apply_cancellation_residual(scope_id)
        for wi in self._scope_work_items(scope_id, kinds=("leaf", "agent")):
            if wi.status not in _TERMINAL_WI:
                self._cancel_work_item(wi)
        if scope.grant_id and scope.grant_id in self._grants:
            grant = self._grants[scope.grant_id]
            if not grant.revoked:
                self._grants[scope.grant_id] = grant.model_copy(
                    update={"revoked": True}
                )
                self._emit(
                    "grant_revoked",
                    operator_id=scope.owner_operator_id,
                    detail={"scope": scope_id},
                )
        return self._resolve_cancelled_outputs(scope_id)

    def _apply_cancellation_residual(self, scope_id: str) -> None:
        """Apply a cancelled scope's join residual policy to its materialized children.

        A declared ``drain``/``continue`` leaves materialized children to settle; the
        default (``cancel``, and any scope without a declared policy) cancels every
        not-yet-settled child.
        """
        join = self._join_of_scope(scope_id)
        if self._residual_policy(join, ResidualPolicy.CANCEL) is ResidualPolicy.CANCEL:
            self._cancel_residual_children(scope_id)

    def _resolve_cancelled_outputs(self, scope_id: str) -> Advance:
        if scope_id in self._released_scopes:
            return Advance()
        owner_op = self._scopes[scope_id].owner_operator_id or ""
        join = self._join_of_scope(scope_id)
        if join is not None:
            outcome = self._no_winner_outcome(join)
            release_op = join.operator_id
        elif self._kind(owner_op) is OperatorKind.LOOP_CONTEXT:
            outcome = PublicationOutcome.EXPLICIT_EMPTY
            release_op = owner_op
        else:
            return Advance()
        self._released_scopes.add(scope_id)
        self._emit(
            "join_released" if join is not None else "loop_egress",
            operator_id=release_op,
            detail={"outcome": outcome.value, "cancelled": "true"},
        )
        self._frontier_closed(scope_id)
        self._publish(release_op, outcome, ValueRef(kind="empty"))
        return self._deliver_record(
            release_op, self._control_activation(release_op), ValueRef(kind="empty")
        )

    def _cancel_work_item(self, wi: WorkItem) -> None:
        # A cancelled in-flight external effect is not compensated here; compensation on
        # cancel rides with the deferred effect-commit machinery.
        if wi.status in _TERMINAL_WI:
            return
        wi.status = WorkItemStatus.CANCELLED
        self._publish(
            wi.operator_id, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
        )
        self._emit(
            "work_item_cancelled",
            work_item_id=wi.work_item_id,
            operator_id=wi.operator_id,
        )

    def _resolve_scope(self, handle: str) -> str | None:
        if handle in self._scopes:
            return handle
        if handle in self._scope_by_activation:
            return self._scope_by_activation[handle]
        if (wi_id := self._wi_by_task.get(handle)) is not None:
            act = self._activations.get(self._work_items[wi_id].activation_id)
            return act.scope_id if act else None
        return self._scope_id_for(handle)

    def _scope_subtree(self, root: str) -> list[str]:
        order = [root]
        seen = {root}
        cursor = 0
        while cursor < len(order):
            current = order[cursor]
            cursor += 1
            for scope in self._scopes.values():
                if scope.parent_scope_id == current and scope.scope_id not in seen:
                    seen.add(scope.scope_id)
                    order.append(scope.scope_id)
        return order

    def _scope_work_items(
        self, scope_id: str, *, kinds: tuple[str, ...]
    ) -> list[WorkItem]:
        return [
            wi
            for a in self._activations.values()
            if a.scope_id == scope_id
            and a.kind in kinds
            and (
                wi := self._work_items.get(
                    self._wi_by_activation.get(a.activation_id, "")
                )
            )
            is not None
        ]

    # ------------------------------------------------------------------ #
    # Record delivery, control firing, and closure
    # ------------------------------------------------------------------ #

    def _deliver_record(
        self, operator_id: str, activation_id: str, value_ref: ValueRef | None
    ) -> Advance:
        self._records.append(
            Record(
                operator_id=operator_id,
                activation_id=activation_id,
                scope_id=self._root_scope.scope_id,
                value_ref=value_ref,
            )
        )
        self._emit("record_delivered", operator_id=operator_id)
        advance = Advance()
        for successor in sorted(self._forward.get(operator_id, ())):
            self._release_one(successor, operator_id, advance)
        return advance

    def _release_one(self, successor: str, from_op: str, advance: Advance) -> None:
        """Deliver a record to one successor: fire a control op, or admit a leaf."""
        if self._is_control(successor):
            cont = self._continuations.get(_control_key(successor))
            if cont is None:
                return
            cont.waiting_on.discard(from_op)
            if not cont.waiting_on:
                self._fire_control(successor, advance)
            return
        wi_id = self._wi_by_operator.get(successor)
        cont = self._continuations.get(wi_id) if wi_id else None
        if cont is None:
            return
        cont.waiting_on.discard(from_op)
        if not cont.waiting_on:
            self._admit(cont.work_item_id, advance)

    def _fire_control(self, operator_id: str, advance: Advance) -> None:
        kind = self._kind(operator_id)
        if kind in _CHILD_INIT_OPENERS:
            self._open_child_init_scope(self._control_activation(operator_id))
        elif kind is OperatorKind.LOOP_CONTEXT:
            self._open_loop(self._control_activation(operator_id))
        elif kind is OperatorKind.MERGE:
            self._emit("merge_combined", operator_id=operator_id)
            advance.extend(
                self._deliver_record(
                    operator_id, self._control_activation(operator_id), None
                )
            )
        elif kind is OperatorKind.BRANCH:
            branch = self._operators[operator_id]
            selection = branch.selection if isinstance(branch, BranchRegion) else None
            if selection and any(
                self._edge_from_port(operator_id, s) == selection
                for s in self._forward.get(operator_id, ())
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
        if opener_activation in self._scope_by_activation:
            return self._scope_by_activation[opener_activation]
        scope = self._new_child_scope(
            opener_activation, parent_scope_id, parent_delegate=parent_delegate
        )
        self._register_scope_owner(scope)
        self._acquire_capability(scope.scope_id, ProgressAxis.CHILD_INIT)
        self._emit(
            "child_init_acquired",
            operator_id=scope.owner_operator_id,
            detail={"scope": scope.scope_id},
        )
        return scope.scope_id

    def _open_loop(
        self, opener_activation: str, *, parent_scope_id: str | None = None
    ) -> str:
        if opener_activation in self._scope_by_activation:
            return self._scope_by_activation[opener_activation]
        scope = self._new_child_scope(opener_activation, parent_scope_id)
        self._register_scope_owner(scope)
        self._loop_time[scope.scope_id] = 0
        self._acquire_capability(scope.scope_id, ProgressAxis.LOOP_TIME, coordinate=0)
        self._emit(
            "loop_ingress",
            operator_id=scope.owner_operator_id,
            detail={"scope": scope.scope_id},
        )
        return scope.scope_id

    def _maybe_release_join(self, scope_id: str) -> Advance:
        scope = self._scopes.get(scope_id)
        if scope is None or scope.owner_operator_id is None:
            return Advance()
        join_op = self._join_for_spawn(scope.owner_operator_id)
        if join_op is None or scope_id in self._released_scopes:
            return Advance()
        cap = self._capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
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
        join = self._operators[join_op]
        if not isinstance(join, JoinRegion) or join.completion not in _EARLY_JOINS:
            return None
        threshold, monotone = self._early_rule(join)
        if not monotone or len(self._qualifiers(scope_id)) < threshold:
            return None
        return self._release_join(join_op, scope_id)

    def _release_join(self, join_op: str, scope_id: str) -> Advance:
        self._released_scopes.add(scope_id)
        join = self._operators[join_op]
        assert isinstance(join, JoinRegion)
        outcome, value_ref = self._join_result(join, scope_id)
        children = self._materialized_children(scope_id)
        self._emit(
            "join_released",
            operator_id=join_op,
            detail={"outcome": outcome.value, "children": str(len(children))},
        )
        self._frontier_closed(scope_id)
        self._apply_residual_policy(join, scope_id)
        self._publish(join_op, outcome, value_ref)
        return self._deliver_record(
            join_op, self._control_activation(join_op), value_ref
        )

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
                winner = self._work_items[
                    self._wi_by_activation[qualifiers[0].activation_id]
                ]
                return PublicationOutcome.SUCCESS, (
                    winner.value_ref or ValueRef(kind="join_result")
                )
            return self._no_winner_outcome(join), ValueRef(kind="empty")
        outcomes = [
            self._work_items[self._wi_by_activation[c.activation_id]].outcome
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
            if self._work_items[self._wi_by_activation[child.activation_id]].outcome
            is PublicationOutcome.SUCCESS
        ]

    def _materialized_children(self, scope_id: str) -> list[Activation]:
        return sorted(
            (
                a
                for a in self._activations.values()
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
        join_op = self._join_for_spawn(self._scopes[scope_id].owner_operator_id or "")
        join = self._operators.get(join_op) if join_op else None
        return join if isinstance(join, JoinRegion) else None

    def _cancel_residual_children(self, scope_id: str) -> None:
        cap = self._capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        for child in self._materialized_children(scope_id):
            wi = self._work_items[self._wi_by_activation[child.activation_id]]
            if wi.status not in _TERMINAL_WI:
                self._cancel_work_item(wi)
                if cap is not None:
                    cap.outstanding = max(0, cap.outstanding - 1)

    def _apply_residual_policy(self, join: JoinRegion, scope_id: str) -> None:
        """Govern a released early join's not-yet-settled materialized children.

        ``continue`` leaves child-init open; ``drain`` seals it; ``cancel`` revokes it
        and cancels the residual children. Residual settlements stay ledger-visible but
        never become the join's implicit winner output.
        """
        if join.completion not in _EARLY_JOINS:
            return
        policy = self._residual_policy(join, ResidualPolicy.CONTINUE)
        cap = self._capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if policy is ResidualPolicy.CONTINUE or cap is None:
            return
        cancel = policy is ResidualPolicy.CANCEL
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.REVOKED if cancel else CapabilityStatus.SEALED
            self._emit(
                "child_init_revoked" if cancel else "child_init_sealed",
                operator_id=self._scopes[scope_id].owner_operator_id,
                detail={"scope": scope_id, "residual": policy.value},
            )
        if cancel:
            self._cancel_residual_children(scope_id)

    def _maybe_egress_loop(self, scope_id: str) -> Advance:
        if not scope_id or scope_id in self._released_scopes:
            return Advance()
        cap = self._capabilities.get((scope_id, ProgressAxis.LOOP_TIME))
        if cap is None or not cap.closed:
            return Advance()
        self._released_scopes.add(scope_id)
        self._frontier_closed(scope_id)
        loop_op = self._scopes[scope_id].owner_operator_id or ""
        self._emit("loop_egress", operator_id=loop_op, detail={"scope": scope_id})
        carried = self._latest_carried(scope_id)
        self._publish(loop_op, PublicationOutcome.SUCCESS, carried)
        return self._deliver_record(loop_op, self._control_activation(loop_op), carried)

    # ------------------------------------------------------------------ #
    # Progress capabilities and scopes
    # ------------------------------------------------------------------ #

    def _acquire_capability(
        self, scope_id: str, axis: ProgressAxis, *, coordinate: int | None = None
    ) -> ProgressCapability:
        cap = ProgressCapability(scope_id=scope_id, axis=axis, coordinate=coordinate)
        self._capabilities[(scope_id, axis)] = cap
        return cap

    def _capability(self, scope_id: str, axis: ProgressAxis) -> ProgressCapability:
        cap = self._capabilities.get((scope_id, axis))
        if cap is None:
            raise RegionError(f"scope {scope_id!r} holds no {axis.value} capability")
        return cap

    def _check_scope_depth(self, parent_scope_id: str) -> None:
        if self._scopes[parent_scope_id].depth + 1 > self._budget.max_scope_depth:
            self._exhaust_budget("scope_depth", self._budget.max_scope_depth)

    def _new_child_scope(
        self,
        opener_activation: str,
        parent_scope_id: str | None,
        *,
        parent_delegate: tuple[str, ...] | None = None,
    ) -> Scope:
        opener_op = self._activations[opener_activation].operator_id
        parent = self._scopes[parent_scope_id or self._root_scope.scope_id]
        self._check_scope_depth(parent.scope_id)
        grant = self._mint_delegated_grant(
            opener_op, parent.scope_id, parent_delegate=parent_delegate
        )
        scope = Scope(
            scope_id=new_scope_id(),
            instance_id=self._instance.instance_id,
            parent_scope_id=parent.scope_id,
            owner_operator_id=opener_op,
            owner_activation_id=opener_activation,
            grant_id=grant.grant_id,
            depth=parent.depth + 1,
        )
        self._scopes[scope.scope_id] = scope
        self._grants[grant.grant_id] = grant.model_copy(
            update={"scope_id": scope.scope_id}
        )
        return scope

    def _register_scope_owner(self, scope: Scope) -> None:
        if scope.owner_activation_id:
            self._scope_by_activation[scope.owner_activation_id] = scope.scope_id
        if scope.owner_operator_id and scope.owner_activation_id:
            self._owner_acts_by_operator.setdefault(scope.owner_operator_id, []).append(
                scope.owner_activation_id
            )

    def _frontier_closed(self, scope_id: str) -> None:
        self._emit("frontier_closed", detail={"scope": scope_id})

    def _charge_activation(self) -> None:
        dynamic = sum(
            1 for a in self._activations.values() if a.kind in ("child", "iteration")
        )
        if dynamic >= self._budget.max_activations:
            self._exhaust_budget("activations", self._budget.max_activations)

    def _exhaust_budget(self, budget: str, limit: int) -> None:
        """Record a durable scope-budget breach, distinct from an authority denial."""
        self._emit(
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
        opener = self._operators[opener_op]
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
            instance_id=self._instance.instance_id,
            scope_id="",
            parent_grant_id=parent.grant_id,
            policy_id=parent.policy_id,
            invoke=invoke,
            delegate=delegate,
            epoch=parent.epoch + 1,
        )
        self._emit(
            "grant_delegated",
            operator_id=opener_op,
            detail={"invoke": ",".join(invoke), "delegate": ",".join(delegate)},
        )
        return grant

    def _grant_for_scope(
        self, scope_id: str
    ) -> AuthorityGrant | DelegatedAuthorityGrant:
        scope = self._scopes.get(scope_id)
        if scope and scope.grant_id and scope.grant_id in self._grants:
            return self._grants[scope.grant_id]
        return self._root_grant

    def _policy_interfaces(self) -> tuple[str, ...]:
        # The pinned policy envelope caps every face; the root grant projects it.
        return self._root_grant.delegate or self._root_grant.invoke

    # ------------------------------------------------------------------ #
    # Readiness, settlement, publication (static path)
    # ------------------------------------------------------------------ #

    def _open_roots(self) -> Advance:
        advance = Advance()
        for wi in list(self._work_items.values()):
            cont = self._continuations.get(wi.work_item_id)
            if cont is not None and not cont.waiting_on and wi.legacy_task_id:
                self._admit(wi.work_item_id, advance)
        for op_id in self._operators:
            # A child-template region opens only when its parent spawn materializes it,
            # and an agent-selected region only when the agent requests it, so neither
            # fires at the root.
            if (
                not self._is_control(op_id)
                or op_id in self._child_templates
                or op_id in self._agent_region_spawns
            ):
                continue
            cont = self._continuations.get(_control_key(op_id))
            if cont is not None and not cont.waiting_on:
                self._fire_control(op_id, advance)
        return advance

    def _admit(self, work_item_id: str, advance: Advance) -> None:
        """Move a work item whose predecessors settled to READY, gating on authority."""
        wi = self._work_items[work_item_id]
        if wi.status is not WorkItemStatus.BLOCKED:
            return
        interface = self._requested_interface(wi.operator_id)
        if interface is not None and interface not in self._root_grant.invoke:
            self._decisions.append(
                AuthorityDecision(
                    work_item_id=work_item_id,
                    grant_id=self._root_grant.grant_id,
                    interface=interface,
                    kind=AuthorityDecisionKind.DENIED,
                    denial_kind=DenialKind.AUTHORITY,
                    reason=f"interface {interface!r} outside root grant invoke face",
                )
            )
            self._emit(
                "authority_denied",
                work_item_id=work_item_id,
                operator_id=wi.operator_id,
            )
            advance.failed.extend(self._settle_failure(work_item_id))
            return
        if interface is not None:
            self._decisions.append(
                AuthorityDecision(
                    work_item_id=work_item_id,
                    grant_id=self._root_grant.grant_id,
                    interface=interface,
                    kind=AuthorityDecisionKind.GRANTED,
                )
            )
        wi.status = WorkItemStatus.READY
        self._emit(
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
        if self._is_control(operator_id):
            if self._kind(operator_id) is OperatorKind.JOIN:
                return
            if (cont := self._continuations.get(_control_key(operator_id))) is not None:
                cont.waiting_on.clear()
            self._emit("region_skipped", operator_id=operator_id)
        else:
            wi_id = self._wi_by_operator.get(operator_id)
            if wi_id is None:
                return
            wi = self._work_items[wi_id]
            if wi.status in _TERMINAL_WI:
                return
            wi.status = WorkItemStatus.SETTLED
            wi.outcome = PublicationOutcome.EXPLICIT_EMPTY
            self._publish(
                operator_id, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
            )
        for successor in sorted(self._forward.get(operator_id, ())):
            self._settle_empty_successor(successor, visited)

    def _settle_failure(self, work_item_id: str) -> list[str]:
        wi = self._work_items[work_item_id]
        if wi.status in _TERMINAL_WI:
            return []
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = PublicationOutcome.DECLARED_FAILURE
        self._publish(wi.operator_id, PublicationOutcome.DECLARED_FAILURE, None)
        cascade = [wi.legacy_task_id]
        for successor in sorted(self._forward.get(wi.operator_id, ())):
            if self._is_control(successor):
                continue
            if succ_wi := self._wi_by_operator.get(successor):
                cascade.extend(self._settle_failure(succ_wi))
        return cascade

    def _publish(
        self, operator_id: str, outcome: PublicationOutcome, value_ref: ValueRef | None
    ) -> None:
        for slot_key in self._slots_by_operator.get(operator_id, ()):
            self._write_publication(self._slots[slot_key], outcome, value_ref)

    def _publish_keyed(
        self,
        spawn_op: str,
        activation: Activation,
        outcome: PublicationOutcome,
        value_ref: ValueRef | None,
    ) -> None:
        for decl in self._bundle.template.result_declarations:
            if (
                decl.source_ref != spawn_op
                or decl.cardinality is not CardinalityKind.KEYED_COLLECTION
            ):
                continue
            self._write_publication(
                ResultSlot(
                    instance_id=self._instance.instance_id,
                    output_id=decl.output_id,
                    source_operator_id=spawn_op,
                    scope_id=activation.scope_id,
                    logical_key=str(activation.child_index),
                ),
                outcome,
                value_ref,
            )

    def _write_publication(
        self, slot: ResultSlot, outcome: PublicationOutcome, value_ref: ValueRef | None
    ) -> None:
        if slot.slot_key in self._publications:
            return
        self._slots[slot.slot_key] = slot.model_copy(update={"published": True})
        self._publications[slot.slot_key] = ResultPublication(
            slot_key=slot.slot_key,
            output_id=slot.output_id,
            outcome=outcome,
            value_ref=value_ref,
        )
        self._emit(
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
        self._emit(
            "effect_receipt",
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )

    def _requested_interface(self, operator_id: str) -> str | None:
        profile = self._profiles.get(operator_id)
        if profile is not None and profile.effect is EffectClass.EXTERNAL_EFFECT:
            return operator_id
        return None

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def resolve_output(self, output_id: str) -> ResultPublication | None:
        """Resolve a declared logical output to its terminal publication, if any."""
        for slot in self._slots.values():
            if slot.output_id == output_id:
                return self._publications.get(slot.slot_key)
        return None

    def resolve_legacy_task(self, task_id: str) -> ResultPublication | None:
        """Resolve a legacy task id's induced output slot (compatibility adapter)."""
        return self.resolve_output(f"legacy:{task_id}")

    def failure_reason(self, task_id: str) -> str | None:
        """The recorded authority-denial reason for a task, when one settled it."""
        wi_id = self._wi_by_task.get(task_id)
        if wi_id is None:
            return None
        for decision in reversed(self._decisions):
            if (
                decision.work_item_id == wi_id
                and decision.kind is AuthorityDecisionKind.DENIED
            ):
                return f"authority denied: {decision.reason or decision.interface}"
        return None

    def recovery_disposition(self, task_id: str) -> RecoveryDisposition | None:
        """Whether the task's operation may be recomputed or must be restored."""
        profile = self._profiles.get(self._operator_for_task(task_id) or "")
        return classify_recovery(profile) if profile else None

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
        return [
            (e.kind, e.operator_id or e.slot_key or e.work_item_id or "")
            for e in self._trace
        ]

    def capability(
        self, scope_id: str | None, axis: ProgressAxis
    ) -> ProgressCapability | None:
        return self._capabilities.get((scope_id, axis)) if scope_id else None

    def scope_for(self, region_op: str) -> str | None:
        return self._scope_id_for(region_op)

    def region_scope_for(self, agent_activation: str, role: str) -> str | None:
        """The child-init scope an agent's declared role region opened, if entered."""
        agent = self._activations.get(agent_activation)
        op = self._operators.get(agent.operator_id) if agent else None
        if not isinstance(op, AgentOperator):
            return None
        region_op = self._agent_region_op(op, role)
        opener = self._region_openers.get((agent_activation, region_op or ""))
        return self._scope_by_activation.get(opener) if opener else None

    def grant_for(self, region_op: str) -> DelegatedAuthorityGrant | None:
        scope_id = self._scope_id_for(region_op)
        if scope_id is None:
            return None
        grant_id = self._scopes[scope_id].grant_id
        return self._grants.get(grant_id) if grant_id else None

    def region_closed(self, region_op: str) -> bool:
        scope_id = (
            self._scope_for_join(region_op)
            if self._kind(region_op) is OperatorKind.JOIN
            else self._scope_id_for(region_op)
        )
        return scope_id in self._released_scopes if scope_id else False

    def spawn_successor(self, operator_id: str) -> str | None:
        """The spawn region an operator feeds via a forward edge, if any."""
        for successor in self._forward.get(operator_id, ()):
            if self._kind(successor) is OperatorKind.SPAWN:
                return successor
        return None

    def child_template_of(self, spawn_op: str) -> str | None:
        """The operator id of a spawn's child template, if it declares one."""
        op = self._operators.get(spawn_op)
        return op.child_template_ref if isinstance(op, SpawnRegion) else None

    def spawn_is_open(self, spawn_op: str) -> bool:
        """Whether a spawn's child-init capability still admits new children.

        False once the spawn has sealed or revoked, or before its child-init scope
        opens, so a re-driven fan-out over an already-closed spawn is a clean no-op. A
        read-only query: it never opens a scope.
        """
        scope_id = self._scope_id_for(spawn_op)
        if scope_id is None:
            return False
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        return cap.status is CapabilityStatus.OPEN

    def episode_spec(self, task_id: str) -> EpisodeSpec | None:
        """The run-to-yield episode a task's operator lowers to, if the plan cut it."""
        wi = self._work_item_for_task(task_id)
        if wi is None:
            return None
        for node in self._bundle.plan.nodes:
            if node.episode is None:
                continue
            if node.logical_ref == wi.operator_id or (
                wi.operator_id in node.episode.fused_refs
            ):
                return node.episode
        return None

    def work_item(self, task_id: str) -> WorkItem | None:
        return self._work_item_for_task(task_id)

    def agent_operator(self, task_id: str) -> AgentOperator | None:
        """The agent operator a dispatched task realizes, resolving its work item."""
        wi = self._work_item_for_task(task_id)
        operator_id = wi.operator_id if wi is not None else task_id
        op = self._operators.get(operator_id)
        return op if isinstance(op, AgentOperator) else None

    def invocation_for_task(self, task_id: str) -> Invocation | None:
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.invocation_id is None:
            return None
        return self._invocations.get(wi.invocation_id)

    @property
    def instance(self) -> WorkflowInstance:
        return self._instance

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            instance=self._instance,
            root_scope=self._root_scope,
            root_grant=self._root_grant,
            scopes=list(self._scopes.values()),
            activations=list(self._activations.values()),
            work_items=list(self._work_items.values()),
            continuations=list(self._continuations.values()),
            records=list(self._records),
            invocations=list(self._invocations.values()),
            attempts=list(self._attempts.values()),
            boundary_events=list(self._boundary_events.values()),
            effect_receipts=list(self._receipts.values()),
            authority_decisions=list(self._decisions),
            delegated_grants=list(self._grants.values()),
            progress_capabilities=list(self._capabilities.values()),
            result_slots=list(self._slots.values()),
            result_publications=list(self._publications.values()),
            trace=list(self._trace),
            released_scopes=sorted(self._released_scopes),
            next_seq=self._next_seq,
        )

    def reconcile_pending(self, task_id: str) -> bool:
        """Re-derive readiness for a task whose durable record shows PENDING.

        Returns whether the work item is ready to admit. A work item the snapshot still
        shows in flight — a crash after a retry persisted the PENDING record but before
        the ledger caught up — is reset to ready with its lost attempt marked, so the
        retry is not orphaned; a work item whose predecessors have not all settled stays
        blocked.
        """
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status in _TERMINAL_WI:
            return False
        cont = self._continuations.get(wi.work_item_id)
        if cont is not None and cont.waiting_on:
            wi.status = WorkItemStatus.BLOCKED
            return False
        if wi.status is WorkItemStatus.DISPATCHED:
            if attempt := self._latest_attempt(wi):
                attempt.status = AttemptStatus.LOST
                attempt.finished_at = now_iso()
            self._emit(
                "attempt_lost_on_restart",
                work_item_id=wi.work_item_id,
                operator_id=wi.operator_id,
            )
        wi.status = WorkItemStatus.READY
        return True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _control_activation(self, operator_id: str) -> str:
        for a in self._activations.values():
            if a.operator_id == operator_id and a.kind not in (
                "child",
                "iteration",
                "region",
            ):
                return a.activation_id
        return operator_id

    def _join_for_spawn(self, spawn_op: str) -> str | None:
        for edge in self._bundle.template.edges:
            if edge.from_op == spawn_op and self._kind(edge.to_op) is OperatorKind.JOIN:
                return edge.to_op
        return None

    def _scope_for_join(self, join_op: str) -> str | None:
        for edge in self._bundle.template.edges:
            if edge.to_op == join_op and self._kind(edge.from_op) is OperatorKind.SPAWN:
                return self._scope_id_for(edge.from_op)
        return None

    def _edge_from_port(self, from_op: str, to_op: str) -> str | None:
        for edge in self._bundle.template.edges:
            if edge.from_op == from_op and edge.to_op == to_op:
                return edge.from_port
        return None

    def _handle_operator(self, handle: str) -> str:
        """The operator id a region handle names: the handle, or its activation's."""
        act = self._activations.get(handle)
        return act.operator_id if act else handle

    def _resolve_opener_activation(self, handle: str) -> str | None:
        """The opener activation a region handle resolves to.

        An activation-id handle is its own opener (a recursive level); an operator-id
        handle resolves to its scope-owning activation, or, before the scope opens, its
        control activation.
        """
        if handle in self._activations:
            return handle
        if acts := self._owner_acts_by_operator.get(handle):
            return acts[-1]
        ctrl = self._control_activation(handle)
        return ctrl if ctrl in self._activations else None

    def _scope_id_for(self, handle: str) -> str | None:
        opener = self._resolve_opener_activation(handle)
        return self._scope_by_activation.get(opener) if opener else None

    def _require_child_init_scope(self, handle: str) -> str:
        if (scope_id := self._scope_id_for(handle)) is not None:
            return scope_id
        opener = self._resolve_opener_activation(handle)
        if opener is None:
            raise RegionError(f"{handle!r} has no opener activation")
        # A lazily opened scope (an agent's first spawn) nests under the opener's own
        # enclosing scope, so a nested agent's recursion depth is counted correctly.
        parent = (
            self._activations[opener].scope_id if opener in self._activations else None
        )
        return self._open_child_init_scope(opener, parent_scope_id=parent)

    def _require_loop_scope(self, handle: str) -> str:
        if (scope_id := self._scope_id_for(handle)) is not None:
            return scope_id
        opener = self._resolve_opener_activation(handle)
        if opener is None:
            raise RegionError(f"{handle!r} has no opener activation")
        return self._open_loop(opener)

    def _latest_carried(self, scope_id: str) -> ValueRef | None:
        latest: ValueRef | None = None
        best = -1
        for record in self._records:
            if record.scope_id == scope_id and record.loop_time >= best:
                best = record.loop_time
                latest = record.value_ref
        return latest

    def _emit(
        self, kind: str, *, detail: dict[str, str] | None = None, **fields: str | None
    ) -> None:
        self._trace.append(
            OrchestrationEvent(
                seq=self._next_seq,
                kind=kind,
                operator_id=fields.get("operator_id"),
                work_item_id=fields.get("work_item_id"),
                attempt_id=fields.get("attempt_id"),
                invocation_id=fields.get("invocation_id"),
                slot_key=fields.get("slot_key"),
                detail=detail
                or {
                    k: v
                    for k, v in fields.items()
                    if k not in _EVENT_FIELDS and v is not None
                },
            )
        )
        self._next_seq += 1

    def _work_item_for_task(self, task_id: str) -> WorkItem | None:
        wi_id = self._wi_by_task.get(task_id)
        return self._work_items.get(wi_id) if wi_id else None

    def _operator_for_task(self, task_id: str) -> str | None:
        wi = self._work_item_for_task(task_id)
        return wi.operator_id if wi else None

    def _latest_attempt(self, wi: WorkItem) -> Attempt | None:
        return self._attempts.get(wi.attempt_ids[-1]) if wi.attempt_ids else None

    def latest_attempt_open(self, task_id: str) -> bool:
        """Whether the work item's latest attempt still expects a terminal report.

        A reroute that re-enqueues an episode closes the attempt that produced the turn,
        so a completion whose attempt is already closed is a superseded replay —
        applying it would preempt the live turn. A genuine terminal report lands while
        its attempt is still issued or running.
        """
        wi = self._work_item_for_task(task_id)
        if wi is None:
            return False
        attempt = self._latest_attempt(wi)
        return attempt is not None and attempt.status in (
            AttemptStatus.ISSUED,
            AttemptStatus.RUNNING,
        )
