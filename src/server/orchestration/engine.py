"""The orchestration engine over the transparent structured-region physical plan.

The engine owns semantic readiness (note 21 §8.3): it turns settled records into ready
work items, incrementally materializing the activation graph rather than precreating
attempts. Static top-level leaf operators materialize eagerly and dispatch through the
runtime; control operators (branch, merge, spawn, join, loop) and agents settle inside
the ledger and never dispatch; spawn children and loop iterations materialize as records
flow. Progress capabilities (child-init and loop-time) are the closure authority — a
region closes only when its combined account seals and drains, never on an observed
empty set. Scheduler/worker placement stays a physical decision that never changes what
the engine considers ready.
"""

from dataclasses import dataclass, field
from typing import Self

from shared.utils import (
    new_activation_id,
    new_attempt_id,
    new_authority_grant_id,
    new_invocation_id,
    new_scope_id,
    new_work_item_id,
)

from ..task.v2.representations.bundle import PersistedV2Workflow
from ..task.v2.representations.operators import (
    AgentOperator,
    BranchRegion,
    EffectClass,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LeafProfile,
    LogicalOperator,
    OperatorKind,
    SpawnRegion,
)
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
# Control operators settle in-ledger and never dispatch; an agent is a scope-opening
# boundary-signature site, not an eagerly dispatchable leaf.
_CONTROL_KINDS = _REGION_KINDS | {OperatorKind.AGENT}
_CHILD_INIT_OPENERS = frozenset({OperatorKind.SPAWN, OperatorKind.AGENT})


class RegionError(ValueError):
    """Raised when a structured-region operation is invalid, e.g. a child after seal."""


def _control_key(operator_id: str) -> str:
    return f"control:{operator_id}"


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
    """In-memory view of one workflow instance's durable orchestration ledger."""

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
        self._budget = budget or ScopeBudget.from_env()
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
        self._child_templates = {
            op.child_template_ref
            for op in bundle.template.operators
            if isinstance(op, SpawnRegion) and op.child_template_ref
        }
        self._wi_by_task = {
            w.legacy_task_id: w.work_item_id
            for w in self._work_items.values()
            if w.legacy_task_id
        }
        self._wi_by_operator = {
            w.operator_id: w.work_item_id
            for w in self._work_items.values()
            if w.legacy_task_id
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
        self._scope_by_owner = {
            s.owner_operator_id: s.scope_id
            for s in self._scopes.values()
            if s.owner_operator_id
        }
        self._loop_time = {
            owner: max(
                (
                    a.loop_time
                    for a in self._activations.values()
                    if a.scope_id == scope_id
                ),
                default=0,
            )
            for owner, scope_id in self._scope_by_owner.items()
            if self._kind(owner) is OperatorKind.LOOP_CONTEXT
        }
        # A region's release/egress record is delivered at the root scope; a loop's
        # feedback records sit at the loop scope, so gate on the root scope to avoid
        # treating a mid-loop snapshot as already egressed.
        self._released_regions = {
            r.operator_id
            for r in self._records
            if r.scope_id == self._root_scope.scope_id
            and self._kind(r.operator_id)
            in (OperatorKind.JOIN, OperatorKind.LOOP_CONTEXT)
        }
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
        child_templates = {
            op.child_template_ref
            for op in template.operators
            if isinstance(op, SpawnRegion) and op.child_template_ref
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
                isinstance(op, LeafOperator) and op.operator_id not in child_templates
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
            assert isinstance(op, LeafOperator)
            work_item = WorkItem(
                work_item_id=new_work_item_id(),
                activation_id=activation.activation_id,
                operator_id=op.operator_id,
                legacy_task_id=op.operator_id,
                effect_class=op.profile.effect,
                recovery=op.profile.recovery,
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
        if wi is None or wi.status is WorkItemStatus.SETTLED:
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
        if wi is None or wi.status is WorkItemStatus.SETTLED:
            return Advance()
        outcome = (
            PublicationOutcome.EXPLICIT_EMPTY if empty else PublicationOutcome.SUCCESS
        )
        value_ref = (
            ValueRef(kind="empty")
            if empty
            else ValueRef(kind="legacy_task_result", legacy_task_id=wi.legacy_task_id)
        )
        if attempt := self._latest_attempt(wi):
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.finished_at = now_iso()
        if wi.invocation_id is not None:
            invocation = self._invocations[wi.invocation_id]
            invocation.state = next_on_terminal(invocation.state)
            self._record_receipt(wi, outcome)
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = outcome
        self._publish(wi.operator_id, outcome, value_ref)
        return self._deliver_record(wi.operator_id, wi.activation_id, value_ref)

    def on_failed(self, task_id: str, error: str, *, retryable: bool) -> Advance:
        """Retry a work item as a fresh attempt, or settle it and cascade failure."""
        wi = self._work_item_for_task(task_id)
        if wi is None or wi.status is WorkItemStatus.SETTLED:
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
        return Advance(failed=self._settle_failure(wi.work_item_id))

    def on_uncertain(self, task_id: str) -> Advance:
        """Resolve a lost acknowledgement or route loss for an in-flight work item."""
        wi = self._work_item_for_task(task_id)
        if (
            wi is None
            or wi.status is WorkItemStatus.SETTLED
            or wi.invocation_id is None
        ):
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
        return Advance(failed=self._settle_failure(wi.work_item_id))

    # ------------------------------------------------------------------ #
    # Structured-region API (control settlement is internal; dynamic
    # cardinality is controller-driven over generic regions)
    # ------------------------------------------------------------------ #

    def spawn_child(self, spawn_op: str, *, operator_id: str | None = None) -> str:
        """Materialize one child activation under a spawn/agent's open child scope.

        Rejects a child after a definitive denial, or after the child-init capability is
        sealed or revoked (late-child prevention). When the child body is itself a scope
        opener, opens a nested scope so grandchildren attenuate from the child grant.
        Returns the child activation id.
        """
        spawn = self._operators[spawn_op]
        child_ref = spawn.child_template_ref if isinstance(spawn, SpawnRegion) else None
        body_ref = operator_id or child_ref or spawn_op
        if spawn_op in self._denied_spawns:
            self._emit(
                "child_rejected", operator_id=spawn_op, detail={"reason": "denied"}
            )
            raise RegionError(f"spawn {spawn_op!r} was denied; no child may be created")
        scope_id = self._require_child_init_scope(spawn_op)
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
        # Validate every budget before materializing, so a rejected child leaves no
        # half-open region whose join could never drain.
        self._charge_activation()
        body_opens_scope = self._kind(body_ref) in _CHILD_INIT_OPENERS or (
            self._kind(body_ref) is OperatorKind.LOOP_CONTEXT
        )
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
        child_wi = WorkItem(
            work_item_id=new_work_item_id(),
            activation_id=activation.activation_id,
            operator_id=body_ref,
            legacy_task_id="",
        )
        self._work_items[child_wi.work_item_id] = child_wi
        self._wi_by_activation[activation.activation_id] = child_wi.work_item_id
        cap.outstanding += 1
        self._emit(
            "child_spawned",
            operator_id=body_ref,
            detail={"scope": scope_id, "index": str(index)},
        )
        if self._kind(body_ref) in _CHILD_INIT_OPENERS:
            self._open_child_init_scope(body_ref, parent_scope_id=scope_id)
        elif self._kind(body_ref) is OperatorKind.LOOP_CONTEXT:
            self._open_loop(body_ref, parent_scope_id=scope_id)
        return activation.activation_id

    def seal_spawn(self, spawn_op: str) -> Advance:
        """Seal a spawn's child-init capability; no further children may be created."""
        scope_id = self._require_child_init_scope(spawn_op)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.SEALED
            self._emit(
                "child_init_sealed", operator_id=spawn_op, detail={"scope": scope_id}
            )
        return self._maybe_release_join(spawn_op)

    def revoke_spawn(self, spawn_op: str) -> None:
        """Revoke a spawn's child-init capability as a progress transition.

        Distinct from sealing: revocation withdraws the capability, while sealing marks
        a producer done. Both close the child-init axis once outstanding children drain.
        """
        scope_id = self._require_child_init_scope(spawn_op)
        cap = self._capability(scope_id, ProgressAxis.CHILD_INIT)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.REVOKED
            self._emit(
                "child_init_revoked", operator_id=spawn_op, detail={"scope": scope_id}
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
        if wi.status is WorkItemStatus.SETTLED:
            return Advance()
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = outcome
        cap = self._capability(activation.scope_id, ProgressAxis.CHILD_INIT)
        cap.outstanding = max(0, cap.outstanding - 1)
        scope = self._scopes[activation.scope_id]
        spawn_op = scope.owner_operator_id or ""
        self._publish_keyed(spawn_op, activation, outcome, value_ref)
        self._emit(
            "child_settled",
            operator_id=activation.operator_id,
            detail={"scope": activation.scope_id, "outcome": outcome.value},
        )
        return self._maybe_release_join(spawn_op)

    def route_branch(self, branch_op: str, selected_port: str) -> Advance:
        """Route a branch record to the selected port; settle the other ports empty."""
        if self._kind(branch_op) is not OperatorKind.BRANCH:
            raise RegionError(f"{branch_op!r} is not a branch region")
        advance = Advance()
        for successor in sorted(self._forward.get(branch_op, ())):
            from_port = self._edge_from_port(branch_op, successor)
            if from_port in (selected_port, None):
                self._release_one(successor, branch_op, advance)
            else:
                self._settle_empty_successor(successor, advance)
        self._emit(
            "branch_routed", operator_id=branch_op, detail={"port": selected_port}
        )
        return advance

    def loop_feedback(self, loop_op: str, *, value_ref: ValueRef | None = None) -> str:
        """Re-materialize a loop body at the next loop-time coordinate.

        Enforces well-founded logical time: loop_time strictly increases and stays under
        the iteration budget, so a finite prefix is acyclic after time unrolling.
        Returns the iteration activation id.
        """
        scope_id = self._require_loop_scope(loop_op)
        cap = self._capability(scope_id, ProgressAxis.LOOP_TIME)
        if cap.status is not CapabilityStatus.OPEN:
            raise RegionError(
                f"loop {loop_op!r} is {cap.status.value}; no feedback may arrive"
            )
        next_time = self._loop_time.get(loop_op, 0) + 1
        if next_time > self._budget.max_loop_iterations:
            self._exhaust_budget("loop_iterations", self._budget.max_loop_iterations)
        self._charge_activation()
        self._loop_time[loop_op] = next_time
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
        owner = self._scopes[activation.scope_id].owner_operator_id or ""
        return self._maybe_egress_loop(owner)

    def loop_seal(self, loop_op: str) -> Advance:
        """Seal a loop: no further feedback; egress once pending iterations drain."""
        scope_id = self._require_loop_scope(loop_op)
        cap = self._capability(scope_id, ProgressAxis.LOOP_TIME)
        if cap.status is CapabilityStatus.OPEN:
            cap.status = CapabilityStatus.SEALED
            self._emit("loop_sealed", operator_id=loop_op, detail={"scope": scope_id})
        return self._maybe_egress_loop(loop_op)

    def deny_spawn(
        self, spawn_op: str, interface: str, *, kind: DenialKind = DenialKind.AUTHORITY
    ) -> None:
        """Record a definitive dynamic authorization denial at a spawn site.

        A denial creates no child activation and no resident claim, and is separate
        from quota/rate/capacity/transport outcomes. It does not seal the child-init
        capability: grant denial and cardinality sealing stay distinct (§6.5).
        """
        self._denied_spawns.add(spawn_op)
        scope_id = self._scope_by_owner.get(spawn_op)
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
        scope_id = self._scope_by_owner.get(region_op)
        return (
            scope_id is not None
            and interface in self._grant_for_scope(scope_id).delegate
        )

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
            self._open_child_init_scope(operator_id)
        elif kind is OperatorKind.LOOP_CONTEXT:
            self._open_loop(operator_id)
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
        self, opener_op: str, *, parent_scope_id: str | None = None
    ) -> str:
        if opener_op in self._scope_by_owner:
            return self._scope_by_owner[opener_op]
        scope = self._new_child_scope(opener_op, parent_scope_id)
        self._scope_by_owner[opener_op] = scope.scope_id
        self._acquire_capability(scope.scope_id, ProgressAxis.CHILD_INIT)
        self._emit(
            "child_init_acquired",
            operator_id=opener_op,
            detail={"scope": scope.scope_id},
        )
        return scope.scope_id

    def _open_loop(self, loop_op: str, *, parent_scope_id: str | None = None) -> str:
        if loop_op in self._scope_by_owner:
            return self._scope_by_owner[loop_op]
        scope = self._new_child_scope(loop_op, parent_scope_id)
        self._scope_by_owner[loop_op] = scope.scope_id
        self._loop_time[loop_op] = 0
        self._acquire_capability(scope.scope_id, ProgressAxis.LOOP_TIME, coordinate=0)
        self._emit(
            "loop_ingress", operator_id=loop_op, detail={"scope": scope.scope_id}
        )
        return scope.scope_id

    def _maybe_release_join(self, spawn_op: str) -> Advance:
        join_op = self._join_for_spawn(spawn_op)
        if join_op is None or join_op in self._released_regions:
            return Advance()
        scope_id = self._scope_by_owner.get(spawn_op)
        if (
            scope_id is None
            or not self._capability(scope_id, ProgressAxis.CHILD_INIT).closed
        ):
            return Advance()
        return self._release_join(join_op, scope_id)

    def _release_join(self, join_op: str, scope_id: str) -> Advance:
        self._released_regions.add(join_op)
        join = self._operators[join_op]
        assert isinstance(join, JoinRegion)
        outcomes = [
            self._work_items[self._wi_by_activation[a.activation_id]].outcome
            for a in self._activations.values()
            if a.scope_id == scope_id and a.kind == "child"
        ]
        outcome = self._join_outcome(join, [o for o in outcomes if o is not None])
        self._emit(
            "join_released",
            operator_id=join_op,
            detail={"outcome": outcome.value, "children": str(len(outcomes))},
        )
        self._frontier_closed(scope_id)
        self._publish(join_op, outcome, ValueRef(kind="join_result"))
        return self._deliver_record(
            join_op, self._control_activation(join_op), ValueRef(kind="join_result")
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

    def _maybe_egress_loop(self, loop_op: str) -> Advance:
        if not loop_op or loop_op in self._released_regions:
            return Advance()
        scope_id = self._scope_by_owner.get(loop_op)
        if (
            scope_id is None
            or not self._capability(scope_id, ProgressAxis.LOOP_TIME).closed
        ):
            return Advance()
        self._released_regions.add(loop_op)
        self._frontier_closed(scope_id)
        self._emit("loop_egress", operator_id=loop_op, detail={"scope": scope_id})
        carried = self._latest_carried(loop_op)
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

    def _new_child_scope(self, owner_op: str, parent_scope_id: str | None) -> Scope:
        parent = self._scopes[parent_scope_id or self._root_scope.scope_id]
        self._check_scope_depth(parent.scope_id)
        grant = self._mint_delegated_grant(owner_op, parent.scope_id)
        scope = Scope(
            scope_id=new_scope_id(),
            instance_id=self._instance.instance_id,
            parent_scope_id=parent.scope_id,
            owner_operator_id=owner_op,
            grant_id=grant.grant_id,
            depth=parent.depth + 1,
        )
        self._scopes[scope.scope_id] = scope
        self._grants[grant.grant_id] = grant.model_copy(
            update={"scope_id": scope.scope_id}
        )
        return scope

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
        self, opener_op: str, parent_scope_id: str
    ) -> DelegatedAuthorityGrant:
        parent = self._grant_for_scope(parent_scope_id)
        opener = self._operators[opener_op]
        ceiling = (
            opener.authority
            if isinstance(opener, (SpawnRegion, AgentOperator))
            else None
        )
        ceiling_invoke = ceiling.invoke if ceiling else parent.delegate
        ceiling_delegate = ceiling.delegate if ceiling else parent.delegate
        envelope = self._policy_interfaces()
        invoke = attenuate(parent.delegate, ceiling_invoke, envelope)
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
            # so it nests under the parent scope rather than firing at the root.
            if not self._is_control(op_id) or op_id in self._child_templates:
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

    def _settle_empty_successor(self, operator_id: str, advance: Advance) -> None:
        """Skip a non-selected branch successor and its whole subtree.

        A leaf resolves empty; a control successor clears its pending inputs and is
        skipped so a downstream join never waits on an untaken path. A join itself is
        left to its own scope closure.
        """
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
            if wi.status is WorkItemStatus.SETTLED:
                return
            wi.status = WorkItemStatus.SETTLED
            wi.outcome = PublicationOutcome.EXPLICIT_EMPTY
            self._publish(
                operator_id, PublicationOutcome.EXPLICIT_EMPTY, ValueRef(kind="empty")
            )
        for successor in sorted(self._forward.get(operator_id, ())):
            self._settle_empty_successor(successor, advance)

    def _settle_failure(self, work_item_id: str) -> list[str]:
        wi = self._work_items[work_item_id]
        if wi.status is WorkItemStatus.SETTLED:
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
        return self._scope_by_owner.get(region_op)

    def grant_for(self, region_op: str) -> DelegatedAuthorityGrant | None:
        scope_id = self._scope_by_owner.get(region_op)
        if scope_id is None:
            return None
        grant_id = self._scopes[scope_id].grant_id
        return self._grants.get(grant_id) if grant_id else None

    def region_closed(self, region_op: str) -> bool:
        return region_op in self._released_regions

    def work_item(self, task_id: str) -> WorkItem | None:
        return self._work_item_for_task(task_id)

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
            effect_receipts=list(self._receipts.values()),
            authority_decisions=list(self._decisions),
            delegated_grants=list(self._grants.values()),
            progress_capabilities=list(self._capabilities.values()),
            result_slots=list(self._slots.values()),
            result_publications=list(self._publications.values()),
            trace=list(self._trace),
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
        if wi is None or wi.status is WorkItemStatus.SETTLED:
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
            if a.operator_id == operator_id and a.kind not in ("child", "iteration"):
                return a.activation_id
        return operator_id

    def _join_for_spawn(self, spawn_op: str) -> str | None:
        for edge in self._bundle.template.edges:
            if edge.from_op == spawn_op and self._kind(edge.to_op) is OperatorKind.JOIN:
                return edge.to_op
        return None

    def _edge_from_port(self, from_op: str, to_op: str) -> str | None:
        for edge in self._bundle.template.edges:
            if edge.from_op == from_op and edge.to_op == to_op:
                return edge.from_port
        return None

    def _require_child_init_scope(self, opener_op: str) -> str:
        return self._scope_by_owner.get(opener_op) or self._open_child_init_scope(
            opener_op
        )

    def _require_loop_scope(self, loop_op: str) -> str:
        return self._scope_by_owner.get(loop_op) or self._open_loop(loop_op)

    def _latest_carried(self, loop_op: str) -> ValueRef | None:
        latest: ValueRef | None = None
        best = -1
        for record in self._records:
            if record.operator_id == loop_op and record.loop_time >= best:
                best = record.loop_time
                latest = record.value_ref
        return latest

    def _emit(
        self, kind: str, *, detail: dict[str, str] | None = None, **fields: str
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
                or {k: v for k, v in fields.items() if k not in _EVENT_FIELDS},
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
