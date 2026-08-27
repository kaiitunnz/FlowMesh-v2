"""The orchestration engine over the acyclic compatibility physical plan.

The ledger owns semantic readiness (note 21 §8.3): it turns settled records into
ready work items, incrementally materializing the activation graph rather than
precreating attempts. Scheduler/worker placement stays a physical decision made by
the runtime and dispatcher and never changes what the ledger considers ready.
"""

from dataclasses import dataclass, field

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
    EffectClass,
    LeafOperator,
    LeafProfile,
)
from ..utils.time import now_iso
from .outcomes import (
    AdmissionError,
    check_admissible,
    classify_recovery,
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
    Continuation,
    EffectReceipt,
    Invocation,
    InvocationState,
    LedgerSnapshot,
    OrchestrationEvent,
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


@dataclass
class Advance:
    """Runtime-visible effect of a ledger transition, in legacy task ids.

    ``ready`` work items become admissible for a new attempt, ``failed`` ones settle
    terminally and cascade, and ``retry`` reissues an existing work item as a fresh
    attempt under its stable identity.
    """

    ready: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    retry: list[str] = field(default_factory=list)


class OrchestrationLedger:
    """In-memory view of one workflow instance's durable orchestration ledger."""

    def __init__(self, snapshot: LedgerSnapshot, bundle: PersistedV2Workflow) -> None:
        self._instance = snapshot.instance
        self._root_scope = snapshot.root_scope
        self._root_grant = snapshot.root_grant
        self._bundle = bundle
        self._next_seq = snapshot.next_seq
        self._initial = Advance()

        self._activations = {a.activation_id: a for a in snapshot.activations}
        self._work_items = {w.work_item_id: w for w in snapshot.work_items}
        self._continuations = {c.work_item_id: c for c in snapshot.continuations}
        self._records = list(snapshot.records)
        self._invocations = {i.invocation_id: i for i in snapshot.invocations}
        self._attempts = {a.attempt_id: a for a in snapshot.attempts}
        self._receipts = {r.invocation_id: r for r in snapshot.effect_receipts}
        self._decisions = list(snapshot.authority_decisions)
        self._slots = {s.slot_key: s for s in snapshot.result_slots}
        self._publications = {p.slot_key: p for p in snapshot.result_publications}
        self._trace = list(snapshot.trace)

        self._wi_by_task = {
            w.legacy_task_id: w.work_item_id for w in self._work_items.values()
        }
        self._wi_by_operator = {
            w.operator_id: w.work_item_id for w in self._work_items.values()
        }
        self._profiles: dict[str, LeafProfile] = {
            op.operator_id: op.profile
            for op in bundle.template.operators
            if isinstance(op, LeafOperator)
        }
        self._slots_by_operator: dict[str, list[str]] = {}
        for slot in self._slots.values():
            self._slots_by_operator.setdefault(slot.source_operator_id, []).append(
                slot.slot_key
            )
        self._succs: dict[str, set[str]] = {op: set() for op in self._wi_by_operator}
        for edge in bundle.template.edges:
            if edge.feedback:
                continue
            if edge.from_op in self._succs and edge.to_op in self._succs:
                self._succs[edge.from_op].add(edge.to_op)

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
    ) -> "OrchestrationLedger":
        """Materialize a ledger from a compiled acyclic bundle.

        Rejects operators the compatibility path cannot honor (agents, non-leaf
        regions, residency, and non-replayable effects), then creates one activation,
        work item, and continuation per result-owning operator and one slot per
        declared output. ``granted_interfaces`` pins the root grant's invoke face; when
        omitted every requested interface is granted.
        """
        template = bundle.template
        replay = {
            b.source_ref: b.replay_contract
            for b in template.effect_boundaries
            if b.source_ref
        }
        requested: set[str] = set()
        for op in template.operators:
            if isinstance(op, AgentOperator):
                raise AdmissionError(
                    f"operator {op.operator_id!r} is an agent; dynamic agent "
                    "orchestration is not part of the acyclic compatibility subset"
                )
            if not isinstance(op, LeafOperator):
                raise AdmissionError(
                    f"operator {op.operator_id!r} of kind {op.kind.value!r} is not "
                    "part of the acyclic compatibility subset"
                )
            check_admissible(
                op.operator_id,
                op.profile.effect,
                replay.get(op.operator_id),
                op.residency_only,
            )
            if op.profile.effect is EffectClass.EXTERNAL_EFFECT:
                requested.add(op.operator_id)

        scope = Scope(scope_id=new_scope_id(), instance_id=instance_id)
        invoke = requested if granted_interfaces is None else set(granted_interfaces)
        grant = AuthorityGrant(
            grant_id=new_authority_grant_id(),
            instance_id=instance_id,
            policy_id=f"policy:{instance_id}",
            invoke=tuple(sorted(invoke)),
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
            if not edge.feedback and edge.to_op in preds:
                preds[edge.to_op].add(edge.from_op)

        activations: list[Activation] = []
        work_items: list[WorkItem] = []
        continuations: list[Continuation] = []
        for op in template.operators:
            assert isinstance(op, LeafOperator)
            activation = Activation(
                activation_id=new_activation_id(),
                instance_id=instance_id,
                scope_id=scope.scope_id,
                operator_id=op.operator_id,
            )
            work_item = WorkItem(
                work_item_id=new_work_item_id(),
                activation_id=activation.activation_id,
                operator_id=op.operator_id,
                legacy_task_id=op.operator_id,
                effect_class=op.profile.effect,
                recovery=op.profile.recovery,
                replay_contract=replay.get(op.operator_id),
            )
            activations.append(activation)
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
        ]
        snapshot = LedgerSnapshot(
            instance=instance,
            root_scope=scope,
            root_grant=grant,
            activations=activations,
            work_items=work_items,
            continuations=continuations,
            result_slots=slots,
        )
        ledger = cls(snapshot, bundle)
        ledger._initial = ledger._open_roots()
        return ledger

    def initial_advance(self) -> Advance:
        """Ready/failed roots admitted at submission time."""
        return self._initial

    # ------------------------------------------------------------------ #
    # Physical attempt lifecycle
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

        ``empty`` marks a conditional-skip settlement, resolving the declared output
        to an explicit-empty publication rather than a value.
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
        self._records.append(
            Record(
                operator_id=wi.operator_id,
                activation_id=wi.activation_id,
                scope_id=self._root_scope.scope_id,
                value_ref=value_ref,
            )
        )
        self._emit(
            "record_delivered", work_item_id=wi.work_item_id, operator_id=wi.operator_id
        )
        ready, failed = self._release_successors(wi.operator_id)
        return Advance(ready=ready, failed=failed)

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
        invocation.state = next_on_uncertain(invocation.state, invocation.replayable)
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
            "invocation_ambiguity_terminal",
            work_item_id=wi.work_item_id,
            invocation_id=wi.invocation_id,
        )
        return Advance(failed=self._settle_failure(wi.work_item_id))

    # ------------------------------------------------------------------ #
    # Readiness, settlement, publication
    # ------------------------------------------------------------------ #

    def _open_roots(self) -> Advance:
        advance = Advance()
        for wi in list(self._work_items.values()):
            if not self._continuations[wi.work_item_id].waiting_on:
                self._admit(wi.work_item_id, advance)
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

    def _release_successors(self, operator_id: str) -> tuple[list[str], list[str]]:
        advance = Advance()
        for successor in sorted(self._succs.get(operator_id, ())):
            cont = self._continuations[self._wi_by_operator[successor]]
            cont.waiting_on.discard(operator_id)
            if not cont.waiting_on:
                self._admit(cont.work_item_id, advance)
        return advance.ready, advance.failed

    def _settle_failure(self, work_item_id: str) -> list[str]:
        wi = self._work_items[work_item_id]
        if wi.status is WorkItemStatus.SETTLED:
            return []
        wi.status = WorkItemStatus.SETTLED
        wi.outcome = PublicationOutcome.DECLARED_FAILURE
        self._publish(wi.operator_id, PublicationOutcome.DECLARED_FAILURE, None)
        cascade = [wi.legacy_task_id]
        for successor in sorted(self._succs.get(wi.operator_id, ())):
            cascade.extend(self._settle_failure(self._wi_by_operator[successor]))
        return cascade

    def _publish(
        self, operator_id: str, outcome: PublicationOutcome, value_ref: ValueRef | None
    ) -> None:
        for slot_key in self._slots_by_operator.get(operator_id, ()):
            if slot_key in self._publications:
                continue
            slot = self._slots[slot_key]
            self._slots[slot_key] = slot.model_copy(update={"published": True})
            self._publications[slot_key] = ResultPublication(
                slot_key=slot_key,
                output_id=slot.output_id,
                outcome=outcome,
                value_ref=value_ref,
            )
            self._emit(
                "result_published",
                operator_id=operator_id,
                slot_key=slot_key,
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
        """The tool/service interface an operator invokes, if any.

        In the compatibility path only an external-effect leaf issues a service
        invocation; its operator id names the interface checked against the grant.
        """
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
            activations=list(self._activations.values()),
            work_items=list(self._work_items.values()),
            continuations=list(self._continuations.values()),
            records=list(self._records),
            invocations=list(self._invocations.values()),
            attempts=list(self._attempts.values()),
            effect_receipts=list(self._receipts.values()),
            authority_decisions=list(self._decisions),
            result_slots=list(self._slots.values()),
            result_publications=list(self._publications.values()),
            trace=list(self._trace),
            next_seq=self._next_seq,
        )

    def reconcile_pending(self, task_id: str) -> bool:
        """Re-derive readiness for a task whose durable record shows PENDING.

        Returns whether the work item is ready to admit. A work item the snapshot
        still shows in flight — a crash after a retry persisted the PENDING record but
        before the ledger caught up — is reset to ready with its lost attempt marked,
        so the retry is not orphaned; a work item whose predecessors have not all
        settled stays blocked.
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

    def _emit(self, kind: str, **fields: str) -> None:
        self._trace.append(
            OrchestrationEvent(
                seq=self._next_seq,
                kind=kind,
                operator_id=fields.get("operator_id"),
                work_item_id=fields.get("work_item_id"),
                attempt_id=fields.get("attempt_id"),
                invocation_id=fields.get("invocation_id"),
                slot_key=fields.get("slot_key"),
                detail={k: v for k, v in fields.items() if k not in _EVENT_FIELDS},
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
