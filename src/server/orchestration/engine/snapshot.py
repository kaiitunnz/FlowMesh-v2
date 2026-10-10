"""Encodes and restores one workflow instance's ledger snapshot."""

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, ClassVar

from pydantic import BaseModel

from ..guardrails import ScopeBudget
from ..journal import AppendOnlyList, JournaledModel, TrackedDict, TrackedSet
from ..ledger_fields import (
    KEYED,
    SET_MEMBER,
    STRINGS,
    LedgerChanges,
    LedgerOrdinals,
    StoredLedger,
    encode_fields,
    encode_ledger,
    field_name,
    history_field,
    keyed_value,
    member_field,
    scalar_field,
    scalar_value,
)
from ..state import (
    AuthorityDecisionKind,
    ControlStatus,
    LedgerSnapshot,
    LoopInstanceStatus,
    PublicationOutcome,
    ResultPublication,
    ResultSlot,
)
from .attempts import AttemptLifecycle
from .authority import AuthorityLedger
from .boundaries import BoundaryLedger
from .embodiments import EmbodimentLedger
from .failures import DECLARED_FAILURE_REASON, FailureLedger
from .inputs import AcceptedInputLedger
from .ledger import OrchestrationLedger
from .publications import PublicationLedger
from .topology import PlanTopology


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


def _detached(snapshot: LedgerSnapshot) -> LedgerSnapshot:
    """The snapshot with a copy of each entity another ledger holds."""
    updates: dict[str, list[BaseModel]] = {}
    for name in KEYED:
        entities: list[BaseModel] = getattr(snapshot, name)
        if any(isinstance(e, JournaledModel) and e.held for e in entities):
            updates[name] = [
                e.model_copy() if isinstance(e, JournaledModel) and e.held else e
                for e in entities
            ]
    return snapshot.model_copy(update=updates) if updates else snapshot


def _applied(fields: Mapping[str, str], changes: LedgerChanges) -> dict[str, str]:
    applied = {} if changes.reset else dict(fields)
    applied.update(changes.fields)
    for name in changes.deleted:
        applied.pop(name, None)
    return applied


class SnapshotCodec:
    """Captures the instance's durable state as a ledger snapshot, and restores a
    stored one into the owners of that state in dependency order.

    The persisted collections it restores record each change they take in the
    ledger's journal, so a write carries only the fields that changed since the last
    one that landed.
    """

    # Test-only: each write's changes are checked to carry every change the ledger
    # took, against the image of the ledger the last landed write left stored.
    verify_changes: ClassVar[bool] = False
    verify_failures: ClassVar[list[str]] = []

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        failures: FailureLedger,
        publication: PublicationLedger,
        embodiments: EmbodimentLedger,
        inputs: AcceptedInputLedger,
        authority: AuthorityLedger,
        boundaries: BoundaryLedger,
        attempt_lifecycle: AttemptLifecycle,
        budget: ScopeBudget,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._failures = failures
        self._publication = publication
        self._embodiments = embodiments
        self._inputs = inputs
        self._authority = authority
        self._boundaries = boundaries
        self._attempt_lifecycle = attempt_lifecycle
        self._budget = budget
        self._journal = ledger.journal
        self._keyed: dict[str, TrackedDict[Any, Any]] = {}
        self._histories: dict[str, AppendOnlyList[Any]] = {}
        self._sets: dict[str, TrackedSet[str]] = {}
        self._written_lengths: dict[str, int] = {}
        self._written_scalars: dict[str, Any] = {}
        # Rewrites of the whole ledger owed and made, by count.
        self._rewrite = 0
        self._rewritten = 0
        self._acknowledged: dict[str, str] | None = None

    def _keyed_dict[K: Hashable, V](
        self, name: str, items: Iterable[tuple[K, V]] = ()
    ) -> TrackedDict[K, V]:
        tracked = TrackedDict[K, V](self._journal, name, items)
        self._keyed[name] = tracked
        return tracked

    def _history[T](self, name: str, entries: Iterable[T]) -> AppendOnlyList[T]:
        history = AppendOnlyList(entries)
        self._histories[name] = history
        return history

    def _set(self, name: str, members: Iterable[str]) -> TrackedSet[str]:
        tracked = TrackedSet(self._journal, name, members)
        self._sets[name] = tracked
        return tracked

    def restore(
        self, snapshot: LedgerSnapshot, ordinals: LedgerOrdinals | None = None
    ) -> None:
        """Restore a ledger snapshot; ``ordinals`` names the stored insertion ordinal of
        each keyed entry when it is the stored ledger, which a write then changes only
        where the restore did. Without them the next write rewrites the ledger."""
        snapshot = _detached(snapshot)
        self._keyed.clear()
        self._histories.clear()
        self._sets.clear()
        self._ledger.scopes = self._keyed_dict("scopes")
        self._ledger.subscopes = {}
        self._ledger.open_subscopes = {}
        for scope in snapshot.scopes:
            self._ledger.add_scope(scope)
        self._ledger.scopes.setdefault(
            self._ledger.root_scope.scope_id, self._ledger.root_scope
        )
        self._ledger.activations = self._keyed_dict("activations")
        self._ledger.children_by_scope = {}
        self._ledger.static_activations = {}
        self._ledger.open_children = {}
        self._ledger.scope_population = Counter()
        self._ledger.scope_children = Counter()
        self._ledger.dynamic_activations = 0
        for activation in snapshot.activations:
            self._ledger.add_activation(activation)
        self._ledger.work_items = self._keyed_dict("work_items")
        self._ledger.open_task_items = {}
        for wi in snapshot.work_items:
            self._ledger.add_work_item(wi)
        self._ledger.continuations = self._keyed_dict("continuations")
        self._ledger.input_candidates = {}
        for continuation in snapshot.continuations:
            self._ledger.set_continuation(continuation)
        self._ledger.records = self._history("records", snapshot.records)
        self._inputs.accepted_inputs = self._history(
            "accepted_inputs", snapshot.accepted_inputs
        )
        self._inputs.accepted_by_activation = {}
        for accepted in self._inputs.accepted_inputs:
            self._inputs.accepted_by_activation.setdefault(
                accepted.activation_id, []
            ).append(accepted)
        self._ledger.region_aggregates = self._history(
            "region_aggregates", snapshot.region_aggregates
        )
        self._ledger.aggregate_by_join = {}
        # A stored ledger may hold a nested level's aggregate after its root level's;
        # the root level's is the one delivered downstream.
        for aggregate in self._ledger.region_aggregates:
            join_key = aggregate.occurrence
            if join_key not in self._ledger.aggregate_by_join or not any(
                (act := self._ledger.activations.get(member.child_activation_id))
                is not None
                and not self._ledger.root_level(act.scope_id)
                for member in aggregate.members
            ):
                self._ledger.aggregate_by_join[join_key] = aggregate
        self._ledger.invocations = self._keyed_dict(
            "invocations", ((i.invocation_id, i) for i in snapshot.invocations)
        )
        self._ledger.attempts = self._keyed_dict(
            "attempts", ((a.attempt_id, a) for a in snapshot.attempts)
        )
        self._embodiments.embodiment_selections = self._keyed_dict(
            "embodiment_selections",
            ((sel.work_item_id, sel) for sel in snapshot.embodiment_selections),
        )
        self._embodiments.input_resolutions = self._keyed_dict(
            "input_resolutions",
            ((res.work_item_id, res) for res in snapshot.input_resolutions),
        )
        self._embodiments.input_preparations = self._keyed_dict(
            "input_preparations",
            ((prep.work_item_id, prep) for prep in snapshot.input_preparations),
        )
        self._attempt_lifecycle.receipts = self._keyed_dict(
            "effect_receipts", ((r.invocation_id, r) for r in snapshot.effect_receipts)
        )
        self._authority.decisions = self._history(
            "authority_decisions", snapshot.authority_decisions
        )
        self._authority.grants = self._keyed_dict(
            "delegated_grants", ((g.grant_id, g) for g in snapshot.delegated_grants)
        )
        self._ledger.capabilities = self._keyed_dict(
            "progress_capabilities",
            (((c.scope_id, c.axis), c) for c in snapshot.progress_capabilities),
        )
        self._publication.slots = self._keyed_dict(
            "result_slots", ((s.slot_key, s) for s in snapshot.result_slots)
        )
        self._publication.publications = self._keyed_dict(
            "result_publications",
            _rekeyed_publications(
                self._publication.slots.values(), snapshot.result_publications
            ).items(),
        )
        self._ledger.trace = self._history("trace", snapshot.trace)
        self._ledger.private_state.adopt(
            self._keyed_dict(
                "private_state",
                (
                    (lineage.binding.reference.activation_id, lineage)
                    for lineage in snapshot.private_state
                ),
            )
        )

        # Not persisted: each region activation names the parent and operator that
        # opened it.
        self._ledger.region_openers = {
            (a.parent_activation_id, a.operator_id): a.activation_id
            for a in self._ledger.activations.values()
            if a.kind == "region" and a.parent_activation_id
        }
        self._boundaries.boundary_events = self._keyed_dict(
            "boundary_events",
            (
                ((b.activation, b.call_correlation), b)
                for b in snapshot.boundary_events
                if b.activation and b.call_correlation
            ),
        )
        self._ledger.wi_by_task = {
            w.legacy_task_id: w.work_item_id
            for w in self._ledger.work_items.values()
            if w.legacy_task_id
        }
        # An operator inside a region definition runs once per occurrence, so its work
        # is addressed by occurrence, task or activation, never by operator alone.
        self._ledger.occurrences = self._keyed_dict("occurrences")
        self._ledger.occurrences_by_scope = {}
        self._ledger.open_occurrences = {}
        self._ledger.occurrence_by_activation = {}
        for occurrence in snapshot.occurrences:
            self._ledger.add_occurrence(occurrence)
        self._ledger.wi_by_occurrence = {
            w.operator_id: w.work_item_id
            for w in self._ledger.work_items.values()
            if w.legacy_task_id
            and not self._ledger.is_dynamic_activation(w.activation_id)
        }
        for w in self._ledger.work_items.values():
            if (
                key := self._ledger.occurrence_by_activation.get(w.activation_id)
            ) is not None:
                self._ledger.wi_by_occurrence[key] = w.work_item_id
        self._ledger.control_states = self._keyed_dict(
            "control_states", ((c.key, c) for c in snapshot.control_states)
        )
        self._ledger.selection_candidates = dict.fromkeys(
            key
            for key, state in self._ledger.control_states.items()
            if state.status is ControlStatus.PENDING and state.inputs
        )
        self._ledger.branch_decisions = self._keyed_dict(
            "branch_decisions", ((d.occurrence, d) for d in snapshot.branch_decisions)
        )
        self._ledger.loop_instances = self._keyed_dict(
            "loop_instances", ((i.scope_id, i) for i in snapshot.loop_instances)
        )
        self._ledger.loop_by_occurrence = {
            i.occurrence: i.scope_id for i in snapshot.loop_instances
        }
        self._ledger.iterations = self._keyed_dict(
            "iteration_resolutions",
            (((r.loop, r.iteration), r) for r in snapshot.iteration_resolutions),
        )
        self._ledger.child_contexts = self._keyed_dict(
            "child_contexts", ((c.context_id, c) for c in snapshot.child_contexts)
        )
        self._ledger.wi_by_activation = {
            w.activation_id: w.work_item_id for w in self._ledger.work_items.values()
        }
        self._ledger.succeeded_children = Counter(
            scope_id
            for scope_id, children in self._ledger.children_by_scope.items()
            for child in children.values()
            if (wi_id := self._ledger.wi_by_activation.get(child)) is not None
            and self._ledger.work_items[wi_id].outcome is PublicationOutcome.SUCCESS
        )
        self._publication.slots_by_operator = {}
        self._publication.slots_by_output = {}
        for slot in self._publication.slots.values():
            self._publication.slots_by_operator.setdefault(
                slot.source_operator_id, []
            ).append(slot.slot_key)
            self._publication.slots_by_output.setdefault(slot.output_id, []).append(
                slot.slot_key
            )

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
        # The occurrence owning each scope it opened: a root control through its
        # control activation, an occurrence through its own activation, and an agent
        # occurrence's region through the region's opener.
        self._ledger.scope_occurrence = {}
        self._ledger.fanout_candidates = {}
        for scope in self._ledger.scopes.values():
            owner_act, owner_op = scope.owner_activation_id, scope.owner_operator_id
            if owner_act is None or owner_op is None:
                continue
            owner = self._ledger.activations.get(owner_act)
            if (
                key := self._ledger.occurrence_by_activation.get(owner_act)
            ) is not None:
                self._ledger.bind_scope_occurrence(scope.scope_id, key)
            elif (
                owner is not None
                and owner.kind == "region"
                and owner.parent_activation_id is not None
                and (
                    key := self._ledger.occurrence_by_activation.get(
                        owner.parent_activation_id
                    )
                )
                is not None
            ):
                self._ledger.bind_scope_occurrence(scope.scope_id, key)
            elif self._topology.is_control(owner_op) and (
                owner_act == self._ledger.control_activation(owner_op)
            ):
                self._ledger.bind_scope_occurrence(scope.scope_id, owner_op)
        for instance in self._ledger.loop_instances.values():
            self._ledger.bind_scope_occurrence(instance.scope_id, instance.occurrence)
        # Released scopes are authoritative scope-level state, restored directly rather
        # than re-derived from records: a recursive region's levels share one join/loop
        # operator, so a record could not attribute a release to the right level.
        self._ledger.released_scopes = self._set(
            "released_scopes", snapshot.released_scopes
        )
        self._ledger.active_loops = {
            i.scope_id
            for i in snapshot.loop_instances
            if i.status in (LoopInstanceStatus.OPEN, LoopInstanceStatus.EXITED)
        }
        self._ledger.active_contexts = {
            c.context_id
            for c in snapshot.child_contexts
            if c.scope_id not in self._ledger.released_scopes
        }
        self._failures.failed_regions = self._set(
            "failed_regions", snapshot.failed_regions
        )
        self._failures.failed_scopes = self._set(
            "failed_scopes", snapshot.failed_scopes
        )
        # A ledger stored without failure reasons names each failed work item's own.
        self._failures.failure_reasons = self._keyed_dict(
            "failure_reasons", snapshot.failure_reasons.items()
        )
        self._failures.instance_failure = snapshot.instance_failure
        self._failures.control_failure = snapshot.control_failure
        self._failures.instance_cancelled = snapshot.instance_cancelled
        for wi in self._ledger.work_items.values():
            if wi.outcome is PublicationOutcome.DECLARED_FAILURE and wi.legacy_task_id:
                self._failures.failure_reasons.setdefault(
                    wi.legacy_task_id, wi.failure_reason or DECLARED_FAILURE_REASON
                )
        self._failures.unapplied = dict.fromkeys(self._failures.failure_reasons)
        # A spawn-site denial names no work item; an agent's denied boundary names one
        # and never refuses a later spawn.
        self._authority.denied_spawns = {
            d.operator_id
            for d in self._authority.decisions
            if d.kind is AuthorityDecisionKind.DENIED
            and d.operator_id
            and d.work_item_id is None
        }
        self._settle(snapshot, ordinals)

    def _settle(
        self, snapshot: LedgerSnapshot, ordinals: LedgerOrdinals | None
    ) -> None:
        """Take the restored ledger as what is stored, owing what the restore changed
        of it, or a rewrite when it is not a stored ledger."""
        self._journal.pending.clear()
        self._written_lengths = {name: len(h) for name, h in self._histories.items()}
        self._written_scalars = self._scalars()
        self._acknowledged = None
        if ordinals is None:
            self.owe_rewrite()
            return
        for order in ordinals.values():
            for ordinal in order.values():
                self._journal.seen_ordinal(ordinal)
        for name, tracked in self._keyed.items():
            stored = ordinals.get(name, {})
            for key in tracked:
                if key in stored:
                    tracked.ordinals[key] = stored[key]
                else:
                    tracked.ordinals[key] = self._journal.ordinal()
                    self._journal.mark((name, key))
            for key in stored.keys() - tracked.keys():
                self._journal.mark((name, key))
        if self.verify_changes:
            self._acknowledged = encode_ledger(StoredLedger(snapshot, ordinals))

    def _foundation(self) -> dict[str, Any]:
        return {
            "instance": self._ledger.workflow_instance,
            "root_scope": self._ledger.root_scope,
            "root_grant": self._ledger.root_grant,
            "max_loop_iterations": self._budget.max_loop_iterations,
        }

    def _scalars(self) -> dict[str, Any]:
        return {
            "next_seq": self._ledger.next_seq,
            "instance_failure": self._failures.instance_failure,
            "control_failure": self._failures.control_failure,
            "instance_cancelled": self._failures.instance_cancelled,
        }

    def to_snapshot(self) -> LedgerSnapshot:
        fields: dict[str, Any] = {**self._foundation(), **self._scalars()}
        for name, tracked in self._keyed.items():
            fields[name] = dict(tracked) if name in STRINGS else list(tracked.values())
        for name, history in self._histories.items():
            fields[name] = list(history)
        for name, members in self._sets.items():
            fields[name] = sorted(members)
        return LedgerSnapshot(**fields)

    def image(self) -> dict[str, str]:
        """Every field of the ledger as it stands."""
        return encode_fields(
            self._foundation(),
            {
                name: [
                    (key, tracked.ordinals[key], value)
                    for key, value in tracked.items()
                ]
                for name, tracked in self._keyed.items()
            },
            self._histories,
            self._sets,
            self._scalars(),
        )

    def changes(self) -> LedgerChanges:
        """The fields a write makes to store the ledger as it stands, given what the
        writes that landed stored. Nothing is taken as written until ``written``."""
        captured = self._journal.captured()
        lengths = {name: len(history) for name, history in self._histories.items()}
        scalars = self._scalars()
        if self._rewrite > self._rewritten:
            changes = LedgerChanges(
                self.image(),
                reset=True,
                captured=captured,
                lengths=lengths,
                scalars=scalars,
                rewrite=self._rewrite,
            )
        else:
            fields: dict[str, str] = {}
            deleted: list[str] = []
            for name, key in captured:
                if (tracked := self._keyed.get(name)) is not None:
                    if key in tracked:
                        fields[field_name(name, key)] = keyed_value(
                            tracked.ordinals[key], tracked[key]
                        )
                    else:
                        deleted.append(field_name(name, key))
                elif key in self._sets[name]:
                    fields[member_field(name, key)] = SET_MEMBER
                else:
                    deleted.append(member_field(name, key))
            for name, history in self._histories.items():
                for position in range(self._written_lengths[name], len(history)):
                    fields[history_field(name, position)] = history[
                        position
                    ].model_dump_json()
            for name, value in scalars.items():
                if value != self._written_scalars[name]:
                    fields[scalar_field(name)] = scalar_value(value)
            changes = LedgerChanges(
                fields,
                tuple(deleted),
                captured=captured,
                lengths=lengths,
                scalars=scalars,
            )
        if self.verify_changes:
            self._verify(changes)
        return changes

    def written(self, changes: LedgerChanges) -> None:
        """Take a write of ``changes`` as landed, keeping each change made since they
        were captured."""
        self._journal.clear(changes.captured)
        for name, length in changes.lengths.items():
            self._written_lengths[name] = max(self._written_lengths[name], length)
        self._written_scalars.update(changes.scalars)
        self._rewritten = max(self._rewritten, changes.rewrite)
        if self.verify_changes:
            self._acknowledged = _applied(self._acknowledged or {}, changes)

    def owe_rewrite(self) -> None:
        """Owe a rewrite of the whole ledger with the next write."""
        self._rewrite += 1

    def _verify(self, changes: LedgerChanges) -> None:
        failures = SnapshotCodec.verify_failures
        for name, tracked in self._keyed.items():
            if name in STRINGS:
                continue
            key_of = KEYED[name][1]
            failures.extend(
                f"{name} holds {key_of(value)!r} under {key!r}"
                for key, value in tracked.items()
                if key_of(value) != key
            )
        if self._acknowledged is None and not changes.reset:
            failures.append("a write of a ledger no write stored carries no rewrite")
            return
        expected = _applied(self._acknowledged or {}, changes)
        image = self.image()
        if expected != image:
            differing = sorted(
                name
                for name in expected.keys() | image.keys()
                if expected.get(name) != image.get(name)
            )
            failures.append(
                f"a write leaves stored fields that differ: {differing[:10]}"
            )
