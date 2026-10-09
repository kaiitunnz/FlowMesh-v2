"""Encodes and restores one workflow instance's ledger snapshot."""

from collections import Counter
from collections.abc import Iterable

from ..guardrails import ScopeBudget
from ..state import (
    AuthorityDecisionKind,
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


class SnapshotCodec:
    """Captures the instance's durable state as a ledger snapshot, and restores a
    stored one into the owners of that state in dependency order."""

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

    def restore(self, snapshot: LedgerSnapshot) -> None:
        self._ledger.scopes = {}
        self._ledger.subscopes = {}
        for scope in snapshot.scopes:
            self._ledger.add_scope(scope)
        self._ledger.scopes.setdefault(
            self._ledger.root_scope.scope_id, self._ledger.root_scope
        )
        self._ledger.activations = {}
        self._ledger.children_by_scope = {}
        self._ledger.scope_population = Counter()
        self._ledger.scope_children = Counter()
        self._ledger.dynamic_activations = 0
        for activation in snapshot.activations:
            self._ledger.add_activation(activation)
        self._ledger.work_items = {w.work_item_id: w for w in snapshot.work_items}
        self._ledger.continuations = {c.work_item_id: c for c in snapshot.continuations}
        self._ledger.records = list(snapshot.records)
        self._inputs.accepted_inputs = list(snapshot.accepted_inputs)
        self._inputs.accepted_by_activation = {}
        for accepted in self._inputs.accepted_inputs:
            self._inputs.accepted_by_activation.setdefault(
                accepted.activation_id, []
            ).append(accepted)
        self._ledger.region_aggregates = list(snapshot.region_aggregates)
        self._ledger.aggregate_by_join = {}
        # A stored ledger may hold a nested level's aggregate after its root level's;
        # the root level's is the one delivered downstream.
        for aggregate in self._ledger.region_aggregates:
            join_op = aggregate.occurrence or aggregate.join_operator_id
            if join_op not in self._ledger.aggregate_by_join or not any(
                (act := self._ledger.activations.get(member.child_activation_id))
                is not None
                and not self._ledger.root_level(act.scope_id)
                for member in aggregate.members
            ):
                self._ledger.aggregate_by_join[join_op] = aggregate
        self._ledger.invocations = {i.invocation_id: i for i in snapshot.invocations}
        self._ledger.attempts = {a.attempt_id: a for a in snapshot.attempts}
        self._embodiments.embodiment_selections = {
            sel.work_item_id: sel for sel in snapshot.embodiment_selections
        }
        self._embodiments.input_resolutions = {
            res.work_item_id: res for res in snapshot.input_resolutions
        }
        self._embodiments.input_preparations = {
            prep.work_item_id: prep for prep in snapshot.input_preparations
        }
        self._attempt_lifecycle.receipts = {
            r.invocation_id: r for r in snapshot.effect_receipts
        }
        self._authority.decisions = list(snapshot.authority_decisions)
        self._authority.grants = {g.grant_id: g for g in snapshot.delegated_grants}
        self._ledger.capabilities = {
            (c.scope_id, c.axis): c for c in snapshot.progress_capabilities
        }
        self._publication.slots = {s.slot_key: s for s in snapshot.result_slots}
        self._publication.publications = _rekeyed_publications(
            self._publication.slots.values(), snapshot.result_publications
        )
        self._ledger.trace = list(snapshot.trace)

        # Not persisted: each region activation names the parent and operator that
        # opened it.
        self._ledger.region_openers = {
            (a.parent_activation_id, a.operator_id): a.activation_id
            for a in self._ledger.activations.values()
            if a.kind == "region" and a.parent_activation_id
        }
        self._boundaries.boundary_events = {
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
        self._ledger.occurrences = {}
        self._ledger.occurrences_by_scope = {}
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
        self._ledger.control_states = {c.key: c for c in snapshot.control_states}
        self._ledger.branch_decisions = {
            d.occurrence: d for d in snapshot.branch_decisions
        }
        self._ledger.loop_instances = {i.scope_id: i for i in snapshot.loop_instances}
        self._ledger.loop_by_occurrence = {
            i.occurrence: i.scope_id for i in snapshot.loop_instances
        }
        self._ledger.iterations = {
            (r.loop, r.iteration): r for r in snapshot.iteration_resolutions
        }
        self._ledger.child_contexts = {c.context_id: c for c in snapshot.child_contexts}
        self._ledger.wi_by_activation = {
            w.activation_id: w.work_item_id for w in self._ledger.work_items.values()
        }
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
        # The control occurrence owning each scope it opened: a root control through its
        # control activation, an occurrence through its own activation.
        self._ledger.scope_occurrence = {}
        for scope in self._ledger.scopes.values():
            owner_act, owner_op = scope.owner_activation_id, scope.owner_operator_id
            if owner_act is None or owner_op is None:
                continue
            if (
                key := self._ledger.occurrence_by_activation.get(owner_act)
            ) is not None:
                self._ledger.scope_occurrence[scope.scope_id] = key
            elif self._topology.is_control(owner_op) and (
                owner_act == self._ledger.control_activation(owner_op)
            ):
                self._ledger.scope_occurrence[scope.scope_id] = owner_op
        for instance in self._ledger.loop_instances.values():
            self._ledger.scope_occurrence[instance.scope_id] = instance.occurrence
        # Released scopes are authoritative scope-level state, restored directly rather
        # than re-derived from records: a recursive region's levels share one join/loop
        # operator, so a record could not attribute a release to the right level.
        self._ledger.released_scopes = set(snapshot.released_scopes)
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
        self._failures.failed_regions = set(snapshot.failed_regions)
        self._failures.failed_scopes = set(snapshot.failed_scopes)
        # A ledger stored without failure reasons names each failed work item's own.
        self._failures.failure_reasons = dict(snapshot.failure_reasons)
        for wi in self._ledger.work_items.values():
            if wi.outcome is PublicationOutcome.DECLARED_FAILURE and wi.legacy_task_id:
                self._failures.failure_reasons.setdefault(
                    wi.legacy_task_id, wi.failure_reason or DECLARED_FAILURE_REASON
                )
        # A spawn-site denial names no work item; an agent's denied boundary names one
        # and never refuses a later spawn.
        self._authority.denied_spawns = {
            d.operator_id
            for d in self._authority.decisions
            if d.kind is AuthorityDecisionKind.DENIED
            and d.operator_id
            and d.work_item_id is None
        }

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
            accepted_inputs=list(self._inputs.accepted_inputs),
            region_aggregates=list(self._ledger.region_aggregates),
            invocations=list(self._ledger.invocations.values()),
            attempts=list(self._ledger.attempts.values()),
            embodiment_selections=list(
                self._embodiments.embodiment_selections.values()
            ),
            input_resolutions=list(self._embodiments.input_resolutions.values()),
            input_preparations=list(self._embodiments.input_preparations.values()),
            boundary_events=list(self._boundaries.boundary_events.values()),
            effect_receipts=list(self._attempt_lifecycle.receipts.values()),
            authority_decisions=list(self._authority.decisions),
            delegated_grants=list(self._authority.grants.values()),
            progress_capabilities=list(self._ledger.capabilities.values()),
            result_slots=list(self._publication.slots.values()),
            result_publications=list(self._publication.publications.values()),
            trace=list(self._ledger.trace),
            private_state=self._ledger.private_state.lineages(),
            released_scopes=sorted(self._ledger.released_scopes),
            failed_regions=sorted(self._failures.failed_regions),
            failed_scopes=sorted(self._failures.failed_scopes),
            failure_reasons=dict(self._failures.failure_reasons),
            next_seq=self._ledger.next_seq,
            occurrences=list(self._ledger.occurrences.values()),
            control_states=list(self._ledger.control_states.values()),
            branch_decisions=list(self._ledger.branch_decisions.values()),
            loop_instances=list(self._ledger.loop_instances.values()),
            iteration_resolutions=list(self._ledger.iterations.values()),
            child_contexts=list(self._ledger.child_contexts.values()),
            max_loop_iterations=self._budget.max_loop_iterations,
        )
