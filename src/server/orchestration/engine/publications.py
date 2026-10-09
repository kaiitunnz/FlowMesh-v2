"""Result-slot publications of one workflow instance."""

from collections.abc import Iterable

from ...task.v2.representations.results import CardinalityKind
from ..state import (
    Activation,
    PublicationOutcome,
    ResultPublication,
    ResultSlot,
    ValueMember,
    ValueRef,
    WorkItemStatus,
    slot_identity,
)
from .ledger import OrchestrationLedger
from .topology import PlanTopology


class PublicationLedger:
    """Holds the instance's result slots and their publications, and publishes to
    them."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self.slots: dict[str, ResultSlot] = {}
        self.publications: dict[str, ResultPublication] = {}
        self.slots_by_operator: dict[str, list[str]] = {}
        self.slots_by_output: dict[str, list[str]] = {}

    def publish(
        self, operator_id: str, outcome: PublicationOutcome, value_ref: ValueRef | None
    ) -> None:
        for slot_key in self.slots_by_operator.get(operator_id, ()):
            self.write_publication(self.slots[slot_key], outcome, value_ref)

    def publish_keyed(
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
            self.write_publication(
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

    def publish_members(self, operator_id: str, members: Iterable[ValueMember]) -> None:
        """Publish each member of an aggregate an operator's keyed collection
        declares, keyed by member."""
        decls = [
            decl
            for decl in self._topology.bundle.template.result_declarations
            if decl.source_ref == operator_id
            and decl.cardinality is CardinalityKind.KEYED_COLLECTION
        ]
        for member in members:
            for decl in decls:
                self.write_publication(
                    ResultSlot(
                        instance_id=self._ledger.workflow_instance.instance_id,
                        output_id=decl.output_id,
                        source_operator_id=operator_id,
                        logical_key=member.key,
                    ),
                    member.outcome,
                    member.value_ref,
                )

    def write_publication(
        self, slot: ResultSlot, outcome: PublicationOutcome, value_ref: ValueRef | None
    ) -> None:
        if slot.slot_key in self.publications:
            return
        if slot.slot_key not in self.slots:
            self.slots_by_output.setdefault(slot.output_id, []).append(slot.slot_key)
        self.slots[slot.slot_key] = slot.model_copy(update={"published": True})
        self.publications[slot.slot_key] = ResultPublication(
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

    def output_publication(
        self,
        output_id: str,
        scope_id: str | None = None,
        logical_key: str | None = None,
        sequence: int | None = None,
    ) -> ResultPublication | None:
        """The terminal publication of exactly one slot, if it has one."""
        return self.publications.get(
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
        return [self.slots[key] for key in self.slots_by_output.get(output_id, ())]

    def output_slot(
        self,
        output_id: str,
        scope_id: str | None = None,
        logical_key: str | None = None,
        sequence: int | None = None,
    ) -> ResultSlot | None:
        """Exactly one slot of a declared output, if it holds one."""
        return self.slots.get(
            slot_identity(
                self._ledger.workflow_instance.instance_id,
                output_id,
                scope_id,
                logical_key,
                sequence,
            )
        )

    def resolve_legacy_task(self, task_id: str) -> ResultPublication | None:
        """Resolve a legacy task id's induced output slot (compatibility adapter)."""
        return self.output_publication(f"legacy:{task_id}")

    def legacy_task_value(
        self, task_id: str
    ) -> tuple[PublicationOutcome, ValueRef | None] | None:
        """The settled value a legacy task id reads as, or None while it is
        unsettled."""
        if (publication := self.resolve_legacy_task(task_id)) is not None:
            return publication.outcome, publication.value_ref
        wi = self._ledger.work_item_for_task(task_id)
        if wi is None or wi.status is not WorkItemStatus.SETTLED or wi.outcome is None:
            return None
        return wi.outcome, wi.value_ref
