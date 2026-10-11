"""Materializing a region definition's operators at one context and time."""

from shared.utils import new_activation_id, new_work_item_id

from ...task.v2.representations.operators import AgentOperator, LeafOperator
from ..state import (
    Activation,
    Continuation,
    DeliveryContext,
    Occurrence,
    WorkItem,
)
from .ledger import OrchestrationLedger, control_key, occurrence_key
from .scopes import ScopeProgress
from .topology import PlanTopology, effect_recovery


class OccurrenceFactory:
    """Creates the tagged occurrences of a definition's members when a loop time or a
    child context enters it.

    Each member gets one occurrence keyed by (operator, context, time) and an
    activation; a leaf or agent also gets a blocked work item. Nothing is dispatched
    here: an occurrence becomes ready only once its inputs resolve.
    """

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        scope_progress: ScopeProgress,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._scope_progress = scope_progress

    def enter(self, definition_id: str, at: DeliveryContext) -> list[str]:
        """Create every member occurrence of a definition; returns their keys.

        The activation budget covers the whole batch before anything is created, so a
        refused entry leaves no partial time or child behind.
        """
        definition = self._topology.definitions[definition_id]
        # A spawn's single-operator child occurs only as that spawn's child, and an
        # agent's child region opens only as its agent spawns into it.
        members = [
            m
            for m in definition.members
            if m not in self._topology.child_templates
            and m not in self._topology.agent_region_spawns
        ]
        self._scope_progress.charge_activations(len(members))
        keys: list[str] = []
        for operator_id in members:
            key = occurrence_key(operator_id, at)
            if key in self._ledger.occurrences:
                keys.append(key)
                continue
            keys.append(key)
            op = self._topology.operators[operator_id]
            activation = Activation(
                activation_id=new_activation_id(),
                instance_id=self._ledger.workflow_instance.instance_id,
                scope_id=at.scope_id,
                operator_id=operator_id,
                kind="occurrence",
            )
            self._ledger.add_activation(activation)
            self._ledger.add_occurrence(
                Occurrence(
                    context_id=at.context_id,
                    scope_id=at.scope_id,
                    time=at.time,
                    key=key,
                    operator_id=operator_id,
                    activation_id=activation.activation_id,
                )
            )
            waiting = frozenset(
                e.from_op for e in self._topology.incoming.get(operator_id, ())
            )
            if not isinstance(op, (LeafOperator, AgentOperator)):
                self._ledger.control_state(key)
                self._ledger.set_continuation(
                    Continuation(work_item_id=control_key(key), waiting_on=waiting)
                )
                continue
            effect, recovery = effect_recovery(op)
            wi = WorkItem(
                work_item_id=new_work_item_id(),
                activation_id=activation.activation_id,
                operator_id=operator_id,
                legacy_task_id=activation.activation_id,
                effect_class=effect,
                recovery=recovery,
                replay_contract=self._topology.replay.get(operator_id),
            )
            self._ledger.add_work_item(wi)
            self._ledger.wi_by_activation[activation.activation_id] = wi.work_item_id
            self._ledger.wi_by_task[wi.legacy_task_id] = wi.work_item_id
            self._ledger.wi_by_occurrence[key] = wi.work_item_id
            self._ledger.set_continuation(
                Continuation(
                    work_item_id=wi.work_item_id,
                    waiting_on=waiting,
                    required_ports=(
                        frozenset(op.declared_input_ports)
                        if isinstance(op, AgentOperator)
                        else frozenset()
                    ),
                )
            )
        self._ledger.emit(
            "definition_entered",
            detail={"definition": definition_id, "context": at.context_id},
        )
        return keys
