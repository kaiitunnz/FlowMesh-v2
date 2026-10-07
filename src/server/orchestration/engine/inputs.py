"""Accepted agent inputs of one workflow instance."""

from ...task.v2.representations.operators import OperatorKind
from ..state import AcceptedInput, PublicationOutcome, ValueRef, WorkItemStatus
from ..tool_dispatch import AgentInputPlan, InputMemberPlan, InputPortPlan
from .ledger import OrchestrationLedger
from .topology import PlanTopology


class AcceptedInputLedger:
    """Holds the inputs each activation accepted and plans an agent's input ports
    from them."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self.accepted_inputs: list[AcceptedInput] = []
        self.accepted_by_activation: dict[str, list[AcceptedInput]] = {}

    def record_accepted_input(self, accepted: AcceptedInput) -> None:
        """Record a durable accepted input on an agent's target port (idempotent)."""
        existing = self.accepted_by_activation.setdefault(accepted.activation_id, [])
        if any(a.target_port == accepted.target_port for a in existing):
            return
        self.accepted_inputs.append(accepted)
        existing.append(accepted)

    def accepted_inputs_for(self, activation_id: str) -> tuple[AcceptedInput, ...]:
        """The recorded accepted inputs for one activation, ordered by ordinal."""
        return tuple(
            sorted(
                self.accepted_by_activation.get(activation_id, ()),
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

    def agent_input_plan(self, task_id: str) -> AgentInputPlan | None:
        """The engine's per-port input membership for an agent, resolved by the
        runtime."""
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

    def child_input(self, task_id: str) -> ValueRef | None:
        """The child-init input a spawned child task runs on, if it has one."""
        wi = self._ledger.work_item_for_task(task_id)
        return wi.child_input if wi is not None else None
