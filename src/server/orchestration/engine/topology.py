"""Immutable lookups over a compiled workflow plan's operators and edges."""

from ...task.v2.representations.bundle import PersistedV2Workflow
from ...task.v2.representations.operators import (
    AgentOperator,
    EffectClass,
    LeafOperator,
    LeafProfile,
    LogicalOperator,
    OperatorKind,
    RecoveryClass,
    SpawnRegion,
)
from ...task.v2.representations.results import ResultDeclaration

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


def _effect_recovery(op: LogicalOperator | None) -> tuple[EffectClass, RecoveryClass]:
    """A dispatchable operator's effect/recovery: a leaf's profile, else pure/recompute.

    An agent episode is itself pure and recomputable; its mediated effects flow through
    boundary events rather than the episode's own effect class.
    """
    if isinstance(op, LeafOperator):
        return op.profile.effect, op.profile.recovery
    return EffectClass.PURE, RecoveryClass.RECOMPUTE


class PlanTopology:
    """Answers operator, edge and region-structure questions about one compiled plan.

    ``operators`` is read on every lookup, never copied.
    """

    def __init__(self, bundle: PersistedV2Workflow) -> None:
        self.bundle = bundle
        self.operators: dict[str, LogicalOperator] = {
            op.operator_id: op for op in bundle.template.operators
        }
        self.profiles: dict[str, LeafProfile] = {
            op.operator_id: op.profile
            for op in bundle.template.operators
            if isinstance(op, LeafOperator)
        }
        self.replay = {
            b.source_ref: b.replay_contract
            for b in bundle.template.effect_boundaries
            if b.source_ref
        }
        self.child_templates = {
            op.child_template_ref
            for op in bundle.template.operators
            if isinstance(op, SpawnRegion) and op.child_template_ref
        }
        # Spawn regions an agent selects by role: entered through the agent-request
        # path, not auto-fired at the root like a producer-driven region.
        self.agent_region_spawns = {
            ref.spawn_ref
            for op in bundle.template.operators
            if isinstance(op, AgentOperator)
            for ref in op.child_region_refs
        }
        self.forward = self._build_topology()

    def _build_topology(self) -> dict[str, list[str]]:
        """Forward successor edges, excluding feedback and spawn->join binding edges."""
        forward: dict[str, list[str]] = {op: [] for op in self.operators}
        for edge in self.bundle.template.edges:
            if edge.feedback or edge.from_op not in forward:
                continue
            if self._is_spawn_join_edge(edge.from_op, edge.to_op):
                continue
            forward[edge.from_op].append(edge.to_op)
        return forward

    def _is_spawn_join_edge(self, from_op: str, to_op: str) -> bool:
        return (
            self.kind(from_op) is OperatorKind.SPAWN
            and self.kind(to_op) is OperatorKind.JOIN
        )

    def kind(self, operator_id: str) -> OperatorKind | None:
        op = self.operators.get(operator_id)
        return op.kind if op else None

    def is_control(self, operator_id: str) -> bool:
        return self.kind(operator_id) in _CONTROL_KINDS

    def agent_region_op(self, op: AgentOperator, role: str | None) -> str | None:
        """The spawn region operator a declared role selects, or None if undeclared."""
        if role is None:
            return None
        return next(
            (ref.spawn_ref for ref in op.child_region_refs if ref.name == role), None
        )

    def agent_entry_port(self, operator_id: str) -> str | None:
        """The single declared input port of an agent child body, or None.

        A spawn child agent declares exactly one input port (compile-enforced); a leaf
        child or an agent with no declared input has no entry port.
        """
        op = self.operators.get(operator_id)
        if not isinstance(op, AgentOperator) or not op.declared_input_ports:
            return None
        return op.declared_input_ports[0]

    def region_owner(self, spawn_op: str) -> str | None:
        """The agent declaring ``spawn_op`` as one of its child regions, if any."""
        return next(
            (
                op.operator_id
                for op in self.operators.values()
                if isinstance(op, AgentOperator)
                and any(ref.spawn_ref == spawn_op for ref in op.child_region_refs)
            ),
            None,
        )

    def requested_interface(self, operator_id: str) -> str | None:
        profile = self.profiles.get(operator_id)
        if profile is not None and profile.effect is EffectClass.EXTERNAL_EFFECT:
            return operator_id
        return None

    def published_outputs(self) -> list[tuple[str, ResultDeclaration]]:
        """Each published declaration with the public name it was authored under."""
        return self.bundle.template.published_outputs()

    def spawn_successor(self, operator_id: str) -> str | None:
        """The spawn region an operator feeds via a forward edge, if any."""
        for successor in self.forward.get(operator_id, ()):
            if self.kind(successor) is OperatorKind.SPAWN:
                return successor
        return None

    def child_template_of(self, spawn_op: str) -> str | None:
        """The operator id of a spawn's child template, if it declares one."""
        op = self.operators.get(spawn_op)
        return op.child_template_ref if isinstance(op, SpawnRegion) else None

    def join_for_spawn(self, spawn_op: str) -> str | None:
        for edge in self.bundle.template.edges:
            if edge.from_op == spawn_op and self.kind(edge.to_op) is OperatorKind.JOIN:
                return edge.to_op
        return None

    def edge_from_port(self, from_op: str, to_op: str) -> str | None:
        for edge in self.bundle.template.edges:
            if edge.from_op == from_op and edge.to_op == to_op:
                return edge.from_port
        return None
