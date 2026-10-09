"""Lookups over a compiled workflow plan's operators and edges."""

from ...task.v2.representations.bundle import PersistedV2Workflow
from ...task.v2.representations.operators import (
    AgentOperator,
    EffectClass,
    LeafOperator,
    LeafProfile,
    LogicalOperator,
    LoopContextRegion,
    OperatorKind,
    RecoveryClass,
    SpawnRegion,
    is_spawn_fanout_port,
)
from ...task.v2.representations.results import ResultDeclaration
from ...task.v2.representations.template import (
    RETURN_KINDS,
    BoundaryKind,
    LogicalWorkflowTemplate,
    RegionDefinition,
    TemplateEdge,
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
CONTROL_KINDS = _REGION_KINDS
CHILD_INIT_OPENERS = frozenset({OperatorKind.SPAWN, OperatorKind.AGENT})


def effect_recovery(op: LogicalOperator | None) -> tuple[EffectClass, RecoveryClass]:
    """A dispatchable operator's effect/recovery: a leaf's profile, else pure/recompute.

    An agent episode is itself pure and recomputable; its mediated effects flow through
    boundary events rather than the episode's own effect class.
    """
    if isinstance(op, LeafOperator):
        return op.profile.effect, op.profile.recovery
    return EffectClass.PURE, RecoveryClass.RECOMPUTE


def child_bodies(template: LogicalWorkflowTemplate) -> frozenset[str]:
    """The spawn child templates that run only as spawned children.

    A region whose entry is its enclosing agent is explicit recursion: that agent also
    runs at the root.
    """
    region_owner = {
        ref.spawn_ref: op.operator_id
        for op in template.operators
        if isinstance(op, AgentOperator)
        for ref in op.child_region_refs
    }
    return frozenset(
        op.child_template_ref
        for op in template.operators
        if isinstance(op, SpawnRegion)
        and op.child_template_ref
        and op.child_template_ref != op.operator_id
        and region_owner.get(op.operator_id) != op.child_template_ref
    )


def blueprint_operators(template: LogicalWorkflowTemplate) -> frozenset[str]:
    """The operators whose tasks are made as their work materializes: every region
    definition member and spawn child template."""
    return frozenset(template.definition_of()) | frozenset(
        op.child_template_ref
        for op in template.operators
        if isinstance(op, SpawnRegion) and op.child_template_ref
    )


def materialized_operators(template: LogicalWorkflowTemplate) -> frozenset[str]:
    """The blueprint operators with no work of their own at the root."""
    return frozenset(template.definition_of()) | child_bodies(template)


class PlanTopology:
    """Answers operator, edge and region-structure questions about one compiled plan.

    Every component reads ``operators`` through this object on each lookup and never
    copies it, so a replaced operator is seen everywhere.
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
        self.definitions: dict[str, RegionDefinition] = {
            d.definition_id: d for d in bundle.template.definitions
        }
        # The definition each member runs in; a root operator is absent.
        self.definition_of = bundle.template.definition_of()
        # Each operator whose whole result a root spawn fans out over, with the spawn.
        self.fanout_spawns: dict[str, str] = {}
        for edge in bundle.template.edges:
            if (
                edge.is_forward
                and self.kind(edge.to_op) is OperatorKind.SPAWN
                and edge.to_op not in self.definition_of
                and edge.to_op not in self.agent_region_spawns
                and is_spawn_fanout_port(edge.to_port)
                and not edge.projection
            ):
                self.fanout_spawns.setdefault(edge.from_op, edge.to_op)
        self.incoming: dict[str, list[TemplateEdge]] = {op: [] for op in self.operators}
        self.outgoing: dict[str, list[TemplateEdge]] = {op: [] for op in self.operators}
        # A definition's entry edges by the member they enter, and its return edges
        # by the member they leave from.
        self.entries_into: dict[str, list[TemplateEdge]] = {}
        self.returns_from: dict[str, list[TemplateEdge]] = {}
        for edge in bundle.template.edges:
            if edge.definition is not None and edge.boundary is BoundaryKind.ENTRY:
                self.entries_into.setdefault(edge.to_op, []).append(edge)
            elif edge.definition is not None and edge.boundary in RETURN_KINDS:
                self.returns_from.setdefault(edge.from_op, []).append(edge)
            elif (
                edge.is_forward
                and edge.from_op in self.operators
                and edge.to_op in self.operators
                and not self._is_spawn_join_edge(edge.from_op, edge.to_op)
            ):
                self.incoming[edge.to_op].append(edge)
                self.outgoing[edge.from_op].append(edge)
        # The authored name of each operator, as its source names it.
        self.source_ids = {
            e.logical_ref: e.source_id for e in bundle.template.source_map
        }
        sources = self.source_ids
        # The name a spec reads each operator's value by within its scope: a
        # definition member's name inside its definition, any other operator's authored
        # name; a call's join carries the call's name.
        self.scope_names: dict[str, str] = {
            op: (
                source.removeprefix(f"{definition}/")
                if (definition := self.definition_of.get(op)) is not None
                else source
            )
            for op, source in sources.items()
        }
        for edge in bundle.template.edges:
            if self._is_spawn_join_edge(edge.from_op, edge.to_op) and (
                name := self.scope_names.get(edge.from_op)
            ):
                self.scope_names.setdefault(edge.to_op, name)

    def _build_topology(self) -> dict[str, list[str]]:
        """Forward successor edges, excluding feedback and spawn->join binding edges."""
        forward: dict[str, list[str]] = {op: [] for op in self.operators}
        for edge in self.bundle.template.edges:
            if not edge.is_forward or edge.from_op not in forward:
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
        return self.kind(operator_id) in CONTROL_KINDS

    def agent_region_op(self, op: AgentOperator, role: str | None) -> str | None:
        """The spawn region operator a declared role selects, or None if undeclared."""
        if role is None:
            return None
        return next(
            (ref.spawn_ref for ref in op.child_region_refs if ref.name == role), None
        )

    def agent_entry_port(self, operator_id: str) -> str | None:
        """The single declared input port of an agent child body, or None."""
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

    def fanout_spawn(self, operator_id: str) -> str | None:
        """The root spawn that fans out over an operator's whole result, if any.

        Only the spawn's fan-out input counts: a capture it also reads, or a fan-out
        over a projection of the result, is read for it through its own value.
        """
        return self.fanout_spawns.get(operator_id)

    def child_template_of(self, spawn_op: str) -> str | None:
        """The operator id of a spawn's child template, if it declares one."""
        op = self.operators.get(spawn_op)
        return op.child_template_ref if isinstance(op, SpawnRegion) else None

    def join_for_spawn(self, spawn_op: str) -> str | None:
        for edge in self.bundle.template.edges:
            if edge.from_op == spawn_op and self.kind(edge.to_op) is OperatorKind.JOIN:
                return edge.to_op
        return None

    def loop_of_body(self, definition_id: str) -> str | None:
        """The loop operator whose body a definition is."""
        return next(
            (
                op.operator_id
                for op in self.operators.values()
                if isinstance(op, LoopContextRegion) and op.body_ref == definition_id
            ),
            None,
        )

    def edge_from_port(self, from_op: str, to_op: str) -> str | None:
        for edge in self.bundle.template.edges:
            if edge.from_op == from_op and edge.to_op == to_op:
                return edge.from_port
        return None


def edge_key(edge: TemplateEdge) -> str:
    """A stable identity for an edge."""
    return edge.edge_id
