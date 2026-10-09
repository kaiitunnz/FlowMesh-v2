"""Normalize ``graph.templates`` into the finite region definitions loops and
spawned or called children enter."""

from typing import Any

from ...parser import ParsedDefinition, ParsedWorkflow
from ..representations.operators import (
    LoopContextRegion,
    Port,
    PortKind,
    SpawnRegion,
)
from ..representations.template import (
    BOUNDARY_NODES,
    BoundaryKind,
    DefinitionKind,
    DefinitionPort,
    DependencyUse,
    EntryRole,
    RegionDefinition,
    TemplateEdge,
)
from .diagnostics import compile_error
from .project import LoweringAccumulator, build_value_ops

_RETURN_KINDS = {
    node: kind
    for kind, node in BOUNDARY_NODES.items()
    if kind is not BoundaryKind.ENTRY
}
_ROLES = {
    DefinitionKind.LOOP_BODY: frozenset({EntryRole.CARRIED, EntryRole.INVARIANT}),
    DefinitionKind.CHILD: frozenset({EntryRole.PARAM, EntryRole.CAPTURE}),
}
_RETURNS = {
    DefinitionKind.LOOP_BODY: frozenset({BoundaryKind.FEEDBACK, BoundaryKind.EGRESS}),
    DefinitionKind.CHILD: frozenset({BoundaryKind.RETURN}),
}


def lower_definitions(parsed: ParsedWorkflow, acc: LoweringAccumulator) -> None:
    """Build one region definition per ``graph.templates`` entry.

    A definition is a loop body when a loop's ``body_ref`` names it and a child when a
    spawn or call does; its members are the operators declared inside it, with any
    region an agent member declares. Its return routes join the edge catalog as
    boundary edges, and every forward edge is scoped to the definition its consumer
    belongs to.
    """
    for definition in parsed.definitions:
        acc.definitions.append(_lower_definition(definition, parsed, acc))
    owner = {
        member: definition.definition_id
        for definition in acc.definitions
        for member in definition.members
    }
    acc.edges = [
        (
            edge.model_copy(update={"definition": owner.get(edge.to_op)})
            if edge.is_forward
            else edge
        )
        for edge in acc.edges
    ]


def _lower_definition(
    definition: ParsedDefinition, parsed: ParsedWorkflow, acc: LoweringAccumulator
) -> RegionDefinition:
    name = definition.name
    kind = _definition_kind(name, acc)
    members = _members(name, parsed, acc)
    inputs = _definition_inputs(definition, kind)
    returns = _definition_returns(definition, kind)
    input_names = {port.name for port in inputs}
    for entry in acc.edges:
        if (
            entry.boundary is BoundaryKind.ENTRY
            and entry.definition == name
            and entry.from_port not in input_names
        ):
            raise compile_error(
                "definition.unknown-input",
                f"$ingress port {entry.from_port!r} is not an input of template "
                f"{name!r}",
                name,
                "graph_node",
            )
    value_ops = build_value_ops(parsed)
    return_ports = (
        {port.name for port in returns}
        if kind is DefinitionKind.CHILD
        else {port.name for port in inputs if port.role is EntryRole.CARRIED}
    )
    for index, edge in enumerate(definition.edges):
        return_kind = _RETURN_KINDS[edge.target]
        if return_kind not in _RETURNS[kind]:
            raise compile_error(
                "definition.bad-return",
                f"a {kind.value.replace('_', ' ')} template returns through "
                + " or ".join(
                    sorted(t for t, k in _RETURN_KINDS.items() if k in _RETURNS[kind])
                )
                + f", not {edge.target}",
                name,
                "graph_node",
            )
        if edge.target_port not in return_ports:
            raise compile_error(
                "definition.bad-return",
                f"{edge.target} port {edge.target_port!r} is not a "
                + (
                    "declared return"
                    if kind is DefinitionKind.CHILD
                    else "carried input"
                )
                + f" of template {name!r}",
                name,
                "graph_node",
            )
        acc.edges.append(
            TemplateEdge(
                from_op=value_ops.get(edge.source, edge.source),
                to_op=edge.target,
                from_port=edge.port,
                to_port=edge.target_port,
                edge_id=f"{name}{edge.target}#{index}",
                use=DependencyUse.VALUE_REQUIRED,
                projection=edge.project,
                boundary=return_kind,
                definition=name,
            )
        )
    return RegionDefinition(
        definition_id=name,
        kind=kind,
        source_id=name,
        members=members,
        inputs=inputs,
        returns=returns,
    )


def _definition_kind(name: str, acc: LoweringAccumulator) -> DefinitionKind:
    loops = [
        op.operator_id
        for op in acc.operators
        if isinstance(op, LoopContextRegion) and op.body_ref == name
    ]
    spawns = [
        op.operator_id
        for op in acc.operators
        if isinstance(op, SpawnRegion) and op.child_definition_ref == name
    ]
    if loops and spawns:
        raise compile_error(
            "definition.mixed-use",
            f"template {name!r} is both a loop body and a child; a template is one",
            name,
            "graph_node",
        )
    if len(loops) > 1:
        raise compile_error(
            "definition.shared-body",
            f"template {name!r} is the body of {len(loops)} loops; a loop body "
            "belongs to one loop",
            name,
            "graph_node",
        )
    if not loops and not spawns:
        raise compile_error(
            "definition.orphan",
            f"no loop, spawn or call enters template {name!r}",
            name,
            "graph_node",
        )
    return DefinitionKind.LOOP_BODY if loops else DefinitionKind.CHILD


def _members(
    name: str, parsed: ParsedWorkflow, acc: LoweringAccumulator
) -> tuple[str, ...]:
    """The operators declared in a template, with the regions its agents declare."""
    declared = {task.task_id for task in parsed.tasks if task.definition == name} | {
        region.name for region in parsed.regions if region.definition == name
    }
    declared |= {build_value_ops(parsed).get(op_id, op_id) for op_id in declared}
    synthesized = {
        entry.logical_ref
        for entry in acc.source_map
        if entry.source_kind == "region" and entry.source_id in declared
    }
    ids = declared | synthesized
    return tuple(op.operator_id for op in acc.operators if op.operator_id in ids)


def _definition_inputs(
    definition: ParsedDefinition, kind: DefinitionKind
) -> tuple[DefinitionPort, ...]:
    ports: list[DefinitionPort] = []
    for raw in definition.inputs:
        entry = _port_mapping(raw, definition.name, "inputs", {"name", "kind", "role"})
        role = next((r for r in _ROLES[kind] if r.value == entry.get("role")), None)
        if role is None:
            allowed = ", ".join(sorted(r.value for r in _ROLES[kind]))
            raise compile_error(
                "definition.bad-input",
                f"input {entry['name']!r} of template {definition.name!r} declares "
                f"role {entry.get('role')!r}; its roles are {allowed}",
                definition.name,
                "graph_node",
            )
        ports.append(
            DefinitionPort(
                name=str(entry["name"]), kind=_port_kind(entry, definition), role=role
            )
        )
    if kind is DefinitionKind.CHILD and (
        sum(port.role is EntryRole.PARAM for port in ports) != 1
    ):
        raise compile_error(
            "definition.param",
            f"child template {definition.name!r} declares exactly one param input, "
            "which its spawned element or call input binds",
            definition.name,
            "graph_node",
        )
    return tuple(ports)


def _definition_returns(
    definition: ParsedDefinition, kind: DefinitionKind
) -> tuple[Port, ...]:
    if kind is DefinitionKind.LOOP_BODY:
        if definition.returns:
            raise compile_error(
                "definition.bad-return",
                f"loop body {definition.name!r} returns through its loop's carried "
                "ports; it declares no returns",
                definition.name,
                "graph_node",
            )
        return ()
    ports = tuple(
        Port(name=str(entry["name"]), kind=_port_kind(entry, definition))
        for entry in (
            _port_mapping(raw, definition.name, "returns", {"name", "kind"})
            for raw in definition.returns
        )
    )
    if not ports:
        raise compile_error(
            "definition.no-return",
            f"child template {definition.name!r} declares at least one return",
            definition.name,
            "graph_node",
        )
    return ports


def _port_mapping(
    raw: Any, template: str, key: str, allowed: set[str]
) -> dict[str, Any]:
    if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
        raise compile_error(
            "definition.bad-port",
            f"each {key} entry of template {template!r} is a mapping with a name",
            template,
            "graph_node",
        )
    if unknown := sorted(set(raw) - allowed):
        raise compile_error(
            "definition.bad-port",
            f"a {key} entry of template {template!r} declares unknown field(s) "
            f"{', '.join(unknown)}",
            template,
            "graph_node",
        )
    return raw


def _port_kind(entry: dict[str, Any], definition: ParsedDefinition) -> PortKind:
    try:
        return PortKind(str(entry.get("kind", PortKind.VALUE.value)))
    except ValueError as exc:
        raise compile_error(
            "definition.bad-port",
            f"unknown port kind {entry.get('kind')!r} in template {definition.name!r}",
            definition.name,
            "graph_node",
        ) from exc
