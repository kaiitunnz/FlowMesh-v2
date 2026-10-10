from dataclasses import replace
from enum import Enum
from typing import Any

from ...parser import (
    INGRESS,
    ParsedDependency,
    ParsedRegion,
    ParsedTask,
    ParsedWorkflow,
)
from ..representations.operators import (
    AgentOperator,
    AuthorityCeiling,
    BoundaryEventKind,
    BoundarySignature,
    BranchRegion,
    ChildRegionRef,
    DeterminismClass,
    EffectClass,
    InputProvenanceKind,
    JoinCompletion,
    JoinPredicate,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    LoopContextRegion,
    MergeCombination,
    MergeRegion,
    Port,
    PortKind,
    RecoveryClass,
    SelectionCase,
    SelectionRule,
    SpawnRegion,
)
from ..representations.plan import PhysicalNode
from ..representations.results import (
    CardinalityKind,
    ReleaseConditionKind,
    ResultDeclaration,
    Visibility,
)
from ..representations.template import (
    DependencyUse,
    ResourceDeclaration,
    SourceMapEntry,
    TemplateEdge,
    ToolDeclaration,
)
from .definitions import lower_definitions
from .diagnostics import compile_error
from .project import (
    LoweringAccumulator,
    agent_region_join_id,
    build_value_ops,
    call_join_id,
    dependency_edge,
)
from .reads import unnamed_projection

# Friendly aliases for the two provenance values authors write in spec.v2.
_PROVENANCE = {
    "live": InputProvenanceKind.LIVE_INPUT,
    "pinned": InputProvenanceKind.EXTERNAL_PINNED,
}


def _enum_or_none[E: Enum](enum_cls: type[E], value: str) -> E | None:
    try:
        return enum_cls(value)
    except ValueError:
        return None


def _str_list(value: Any, name: str, source_kind: str = "region") -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise compile_error(
        "v2.not-string-list",
        f"expected a string or list of strings, got {type(value).__name__}",
        name,
        source_kind,
    )


def _authority(raw: Any, name: str, source_kind: str = "region") -> AuthorityCeiling:
    if not isinstance(raw, dict):
        return AuthorityCeiling()
    return AuthorityCeiling(
        invoke=_str_list(raw.get("invoke"), name, source_kind),
        delegate=_str_list(raw.get("delegate"), name, source_kind),
    )


def lower_frontend_v2(parsed: ParsedWorkflow, acc: LoweringAccumulator) -> None:
    """Normalize v2 frontend constructs into the canonical template form.

    Applies ``spec.v2`` leaf declarations to already-lowered task operators, lowers
    structured regions into canonical operators/ports/regions, and normalizes a legacy
    agent child target into one declared child region.
    Malformed constructs raise :class:`CompileError` with a source location; semantic
    checks are left to the validation passes.
    """
    _apply_leaf_declarations(parsed, acc)
    _lower_regions(parsed, acc)
    normalize_agent_child_regions(acc)
    lower_definitions(parsed, acc)
    _reject_published_children(parsed, acc)


def _reject_published_children(
    parsed: ParsedWorkflow, acc: LoweringAccumulator
) -> None:
    """Refuse a published leaf that runs as a region's child template.

    A child settles into its region, so such a leaf's own slot never publishes; a spawn
    publishes its children through region.result.
    """
    op_to_name = {
        op_id: name
        for scope in {None, *(d.name for d in parsed.definitions)}
        for name, op_id in acc.names(parsed, scope).items()
    }
    children = {
        op.child_template_ref
        for op in acc.operators
        if isinstance(op, SpawnRegion) and op.child_template_ref is not None
    } | {member for d in acc.definitions for member in d.members}
    for decl in acc.result_declarations:
        if decl.visibility is Visibility.PUBLISHED and decl.source_ref in children:
            name = op_to_name.get(decl.source_ref, decl.source_ref)
            raise compile_error(
                "result.published-child",
                f"{name!r} runs as a region's child and publishes nothing of its own; "
                "publish its results through the spawn's region.result",
                name,
                source_kind="task",
            )


def normalize_agent_child_regions(acc: LoweringAccumulator) -> None:
    """Normalize an agent's legacy single child target into one declared region.

    A pure-legacy agent (a ``child_template_ref`` and no ``child_region_refs``) gains a
    matched ``Spawn``/``Join`` pair over that entry, keyed by a role named for the
    entry. The region's per-site ceiling is the agent's delegate face — what it may hand
    to a child — so a child grant stays bounded by the delegate face, and the region
    stays within it. An agent declaring both forms is left untouched for validation.
    """
    for idx, op in enumerate(acc.operators):
        if not isinstance(op, AgentOperator) or op.child_template_ref is None:
            continue
        if op.child_region_refs:
            continue
        entry = op.child_template_ref
        spawn_id = f"{op.operator_id}:child"
        join_id = f"{spawn_id}:join"
        delegate = op.authority.delegate
        spawn = SpawnRegion(
            operator_id=spawn_id,
            source_ref=op.source_ref,
            outputs=(Port(name="children"),),
            child_template_ref=entry,
            authority=AuthorityCeiling(invoke=delegate, delegate=delegate),
        )
        join = JoinRegion(
            operator_id=join_id,
            source_ref=op.source_ref,
            inputs=(Port(name="children"),),
            outputs=(Port(name="out"),),
            completion=JoinCompletion.ALL_SETTLED,
        )
        acc.operators[idx] = op.model_copy(
            update={
                "child_region_refs": (ChildRegionRef(name=entry, spawn_ref=spawn_id),),
                "child_template_ref": None,
            }
        )
        _add_synth_operator(spawn, op.operator_id, acc)
        _add_synth_operator(join, op.operator_id, acc)
        acc.edges.append(membership_edge(spawn_id, join_id))


def membership_edge(spawn_id: str, join_id: str) -> TemplateEdge:
    """The edge by which a join collects its spawn's children: the join runs only
    when the spawn does."""
    return TemplateEdge(
        from_op=spawn_id,
        to_op=join_id,
        edge_id=f"{join_id}#members",
        use=DependencyUse.ROUTE_REQUIRED,
    )


def _add_synth_operator(
    op: LogicalOperator, source_id: str, acc: LoweringAccumulator
) -> None:
    acc.operators.append(op)
    acc.source_map.append(
        SourceMapEntry(
            logical_ref=op.operator_id, source_kind="region", source_id=source_id
        )
    )
    acc.nodes.append(
        PhysicalNode(
            node_id=f"phys:{op.operator_id}",
            source_ref=op.operator_id,
            logical_ref=op.operator_id,
        )
    )


def _apply_leaf_declarations(parsed: ParsedWorkflow, acc: LoweringAccumulator) -> None:
    by_id = {op.operator_id: idx for idx, op in enumerate(acc.operators)}
    for task in parsed.tasks:
        if not task.v2:
            continue
        idx = by_id.get(task.task_id)
        if idx is None:
            continue
        op = acc.operators[idx]
        name_to_op = acc.names(parsed, task.definition)
        acc.operators[idx] = _apply_one(task, op, name_to_op, acc)


def _apply_one(
    task: ParsedTask,
    op: LogicalOperator,
    name_to_op: dict[str, str],
    acc: LoweringAccumulator,
) -> LogicalOperator:
    v2 = task.v2 or {}
    name = task.graph_node_name or task.local_name or task.task_id

    tools = v2.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise compile_error(
                "v2.tools-not-list", "spec.v2.tools must be a list", name, "graph_node"
            )
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                raise compile_error(
                    "v2.tool-no-name",
                    "each spec.v2.tool needs a name",
                    name,
                    "graph_node",
                )
            acc.tool_declarations.append(
                ToolDeclaration(
                    name=str(tool["name"]),
                    interface=(
                        str(iface) if (iface := tool.get("interface")) else None
                    ),
                    authority_ref=(
                        str(tool["authority_ref"])
                        if tool.get("authority_ref")
                        else None
                    ),
                )
            )

    for resource in v2.get("resources", []) or []:
        if isinstance(resource, dict) and resource.get("name"):
            acc.resource_declarations.append(
                ResourceDeclaration(
                    name=str(resource["name"]),
                    kind=str(k) if (k := resource.get("kind")) else None,
                )
            )

    _apply_result_visibility(v2.get("result"), op.operator_id, acc)

    if isinstance(op, AgentOperator):
        return _apply_agent_child_regions(
            _apply_agent_inputs(_apply_agent_v2(op, v2, name), v2),
            v2,
            name_to_op,
            acc,
        )
    if isinstance(op, LeafOperator):
        return _apply_leaf_v2(op, v2, name)
    raise compile_error(
        "v2.unsupported-operator",
        "spec.v2 applies only to task/agent leaves",
        name,
        "graph_node",
    )


def _apply_agent_child_regions(
    op: AgentOperator,
    v2: dict[str, Any],
    name_to_op: dict[str, str],
    acc: LoweringAccumulator,
) -> AgentOperator:
    """Build one declared child region per ``v2.child`` entry, keyed by the child name.

    Each entry synthesizes a matched Spawn/Join pair over the resolved entry operator
    and exposes it as a ``ChildRegionRef`` whose role is the author-facing child name,
    so a ``spawn_agent`` selects the region by name without knowing the compiled ids.
    """
    children = v2.get("child")
    if isinstance(children, (str, dict)):
        children = [children]
    if not isinstance(children, list):
        return op
    refs: list[ChildRegionRef] = list(op.child_region_refs)
    for child in children:
        # A child is a bare role name (empty region ceiling, fail-closed) or a mapping
        # {name, authority} that declares the region's per-site invoke/delegate ceiling.
        if isinstance(child, dict):
            role = str(child.get("name"))
            ceiling = _authority(child.get("authority"), role, "graph_node")
        else:
            role, ceiling = str(child), AuthorityCeiling()
        entry_op = name_to_op.get(role, role)
        spawn_id = f"{op.operator_id}:{role}:spawn"
        join_id = agent_region_join_id(op.operator_id, role)
        spawn = SpawnRegion(
            operator_id=spawn_id,
            source_ref=op.source_ref,
            outputs=(Port(name="children"),),
            child_template_ref=entry_op,
            authority=ceiling,
        )
        join = JoinRegion(
            operator_id=join_id,
            source_ref=op.source_ref,
            inputs=(Port(name="children"),),
            outputs=(Port(name="out"),),
            completion=JoinCompletion.ALL_SETTLED,
        )
        _add_synth_operator(spawn, op.operator_id, acc)
        _add_synth_operator(join, op.operator_id, acc)
        acc.edges.append(membership_edge(spawn_id, join_id))
        refs.append(ChildRegionRef(name=role, spawn_ref=spawn_id))
    return op.model_copy(update={"child_region_refs": tuple(refs)})


def _apply_agent_inputs(op: AgentOperator, v2: dict[str, Any]) -> AgentOperator:
    """Declare an agent's delivered input ports and their producer bindings.

    Each ``spec.v2.inputs`` entry adds a named input port delivered on the agent's first
    turn. A ``{name, from}`` entry is bound like a ``dependsOn`` entry naming the
    producer and the input; a ``{name, from, region}`` entry binds the parent agent's
    child-region join aggregate (its region output) — dynamically-spawned children
    merged downstream, not mid-run; a bare ``{name}`` entry (a spawn child's entry port)
    is filled by the spawned element. Declaring inputs is opt-in — an agent with none
    keeps ordering-only deps. The port set augments the lowered ports so model/ordering
    ports stay.
    """
    inputs = v2.get("inputs")
    if inputs is None:
        return op
    if not isinstance(inputs, list):
        raise compile_error(
            "v2.inputs-not-list", "spec.v2.inputs must be a list", op.source_ref
        )
    ports: list[Port] = list(op.inputs)
    declared: list[str] = []
    for entry in inputs:
        if isinstance(entry, str):
            port_name = entry
        elif isinstance(entry, dict) and entry.get("name"):
            port_name = str(entry["name"])
        else:
            raise compile_error(
                "v2.bad-input-port",
                "each spec.v2.input is a name or a {name, from} mapping",
                op.source_ref,
            )
        declared.append(port_name)
        if not any(port.name == port_name for port in ports):
            ports.append(Port(name=port_name))
    return op.model_copy(
        update={
            "inputs": tuple(ports),
            "declared_input_ports": tuple(
                dict.fromkeys((*declared, *op.declared_input_ports))
            ),
        }
    )


def _apply_agent_v2(op: AgentOperator, v2: dict[str, Any], name: str) -> AgentOperator:
    updates: dict[str, Any] = {}
    if "authority" in v2:
        updates["authority"] = _authority(v2["authority"], name, "graph_node")
    boundary = v2.get("boundary")
    if boundary is not None:
        events = []
        for event in _str_list(boundary, name, "graph_node"):
            member = _enum_or_none(BoundaryEventKind, event)
            if member is None:
                raise compile_error(
                    "v2.bad-boundary-event",
                    f"unknown boundary event {event!r}",
                    name,
                    "graph_node",
                )
            events.append(member)
        updates["boundary"] = BoundarySignature(events=tuple(events))
    return op.model_copy(update=updates) if updates else op


def _apply_leaf_v2(op: LeafOperator, v2: dict[str, Any], name: str) -> LeafOperator:
    if "authority" in v2 or "tools" in v2 or "boundary" in v2:
        raise compile_error(
            "v2.authority-on-leaf",
            "authority/tools/boundary apply only to agent leaves",
            name,
            "graph_node",
        )
    profile = op.profile
    overrides: dict[str, Any] = {}
    if "provenance" in v2:
        member = _PROVENANCE.get(str(v2["provenance"]))
        if member is None:
            raise compile_error(
                "v2.bad-provenance",
                f"unknown provenance {v2['provenance']!r}",
                name,
                "graph_node",
            )
        overrides["input_provenance"] = member
    for key, enum_cls, field in (
        ("determinism", DeterminismClass, "determinism"),
        ("effect", EffectClass, "effect"),
        ("recovery", RecoveryClass, "recovery"),
    ):
        if key in v2:
            profile_value = _enum_or_none(enum_cls, str(v2[key]))
            if profile_value is None:
                raise compile_error(
                    "v2.bad-profile",
                    f"unknown {key} {v2[key]!r}",
                    name,
                    "graph_node",
                )
            overrides[field] = profile_value
    if not overrides:
        return op
    return op.model_copy(update={"profile": profile.model_copy(update=overrides)})


def _apply_result_visibility(
    result: Any, operator_id: str, acc: LoweringAccumulator
) -> None:
    if not isinstance(result, dict):
        return
    visibility = result.get("visibility")
    if visibility != Visibility.PUBLISHED.value:
        return
    for idx, decl in enumerate(acc.result_declarations):
        if decl.source_ref == operator_id:
            acc.result_declarations[idx] = decl.model_copy(
                update={"visibility": Visibility.PUBLISHED}
            )


_PUBLISHING_KINDS = frozenset({"spawn", "merge", "join", "loop"})
_REGION_KEYS = {
    "branch": frozenset({"kind", "inputs", "outputs", "selection", "forward"}),
    "loop": frozenset(
        {"kind", "body_ref", "loop_coordinate", "carried", "invariants", "result"}
    ),
}
_EARLY_COMPLETIONS = frozenset(
    {JoinCompletion.ANY, JoinCompletion.FIRST_K, JoinCompletion.PREDICATE}
)


def _lower_regions(parsed: ParsedWorkflow, acc: LoweringAccumulator) -> None:
    value_ops = build_value_ops(parsed)
    definitions = {definition.name for definition in parsed.definitions}
    names = {
        scope: acc.names(parsed, scope)
        for scope in {None, *definitions, *(r.definition for r in parsed.regions)}
    }
    region_kinds = {
        region.name: str(region.region.get("kind", "")).strip()
        for region in parsed.regions
    }
    for region in parsed.regions:
        _lower_region(
            region,
            names[region.definition],
            value_ops,
            definitions,
            region_kinds,
            acc,
        )


def _lower_region(
    region: ParsedRegion,
    name_to_op: dict[str, str],
    value_ops: dict[str, str],
    definitions: set[str],
    region_kinds: dict[str, str],
    acc: LoweringAccumulator,
) -> None:
    kind = str(region.region.get("kind", "")).strip()
    name = region.authored_name
    has_input = bool(region.dependencies)
    # A loop's unnamed input only orders it and a join's inputs only route or order
    # it, so a projection there selects nothing anyone reads.
    if (
        kind in ("loop", "join")
        and (unnamed := unnamed_projection(region.dependencies, name_to_op)) is not None
    ):
        raise compile_error(
            "reads.unnamed-projection",
            f"the dependency on {unnamed!r} projects a part of its value but names no "
            f"input of the {kind} to carry it; add an input name",
            name,
        )
    if (allowed := _REGION_KEYS.get(kind)) is not None and (
        unknown := sorted(set(region.region) - allowed)
    ):
        raise compile_error(
            "region.unknown-field",
            f"a {kind} region declares unknown field(s) {', '.join(unknown)}",
            name,
        )
    if (result := region.region.get("result")) is not None and (
        kind not in _PUBLISHING_KINDS
    ):
        raise compile_error(
            "region.result-unsupported",
            f"a {kind or 'region'} region publishes no result; a spawn, merge, join "
            "or loop region declares result.visibility",
            name,
        )

    dependencies = region.dependencies
    match kind:
        case "merge":
            dependencies = _merge_bindings(region, name_to_op)
            merge = _merge(region, dependencies, has_input)
            _add_operator(merge, region, acc)
            if result is not None:
                _publish_region(result, merge, region, acc)
        case "spawn":
            spawn = _spawn(region, name_to_op, definitions, has_input)
            _add_operator(spawn, region, acc)
            if result is not None:
                _publish_spawn(result, spawn, name, acc)
        case "join":
            join = _join(region, has_input)
            _add_operator(join, region, acc)
            if result is not None:
                _publish_region(result, join, region, acc)
        case "branch":
            _add_operator(_branch(region), region, acc)
        case "loop":
            loop = _loop(region, definitions)
            _add_operator(loop, region, acc)
            if result is not None:
                _publish_region(result, loop, region, acc)
        case "call":
            _lower_call(region, name_to_op, value_ops, definitions, region_kinds, acc)
            return
        case _:
            raise compile_error(
                "region.unknown-kind", f"unknown region kind {kind!r}", name
            )
    _wire_region_dependencies(
        region, region.name, kind, value_ops, region_kinds, acc, dependencies
    )


def _wire_region_dependencies(
    region: ParsedRegion,
    operator_id: str,
    kind: str,
    value_ops: dict[str, str],
    region_kinds: dict[str, str],
    acc: LoweringAccumulator,
    dependencies: list[ParsedDependency],
) -> None:
    """Turn a region's ``dependsOn`` entries into edges into ``operator_id``.

    A region consumes the value on each input, except that a loop's unnamed input
    orders it only. A join collects its spawn's children and runs only on the branch
    arms it depends on; any other join input orders it only.
    """
    for index, dep in enumerate(dependencies):
        source_kind = region_kinds.get(dep.source)
        if kind == "join":
            use = (
                DependencyUse.ROUTE_REQUIRED
                if source_kind == "spawn" or (dep.port and source_kind == "branch")
                else DependencyUse.ORDER_ONLY
            )
        elif kind == "loop" and dep.input is None:
            use = DependencyUse.ORDER_ONLY
        else:
            use = DependencyUse.VALUE_REQUIRED
        acc.edges.append(
            dependency_edge(dep, operator_id, index, use, region.definition, value_ops)
        )


def _add_operator(
    op: LogicalOperator, region: ParsedRegion, acc: LoweringAccumulator
) -> None:
    acc.operators.append(op)
    acc.source_map.append(
        SourceMapEntry(
            logical_ref=op.operator_id, source_kind="region", source_id=region.name
        )
    )
    acc.nodes.append(
        PhysicalNode(
            node_id=f"phys:{op.operator_id}",
            source_ref=op.operator_id,
            logical_ref=op.operator_id,
        )
    )


def _inputs(has_input: bool, name: str = "in") -> tuple[Port, ...]:
    return (Port(name=name),) if has_input else ()


def _merge_bindings(
    region: ParsedRegion, name_to_op: dict[str, str]
) -> list[ParsedDependency]:
    """A merge's dependencies, each bound to a distinct input named for it.

    An unnamed input takes its source's name, qualified by the port it reads; inputs
    otherwise alike are numbered.
    """
    authored = {op: name for name, op in name_to_op.items()}
    taken: list[str] = []
    bindings: list[ParsedDependency] = []
    for dep in region.dependencies:
        if dep.input is not None:
            base = dep.input
        elif dep.source == INGRESS:
            base = dep.port or INGRESS
        else:
            source = authored.get(dep.source, dep.source)
            base = f"{source}.{dep.port}" if dep.port else source
        name, count = base, 1
        while name in taken:
            count += 1
            name = f"{base}#{count}"
        taken.append(name)
        bindings.append(replace(dep, input=name))
    return bindings


def _merge(
    region: ParsedRegion, dependencies: list[ParsedDependency], has_input: bool
) -> MergeRegion:
    inputs = tuple(
        Port(name=dep.input) for dep in dependencies if dep.input
    ) or _inputs(has_input)
    raw = region.region.get("combination")
    combination = None
    if raw is not None:
        try:
            combination = MergeCombination(str(raw))
        except ValueError as exc:
            raise compile_error(
                "region.bad-combination",
                f"unknown merge combination {raw!r}; a merge is one_live or concat",
                region.authored_name,
            ) from exc
    return MergeRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=inputs,
        outputs=(Port(name="out"),),
        combination=combination,
    )


def _ports_field(region: ParsedRegion, key: str) -> tuple[Port, ...]:
    """Parse a region's ``[{name, kind}]`` port list."""
    raw = region.region.get(key)
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise compile_error(
            "region.bad-ports", f"region.{key} must be a list", region.authored_name
        )
    ports: list[Port] = []
    for entry in raw:
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            raise compile_error(
                "region.bad-ports",
                f"each region.{key} entry is a {{name, kind}} mapping",
                region.authored_name,
            )
        if unknown := sorted(set(entry) - {"name", "kind"}):
            raise compile_error(
                "region.bad-ports",
                f"region.{key} entry declares unknown field(s) {', '.join(unknown)}",
                region.authored_name,
            )
        kind = _enum_or_none(PortKind, str(entry.get("kind", PortKind.VALUE.value)))
        if kind is None:
            raise compile_error(
                "region.bad-ports",
                f"unknown port kind {entry.get('kind')!r} in region.{key}",
                region.authored_name,
            )
        name = str(entry["name"]).strip()
        if any(port.name == name for port in ports):
            raise compile_error(
                "region.bad-ports",
                f"region.{key} declares {name!r} twice",
                region.authored_name,
            )
        ports.append(Port(name=name, kind=kind))
    return tuple(ports)


def _branch(region: ParsedRegion) -> BranchRegion:
    name = region.authored_name
    inputs = _ports_field(region, "inputs") or _inputs(bool(region.dependencies))
    outputs = _ports_field(region, "outputs")
    if len(outputs) < 2:
        raise compile_error(
            "branch.outputs",
            "a branch declares at least two output ports to choose among",
            name,
        )
    raw = region.region.get("selection")
    if not isinstance(raw, dict):
        raise compile_error(
            "branch.bad-selection",
            "region.selection is a {input, field, cases} mapping",
            name,
        )
    if unknown := sorted(set(raw) - {"input", "field", "cases"}):
        raise compile_error(
            "branch.bad-selection",
            f"region.selection declares unknown field(s) {', '.join(unknown)}",
            name,
        )
    selected_input = str(raw.get("input") or (inputs[0].name if inputs else ""))
    field = raw.get("field", [])
    steps = field if isinstance(field, list) else [field]
    if not all(isinstance(step, (str, int)) and step != "" for step in steps):
        raise compile_error(
            "branch.bad-selection",
            "region.selection.field is a field name, an index, or a list of them",
            name,
        )
    cases = None
    if (raw_cases := raw.get("cases")) is not None:
        if not isinstance(raw_cases, dict) or not raw_cases:
            raise compile_error(
                "branch.bad-selection",
                "region.selection.cases maps each literal value to an output port",
                name,
            )
        if bad := [v for v in raw_cases if not isinstance(v, str)]:
            raise compile_error(
                "branch.bad-case",
                f"case value {bad[0]!r} is not a string; a selector matches a string, "
                "so quote the case value",
                name,
            )
        cases = tuple(
            SelectionCase(value=value, port=str(port))
            for value, port in raw_cases.items()
        )
    return BranchRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=inputs,
        outputs=outputs,
        rule=SelectionRule(input=selected_input, field=tuple(steps), cases=cases),
        forward=_forward(region, selected_input),
    )


def _forward(region: ParsedRegion, selected_input: str) -> str:
    """The input a branch passes on: the one it declares, else the one it selects
    on."""
    raw = region.region.get("forward")
    if raw is None:
        return selected_input
    if not isinstance(raw, str) or not raw.strip():
        raise compile_error(
            "branch.unknown-forward",
            "region.forward names one of the branch's inputs",
            region.authored_name,
        )
    return raw.strip()


def _loop(region: ParsedRegion, definitions: set[str]) -> LoopContextRegion:
    name = region.authored_name
    carried = _ports_field(region, "carried")
    invariants = _ports_field(region, "invariants")
    body = str(region.region.get("body_ref") or "").strip()
    coordinate = str(region.region.get("loop_coordinate") or "").strip() or name
    if body not in definitions:
        raise compile_error(
            "loop.unknown-body",
            f"a loop's body_ref names a graph template; {body!r} names none",
            name,
        )
    if not carried:
        raise compile_error(
            "loop.no-carried", "a loop declares at least one carried port", name
        )
    if overlap := {p.name for p in carried} & {p.name for p in invariants}:
        raise compile_error(
            "loop.carried-invariant",
            f"{', '.join(sorted(overlap))} is both carried and invariant",
            name,
        )
    return LoopContextRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=(*carried, *invariants),
        outputs=carried,
        loop_coordinate=coordinate,
        carried=carried,
        invariants=invariants,
        body_ref=body,
    )


def _publish_region(
    result: Any,
    op: MergeRegion | JoinRegion | LoopContextRegion,
    region: ParsedRegion,
    acc: LoweringAccumulator,
) -> None:
    """Declare the output a value-yielding region publishes.

    A ``one_live`` merge and a loop publish the one value they forward; a join and a
    ``concat`` merge publish their aggregate as a collection keyed by member.
    """
    name = region.authored_name
    if not isinstance(result, dict):
        raise compile_error(
            "region.result-invalid", "region.result must be a mapping", name
        )
    allowed = (
        {"visibility", "source_port"}
        if isinstance(op, LoopContextRegion)
        else {"visibility"}
    )
    if unknown := sorted(set(result) - allowed):
        raise compile_error(
            "region.result-unknown-field",
            f"region.result declares unknown field(s) {', '.join(unknown)}",
            name,
        )
    if result.get("visibility") != Visibility.PUBLISHED.value:
        raise compile_error(
            "region.result-conflict",
            "a region's result declares visibility: published",
            name,
        )
    source_port: str | None = None
    if isinstance(op, LoopContextRegion):
        ports = [port.name for port in op.carried]
        source_port = str(result.get("source_port") or "") or (
            ports[0] if len(ports) == 1 else ""
        )
        if source_port not in ports:
            raise compile_error(
                "region.result-source-port",
                "a loop publishes one carried port; result.source_port names it",
                name,
            )
    collection = isinstance(op, JoinRegion) or (
        isinstance(op, MergeRegion) and op.combination is not MergeCombination.ONE_LIVE
    )
    acc.result_declarations.append(
        ResultDeclaration(
            output_id=f"{'collection' if collection else 'output'}:{op.operator_id}",
            source_ref=op.operator_id,
            cardinality=(
                CardinalityKind.KEYED_COLLECTION
                if collection
                else CardinalityKind.SINGLETON
            ),
            release=(
                ReleaseConditionKind.JOIN_WINNER
                if isinstance(op, JoinRegion) and op.completion in _EARLY_COMPLETIONS
                else ReleaseConditionKind.SCOPE_CLOSED
            ),
            visibility=Visibility.PUBLISHED,
            keying="member" if collection else None,
            source_port=source_port,
        )
    )


def _spawn(
    region: ParsedRegion,
    name_to_op: dict[str, str],
    definitions: set[str],
    has_input: bool,
) -> SpawnRegion:
    child_ref, definition_ref = _child_target(region, name_to_op, definitions)
    return SpawnRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=_spawn_inputs(region, has_input),
        outputs=(Port(name="children"),),
        child_template_ref=child_ref,
        child_definition_ref=definition_ref,
        authority=_authority(region.region.get("authority"), region.authored_name),
    )


def _child_target(
    region: ParsedRegion, name_to_op: dict[str, str], definitions: set[str]
) -> tuple[str | None, str | None]:
    """A spawn/call child: a graph template's definition, or one operator.

    An operator name resolves to its operator id so the engine both excludes the child
    leaf from eager dispatch and finds its body when materializing a child; an
    unresolved name (no such graph node) is kept as written.
    """
    child = region.region.get("child")
    if not child:
        return None, None
    if (name := str(child)) in definitions:
        return None, name
    return name_to_op.get(name, name), None


def _spawn_inputs(region: ParsedRegion, has_input: bool) -> tuple[Port, ...]:
    """A spawn's fan-out input and the named captures its child definition binds."""
    named = tuple(
        Port(name=dep.input) for dep in region.dependencies if dep.input is not None
    )
    unnamed = any(dep.input is None for dep in region.dependencies)
    return (*(_inputs(True) if unnamed else ()), *named) or _inputs(has_input)


# The fixed shape of a published spawn's collection.
_SPAWN_RESULT_FIXED = {
    "cardinality": CardinalityKind.KEYED_COLLECTION.value,
    "keying": "child_index",
    "release": ReleaseConditionKind.SOURCE_SETTLED.value,
}


def _publish_spawn(
    result: Any, spawn: SpawnRegion, name: str, acc: LoweringAccumulator
) -> None:
    """Declare the keyed collection a spawn region publishes from its children."""
    if not isinstance(result, dict):
        raise compile_error(
            "region.result-invalid", "region.result must be a mapping", name
        )
    if unknown := sorted(set(result) - {"visibility", *_SPAWN_RESULT_FIXED}):
        raise compile_error(
            "region.result-unknown-field",
            f"region.result declares unknown field(s) {', '.join(unknown)}",
            name,
        )
    if result.get("visibility") != Visibility.PUBLISHED.value:
        raise compile_error(
            "region.result-conflict",
            "a spawn region's result declares visibility: published",
            name,
        )
    for field, fixed in _SPAWN_RESULT_FIXED.items():
        if field in result and result[field] != fixed:
            raise compile_error(
                "region.result-conflict",
                f"a published spawn is a {_SPAWN_RESULT_FIXED['cardinality']} keyed "
                f"by child_index and released as each child settles; region.result."
                f"{field} must be {fixed!r}",
                name,
            )
    child_type = next(
        (
            decl.value_type
            for decl in acc.result_declarations
            if decl.source_ref == spawn.child_template_ref and decl.value_type
        ),
        None,
    )
    if child_type is None and spawn.child_definition_ref is None:
        raise compile_error(
            "region.result-unresolved-child",
            f"spawn child {spawn.child_template_ref!r} is not a leaf with a declared "
            "result type, so its collection cannot be published",
            name,
        )
    acc.result_declarations.append(
        ResultDeclaration(
            output_id=f"collection:{spawn.operator_id}",
            source_ref=spawn.operator_id,
            cardinality=CardinalityKind.KEYED_COLLECTION,
            release=ReleaseConditionKind.SOURCE_SETTLED,
            visibility=Visibility.PUBLISHED,
            keying=_SPAWN_RESULT_FIXED["keying"],
            value_type=child_type,
        )
    )


def _join(region: ParsedRegion, has_input: bool) -> JoinRegion:
    completion_raw = str(region.region.get("completion", "")).strip()
    try:
        completion = JoinCompletion(completion_raw)
    except ValueError as exc:
        raise compile_error(
            "region.bad-completion",
            f"unknown join completion {completion_raw!r}",
            region.name,
        ) from exc
    residual = region.region.get("residual")
    predicate_raw = region.region.get("predicate")
    predicate = None
    if isinstance(predicate_raw, dict):
        predicate = JoinPredicate(
            min_qualifiers=int(predicate_raw.get("min_qualifiers", 1)),
            monotone=bool(predicate_raw.get("monotone", True)),
        )
    k = region.region.get("k")
    return JoinRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=_inputs(has_input, "children"),
        outputs=(Port(name="out"),),
        completion=completion,
        residual_policy=str(residual) if residual else None,
        first_k=int(k) if k is not None else None,
        predicate=predicate,
        no_winner_failure=bool(region.region.get("no_winner_failure", False)),
    )


def _lower_call(
    region: ParsedRegion,
    name_to_op: dict[str, str],
    value_ops: dict[str, str],
    definitions: set[str],
    region_kinds: dict[str, str],
    acc: LoweringAccumulator,
) -> None:
    """Normalize a ``call`` into a structured ``Spawn(1)`` then ``Join`` pair."""
    child_ref, definition_ref = _child_target(region, name_to_op, definitions)
    returns = _str_list(region.region.get("returns"), region.authored_name)
    spawn_id = region.name
    join_id = call_join_id(region.name)
    spawn = SpawnRegion(
        operator_id=spawn_id,
        source_ref=region.name,
        inputs=_spawn_inputs(region, bool(region.dependencies)),
        outputs=(Port(name="child"),),
        child_template_ref=child_ref,
        child_definition_ref=definition_ref,
        authority=_authority(region.region.get("authority"), region.authored_name),
    )
    join = JoinRegion(
        operator_id=join_id,
        source_ref=region.name,
        inputs=(Port(name="child"),),
        outputs=tuple(Port(name=port) for port in returns) or (Port(name="out"),),
        completion=JoinCompletion.ALL_SUCCEED,
        call=True,
    )
    _add_operator(spawn, region, acc)
    _add_operator(join, region, acc)
    acc.edges.append(membership_edge(spawn_id, join_id))
    _wire_region_dependencies(
        region, spawn_id, "call", value_ops, region_kinds, acc, region.dependencies
    )
