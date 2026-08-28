from enum import Enum
from typing import Any

from ...parser import ParsedRegion, ParsedTask, ParsedWorkflow
from ..representations.operators import (
    AgentOperator,
    AuthorityCeiling,
    BoundaryEventKind,
    BoundarySignature,
    BranchRegion,
    DeterminismClass,
    EffectClass,
    InputProvenanceKind,
    JoinCompletion,
    JoinPredicate,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    LoopContextRegion,
    MergeRegion,
    ModelRef,
    Port,
    PortKind,
    RecoveryClass,
    SpawnRegion,
)
from ..representations.plan import PhysicalNode
from ..representations.results import Visibility
from ..representations.template import (
    ResourceDeclaration,
    SourceMapEntry,
    TemplateEdge,
    ToolDeclaration,
)
from .diagnostics import compile_error
from .project import LoweringAccumulator

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
    structured regions into canonical operators/ports/regions, and adds structured
    feedback edges. Malformed constructs raise :class:`CompileError` with a source
    location; semantic checks are left to the validation passes.
    """
    _apply_leaf_declarations(parsed, acc)
    _lower_regions(parsed, acc)
    _lower_feedback(parsed, acc)


def _apply_leaf_declarations(parsed: ParsedWorkflow, acc: LoweringAccumulator) -> None:
    by_id = {op.operator_id: idx for idx, op in enumerate(acc.operators)}
    for task in parsed.tasks:
        if not task.v2:
            continue
        idx = by_id.get(task.task_id)
        if idx is None:
            continue
        op = acc.operators[idx]
        acc.operators[idx] = _apply_one(task, op, acc)


def _apply_one(
    task: ParsedTask, op: LogicalOperator, acc: LoweringAccumulator
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
        return _apply_agent_v2(op, v2, name)
    if isinstance(op, LeafOperator):
        return _apply_leaf_v2(op, v2, name)
    raise compile_error(
        "v2.unsupported-operator",
        "spec.v2 applies only to task/agent leaves",
        name,
        "graph_node",
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


def _lower_regions(parsed: ParsedWorkflow, acc: LoweringAccumulator) -> None:
    for region in parsed.regions:
        _lower_region(region, acc)


def _lower_region(region: ParsedRegion, acc: LoweringAccumulator) -> None:
    kind = str(region.region.get("kind", "")).strip()
    name = region.name
    has_input = bool(region.depends_on)

    if kind == "branch":
        _add_operator(_branch(region, has_input), region, acc)
    elif kind == "merge":
        _add_operator(_merge(region, has_input), region, acc)
    elif kind == "spawn":
        _add_operator(_spawn(region, has_input), region, acc)
    elif kind == "join":
        _add_operator(_join(region, has_input), region, acc)
    elif kind == "loop":
        _add_operator(_loop(region, has_input), region, acc)
    elif kind == "call":
        _lower_call(region, acc)
        return
    else:
        raise compile_error(
            "region.unknown-kind", f"unknown region kind {kind!r}", name
        )

    for dep in region.depends_on:
        acc.edges.append(TemplateEdge(from_op=dep, to_op=name))


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


def _branch(region: ParsedRegion, has_input: bool) -> BranchRegion:
    ports = _str_list(region.region.get("ports"), region.name)
    return BranchRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=_inputs(has_input),
        outputs=tuple(Port(name=port) for port in ports),
        selection=(str(sel) if (sel := region.region.get("selection")) else None),
    )


def _merge(region: ParsedRegion, has_input: bool) -> MergeRegion:
    inputs = tuple(Port(name=dep) for dep in region.depends_on) or _inputs(has_input)
    return MergeRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=inputs,
        outputs=(Port(name="out"),),
        combination=(
            str(region.region["combination"])
            if region.region.get("combination")
            else None
        ),
    )


def _spawn(region: ParsedRegion, has_input: bool) -> SpawnRegion:
    child = region.region.get("child")
    return SpawnRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=_inputs(has_input),
        outputs=(Port(name="children"),),
        child_template_ref=str(child) if child else None,
        authority=_authority(region.region.get("authority"), region.name),
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


def _loop(region: ParsedRegion, has_input: bool) -> LoopContextRegion:
    coordinate = str(region.region.get("coordinate", "")).strip()
    if not coordinate:
        raise compile_error(
            "region.loop-no-coordinate",
            "loop region requires a coordinate",
            region.name,
        )
    carried: list[Port] = []
    for entry in region.region.get("carried", []) or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise compile_error(
                "region.loop-bad-carried",
                "each carried entry needs a name",
                region.name,
            )
        port_kind = str(entry.get("kind", "value"))
        model_ref = None
        if port_kind == PortKind.MODEL_REF.value:
            ref = entry.get("modelRef") or {}
            model_ref = ModelRef(
                architecture=str(ref.get("architecture", entry["name"])),
                version=str(v) if (v := ref.get("version")) else None,
            )
        carried.append(
            Port(
                name=str(entry["name"]),
                kind=(
                    PortKind(port_kind)
                    if port_kind in {k.value for k in PortKind}
                    else PortKind.VALUE
                ),
                model_ref=model_ref,
            )
        )
    return LoopContextRegion(
        operator_id=region.name,
        source_ref=region.name,
        inputs=_inputs(has_input, "ingress") + tuple(carried),
        outputs=(Port(name="egress"), *carried),
        loop_coordinate=coordinate,
        carried=tuple(carried),
    )


def _lower_call(region: ParsedRegion, acc: LoweringAccumulator) -> None:
    """Normalize a ``call`` into a structured ``Spawn(1)`` then ``Join`` pair."""
    child = region.region.get("child")
    returns = _str_list(region.region.get("returns"), region.name)
    spawn_id = region.name
    join_id = f"{region.name}:join"
    spawn = SpawnRegion(
        operator_id=spawn_id,
        source_ref=region.name,
        inputs=_inputs(bool(region.depends_on)),
        outputs=(Port(name="child"),),
        child_template_ref=str(child) if child else None,
        authority=_authority(region.region.get("authority"), region.name),
    )
    join = JoinRegion(
        operator_id=join_id,
        source_ref=region.name,
        inputs=(Port(name="child"),),
        outputs=tuple(Port(name=port) for port in returns) or (Port(name="out"),),
        completion=JoinCompletion.ALL_SUCCEED,
    )
    _add_operator(spawn, region, acc)
    _add_operator(join, region, acc)
    acc.edges.append(TemplateEdge(from_op=spawn_id, to_op=join_id))
    for dep in region.depends_on:
        acc.edges.append(TemplateEdge(from_op=dep, to_op=spawn_id))


def _lower_feedback(parsed: ParsedWorkflow, acc: LoweringAccumulator) -> None:
    operator_ids = acc.operator_ids
    loop_ids = {
        op.operator_id for op in acc.operators if isinstance(op, LoopContextRegion)
    }
    for source_id, feedback, label in _feedback_sources(parsed):
        target = str(feedback.get("to", "")).strip()
        if target not in operator_ids:
            raise compile_error(
                "feedback.unknown-target",
                f"feedback targets unknown operator {target!r}",
                label,
            )
        if target not in loop_ids:
            raise compile_error(
                "feedback.not-loop",
                f"feedback target {target!r} is not a LoopContext region",
                label,
            )
        port = feedback.get("port")
        acc.edges.append(
            TemplateEdge(
                from_op=source_id,
                to_op=target,
                to_port=str(port) if port else None,
                feedback=True,
            )
        )


def _feedback_sources(
    parsed: ParsedWorkflow,
) -> list[tuple[str, dict[str, Any], str]]:
    sources: list[tuple[str, dict[str, Any], str]] = []
    for task in parsed.tasks:
        if task.feedback:
            label = task.graph_node_name or task.local_name or task.task_id
            sources.append((task.task_id, task.feedback, label))
    for region in parsed.regions:
        if region.feedback:
            sources.append((region.name, region.feedback, region.name))
    return sources
