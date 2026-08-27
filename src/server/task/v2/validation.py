from .diagnostics import Diagnostic, Severity, SourceLocation
from .operators import (
    AgentOperator,
    AuthorityCeiling,
    BranchRegion,
    DeterminismClass,
    EffectClass,
    InputProvenanceKind,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    LoopContextRegion,
    MergeRegion,
    RecoveryClass,
    SpawnRegion,
)
from .plan import PhysicalExecutionPlan
from .results import CardinalityKind, ReleaseConditionKind
from .template import LogicalWorkflowTemplate

_DETERMINISTIC = (
    DeterminismClass.DETERMINISTIC_BITWISE,
    DeterminismClass.DETERMINISTIC_SEMANTIC,
)
_EARLY_JOINS = (JoinCompletion.ANY, JoinCompletion.FIRST_K, JoinCompletion.PREDICATE)


def _location_index(
    template: LogicalWorkflowTemplate,
) -> dict[str, SourceLocation]:
    index: dict[str, SourceLocation] = {}
    for entry in template.source_map:
        index[entry.logical_ref] = SourceLocation(
            source_kind=entry.source_kind, source_id=entry.source_id
        )
    return index


def _port_names(op: LogicalOperator) -> set[str]:
    return {port.name for port in (*op.inputs, *op.outputs)}


def _check_source_map(
    template: LogicalWorkflowTemplate,
    plan: PhysicalExecutionPlan,
    loc: dict[str, SourceLocation],
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    ids = template.operator_ids
    for op in template.operators:
        if op.operator_id not in loc:
            diags.append(
                Diagnostic(
                    code="source-map.incomplete",
                    message=f"operator {op.operator_id!r} has no source-map entry",
                )
            )
    for node in plan.nodes:
        if node.logical_ref is not None and node.logical_ref not in ids:
            diags.append(
                Diagnostic(
                    code="source-map.dangling-plan-node",
                    message=(
                        f"physical node {node.node_id!r} maps to unknown "
                        f"operator {node.logical_ref!r}"
                    ),
                )
            )
    return diags


def _check_ports(
    template: LogicalWorkflowTemplate, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    ports_by_op = {op.operator_id: _port_names(op) for op in template.operators}
    for edge in template.edges:
        if edge.from_port is not None:
            names = ports_by_op.get(edge.from_op, set())
            if edge.from_port not in names:
                diags.append(
                    Diagnostic(
                        code="ports.unknown-output",
                        message=(
                            f"edge references output port {edge.from_port!r} absent "
                            f"on operator {edge.from_op!r}"
                        ),
                        location=loc.get(edge.from_op),
                    )
                )
        if edge.to_port is not None:
            names = ports_by_op.get(edge.to_op, set())
            if edge.to_port not in names:
                diags.append(
                    Diagnostic(
                        code="ports.unknown-input",
                        message=(
                            f"edge references input port {edge.to_port!r} absent "
                            f"on operator {edge.to_op!r}"
                        ),
                        location=loc.get(edge.to_op),
                    )
                )
    return diags


def _check_recompute_legality(
    op: LeafOperator, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    profile = op.profile
    if profile.recovery != RecoveryClass.RECOMPUTE:
        return []
    reasons: list[str] = []
    if profile.effect != EffectClass.PURE:
        reasons.append(f"effect is {profile.effect.value}")
    if profile.input_provenance != InputProvenanceKind.EXTERNAL_PINNED:
        reasons.append("input is a live external read")
    if profile.determinism not in _DETERMINISTIC:
        reasons.append("output is nondeterministic")
    if not reasons:
        return []
    return [
        Diagnostic(
            code="recovery.illegal-recompute",
            message=(
                "leaf declares recompute recovery but "
                + ", ".join(reasons)
                + "; only a pure, deterministic operation over pinned inputs may "
                "recompute"
            ),
            location=loc.get(op.operator_id),
        )
    ]


def _check_effect_boundary(
    template: LogicalWorkflowTemplate,
    op: LeafOperator,
    loc: dict[str, SourceLocation],
) -> list[Diagnostic]:
    if op.profile.effect != EffectClass.EXTERNAL_EFFECT or op.residency_only:
        return []
    declared = {b.source_ref for b in template.effect_boundaries}
    if op.operator_id in declared:
        return []
    return [
        Diagnostic(
            code="effect.missing-boundary",
            message=(
                f"external-effect leaf {op.operator_id!r} has no declared effect "
                "boundary"
            ),
            location=loc.get(op.operator_id),
        )
    ]


def _check_authority(
    op: AgentOperator | SpawnRegion,
    authority: AuthorityCeiling,
    declared_tools: set[str],
    loc: dict[str, SourceLocation],
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    invoke = set(authority.invoke)
    delegate = set(authority.delegate)
    over_delegate = delegate - invoke
    if over_delegate:
        diags.append(
            Diagnostic(
                code="authority.delegate-exceeds-invoke",
                message=(
                    "delegate face "
                    + ", ".join(sorted(over_delegate))
                    + " is not attenuated by the invoke face"
                ),
                location=loc.get(op.operator_id),
            )
        )
    undeclared = invoke - declared_tools
    if undeclared:
        diags.append(
            Diagnostic(
                code="authority.undeclared-tool",
                message=(
                    "invoke face references undeclared tool(s) "
                    + ", ".join(sorted(undeclared))
                ),
                location=loc.get(op.operator_id),
            )
        )
    return diags


def _check_region(
    op: LogicalOperator, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    location = loc.get(op.operator_id)
    if isinstance(op, BranchRegion):
        if not op.outputs:
            diags.append(
                Diagnostic(
                    code="region.branch-no-ports",
                    message="branch region declares no output ports",
                    location=location,
                )
            )
        if not op.selection:
            diags.append(
                Diagnostic(
                    code="region.branch-no-selection",
                    message="branch region declares no selection rule",
                    location=location,
                )
            )
    elif isinstance(op, MergeRegion):
        if not op.inputs:
            diags.append(
                Diagnostic(
                    code="region.merge-no-ports",
                    message="merge region declares no input ports",
                    location=location,
                )
            )
    elif isinstance(op, SpawnRegion):
        if not op.child_template_ref:
            diags.append(
                Diagnostic(
                    code="region.spawn-no-child",
                    message="spawn region declares no child template",
                    location=location,
                )
            )
    elif isinstance(op, JoinRegion):
        if op.completion in _EARLY_JOINS and not op.residual_policy:
            diags.append(
                Diagnostic(
                    code="region.join-no-residual",
                    message=(
                        f"early join ({op.completion.value}) declares no "
                        "residual-child policy"
                    ),
                    location=location,
                )
            )
    elif isinstance(op, LoopContextRegion):
        if not op.loop_coordinate:
            diags.append(
                Diagnostic(
                    code="region.loop-no-coordinate",
                    message="loop-context region declares no loop coordinate",
                    location=location,
                )
            )
    return diags


def _check_result_declarations(
    template: LogicalWorkflowTemplate, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    join_ids = {
        op.operator_id for op in template.operators if isinstance(op, JoinRegion)
    }
    for decl in template.result_declarations:
        if decl.cardinality == CardinalityKind.KEYED_COLLECTION and not decl.keying:
            diags.append(
                Diagnostic(
                    code="result.keyed-without-key",
                    message=(
                        f"result {decl.output_id!r} is a keyed collection but "
                        "declares no keying"
                    ),
                    location=loc.get(decl.source_ref),
                )
            )
        if (
            decl.release == ReleaseConditionKind.JOIN_WINNER
            and decl.source_ref not in join_ids
        ):
            diags.append(
                Diagnostic(
                    code="result.join-winner-not-join",
                    message=(
                        f"result {decl.output_id!r} releases on a join winner but "
                        f"does not resolve from a join region"
                    ),
                    location=loc.get(decl.source_ref),
                )
            )
    return diags


def _check_cycles(
    template: LogicalWorkflowTemplate, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    adjacency: dict[str, list[str]] = {op.operator_id: [] for op in template.operators}
    for edge in template.edges:
        if edge.feedback:
            continue
        if edge.from_op in adjacency:
            adjacency[edge.from_op].append(edge.to_op)

    visiting, visited = set(), set()

    def _walk(node: str) -> str | None:
        visiting.add(node)
        for nxt in adjacency.get(node, ()):
            if nxt in visiting:
                return nxt
            if nxt not in visited:
                found = _walk(nxt)
                if found is not None:
                    return found
        visiting.discard(node)
        visited.add(node)
        return None

    for op_id in adjacency:
        if op_id not in visited:
            hit = _walk(op_id)
            if hit is not None:
                return [
                    Diagnostic(
                        code="topology.unstructured-cycle",
                        message=(
                            f"operator {hit!r} participates in an unstructured cycle; "
                            "declare structured feedback with a LoopContext region"
                        ),
                        location=loc.get(hit),
                    )
                ]
    return []


def validate_compilation(
    template: LogicalWorkflowTemplate, plan: PhysicalExecutionPlan
) -> list[Diagnostic]:
    """Run every validation pass over a compiled template and physical plan.

    Passes check declaration *consistency*: an unpinned live read is legal
    latitude, not an error. Only a contradictory declaration (e.g. recompute over
    a live external read) or a malformed region/authority face fails validation.
    """
    loc = _location_index(template)
    declared_tools = {tool.name for tool in template.tool_declarations}
    diags: list[Diagnostic] = []

    diags.extend(_check_source_map(template, plan, loc))
    diags.extend(_check_ports(template, loc))
    diags.extend(_check_result_declarations(template, loc))
    diags.extend(_check_cycles(template, loc))

    for op in template.operators:
        diags.extend(_check_region(op, loc))
        if isinstance(op, LeafOperator):
            diags.extend(_check_recompute_legality(op, loc))
            diags.extend(_check_effect_boundary(template, op, loc))
        elif isinstance(op, AgentOperator):
            diags.extend(_check_authority(op, op.authority, declared_tools, loc))
        elif isinstance(op, SpawnRegion):
            diags.extend(_check_authority(op, op.authority, declared_tools, loc))

    return diags


def has_errors(diagnostics: list[Diagnostic]) -> bool:
    """Whether any diagnostic is error-severity."""
    return any(diag.severity is Severity.ERROR for diag in diagnostics)
