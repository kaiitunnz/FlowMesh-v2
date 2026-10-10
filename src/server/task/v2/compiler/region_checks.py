"""Validation of branches, loops, merges and the region definitions they enter.

Routes are proved exclusive by the branch arms each operator depends on: an operator
runs only when every arm its required inputs hang from is selected, so two arms of
one branch never both hold.
"""

from collections.abc import Iterable, Mapping, Sequence

from ..representations.operators import (
    AgentOperator,
    BranchRegion,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    LoopContextRegion,
    MergeCombination,
    MergeRegion,
    PortKind,
    SpawnRegion,
    is_spawn_fanout_port,
)
from ..representations.template import (
    BOUNDARY_NODES,
    BoundaryKind,
    DefinitionKind,
    DependencyUse,
    EntryRole,
    LogicalWorkflowTemplate,
    RegionDefinition,
    TemplateEdge,
)
from .diagnostics import Diagnostic, SourceLocation

type Arm = tuple[str, str]
type Arms = frozenset[Arm]
# A return route: the one source record carrying a bundle out of a definition.
type ReturnRoute = tuple[BoundaryKind, str, str | None]


def check_control_flow(
    template: LogicalWorkflowTemplate, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    """Validate every branch, merge, loop and region definition of a template."""
    ops = {op.operator_id: op for op in template.operators}
    definitions = {d.definition_id: d for d in template.definitions}
    diags: list[Diagnostic] = []
    diags.extend(_check_scopes(template, loc))
    arms = _route_arms(template, ops)
    agent_spawns = {
        ref.spawn_ref
        for op in template.operators
        if isinstance(op, AgentOperator)
        for ref in op.child_region_refs
    }
    for op in template.operators:
        match op:
            case BranchRegion():
                diags.extend(_check_branch(op, template.edges, loc))
            case MergeRegion() if op.combination is MergeCombination.ONE_LIVE:
                diags.extend(_check_one_live(op, template.edges, ops, arms, loc))
            case LoopContextRegion():
                diags.extend(_check_loop(op, template, definitions, ops, arms, loc))
            case SpawnRegion() if (
                op.child_template_ref is not None and op.operator_id not in agent_spawns
            ):
                diags.extend(_check_operator_child_entry(op, template, loc))
            case SpawnRegion() if op.child_definition_ref is not None:
                diags.extend(
                    _check_child_entry(op, template, definitions, ops, arms, loc)
                )
    for op_id, needed in arms.items():
        if (conflict := _conflict(needed)) is not None:
            diags.append(
                _error(
                    "reads.exclusive",
                    f"{op_id!r} needs both arms {conflict[0]!r} and {conflict[1]!r} "
                    "of one branch, which never both run; reconverge them through a "
                    "merge",
                    loc.get(op_id),
                )
            )
    for definition in template.definitions:
        entering = next(
            (
                op.operator_id
                for op in template.operators
                if (
                    isinstance(op, LoopContextRegion)
                    and op.body_ref == definition.definition_id
                )
                or (
                    isinstance(op, SpawnRegion)
                    and op.child_definition_ref == definition.definition_id
                )
            ),
            None,
        )
        diags.extend(
            _check_return_routes(
                definition,
                _return_routes(template, definition.definition_id),
                ops,
                arms,
                loc.get(entering or ""),
            )
        )
    diags.extend(_check_definition_nesting(template, ops, definitions, loc))
    return diags


def releases_one_value(
    op_id: str,
    ops: Mapping[str, LogicalOperator],
    incoming: Mapping[str, Sequence[TemplateEdge]],
) -> bool:
    """Whether an operator releases one value rather than an aggregate: a task's
    result, a loop's exit value, a branch arm forwarding one, or a one_live merge each
    of whose arms releases one; ``incoming`` holds each operator's incoming edges."""

    def _one(current: str, seen: frozenset[str]) -> bool:
        match ops.get(current):
            case LeafOperator() | AgentOperator() | LoopContextRegion():
                return True
            case BranchRegion() as branch:
                edges = incoming.get(current, ())
                index = branch.input_index([e.to_port for e in edges], branch.forward)
                if index is None:
                    return False
                edge = edges[index]
                return (
                    edge.boundary is BoundaryKind.ENTRY
                    or edge.from_op in seen
                    or _one(edge.from_op, seen | {current})
                )
            case MergeRegion(combination=MergeCombination.ONE_LIVE):
                return all(
                    edge.boundary is BoundaryKind.ENTRY
                    or edge.from_op in seen
                    or _one(edge.from_op, seen | {current})
                    for edge in incoming.get(current, ())
                )
            case _:
                return False

    return _one(op_id, frozenset())


def _error(code: str, message: str, location: SourceLocation | None) -> Diagnostic:
    return Diagnostic(code=code, message=message, location=location)


def _check_scopes(
    template: LogicalWorkflowTemplate, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    """An edge stays inside one definition; it crosses only through its boundary."""
    owner = template.definition_of()
    diags: list[Diagnostic] = []
    for edge in template.edges:
        if edge.is_forward:
            inside = owner.get(edge.from_op) == owner.get(edge.to_op) == edge.definition
            member = edge.to_op
        else:
            member = edge.to_op if edge.boundary is BoundaryKind.ENTRY else edge.from_op
            inside = edge.definition is not None and owner.get(member) == (
                edge.definition
            )
        if not inside:
            diags.append(
                _error(
                    "definition.boundary",
                    f"edge {edge.from_op!r} -> {edge.to_op!r} crosses a template "
                    "boundary; a template reads its inputs through $ingress and "
                    "returns through its boundary edges",
                    loc.get(member),
                )
            )
    return diags


def _check_branch(
    op: BranchRegion, edges: Iterable[TemplateEdge], loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    location = loc.get(op.operator_id)
    outputs = {port.name for port in op.outputs}
    diags: list[Diagnostic] = []
    if op.rule.input not in {port.name for port in op.inputs}:
        diags.append(
            _error(
                "branch.bad-selection",
                f"selection input {op.rule.input!r} is not an input of the branch",
                location,
            )
        )
    if op.forward not in {port.name for port in op.inputs}:
        diags.append(
            _error(
                "branch.unknown-forward",
                f"forwarded input {op.forward!r} is not an input of the branch",
                location,
            )
        )
    for case in op.rule.cases or ():
        if case.port not in outputs:
            diags.append(
                _error(
                    "branch.bad-case",
                    f"case {case.value!r} selects {case.port!r}, which is not an "
                    "output port of the branch",
                    location,
                )
            )
    for edge in edges:
        if edge.from_op == op.operator_id and edge.from_port is None:
            diags.append(
                _error(
                    "branch.unported-dependency",
                    f"{edge.to_op!r} depends on branch {op.operator_id!r} without "
                    "naming the output port it takes",
                    loc.get(edge.to_op),
                )
            )
    return diags


def _route_arms(
    template: LogicalWorkflowTemplate, ops: dict[str, LogicalOperator]
) -> dict[str, Arms]:
    """The branch arms each operator runs only under.

    An operator needs the arms of each required input; with only ordering inputs it
    needs the arms common to them all, and a merge needs only those common to its
    inputs. A join needs the arms common to its spawns and those of its other
    required inputs.
    """
    incoming: dict[str, list[TemplateEdge]] = {op_id: [] for op_id in ops}
    for edge in template.edges:
        if edge.to_op in incoming:
            incoming[edge.to_op].append(edge)
    arms: dict[str, Arms] = {}
    visiting: set[str] = set()

    def _edge_arms(edge: TemplateEdge) -> Arms:
        source = _arms_of(edge.from_op)
        if isinstance(ops.get(edge.from_op), BranchRegion) and edge.from_port:
            return source | {(edge.from_op, edge.from_port)}
        return source

    def _arms_of(op_id: str) -> Arms:
        if op_id in arms:
            return arms[op_id]
        if op_id in visiting:  # a cycle, refused by the topology pass
            return frozenset()
        visiting.add(op_id)
        op = ops.get(op_id)
        edges = incoming.get(op_id, [])
        if isinstance(op, MergeRegion):
            needed = _common(_edge_arms(edge) for edge in edges)
        elif isinstance(op, JoinRegion):
            spawned = [e for e in edges if isinstance(ops.get(e.from_op), SpawnRegion)]
            needed = _common(_edge_arms(edge) for edge in spawned).union(
                *(
                    _edge_arms(edge)
                    for edge in edges
                    if edge not in spawned and edge.use is not DependencyUse.ORDER_ONLY
                )
            )
        elif required := [e for e in edges if e.use is not DependencyUse.ORDER_ONLY]:
            needed = frozenset().union(*(_edge_arms(edge) for edge in required))
        else:
            needed = _common(_edge_arms(edge) for edge in edges)
        visiting.discard(op_id)
        arms[op_id] = needed
        return needed

    for op_id in ops:
        _arms_of(op_id)
    return arms


def _common(sets: Iterable[Arms]) -> Arms:
    result: Arms | None = None
    for arms in sets:
        result = arms if result is None else result & arms
    return result or frozenset()


def _conflict(arms: Arms) -> tuple[str, str] | None:
    seen: dict[str, str] = {}
    for branch, port in sorted(arms):
        if (other := seen.get(branch)) is not None and other != port:
            return f"{branch}.{other}", f"{branch}.{port}"
        seen[branch] = port
    return None


def _exclusive(left: Arms, right: Arms) -> bool:
    return _conflict(left | right) is not None


def _source_arms(
    op_id: str, port: str | None, ops: dict[str, LogicalOperator], arms: dict[str, Arms]
) -> Arms:
    needed = arms.get(op_id, frozenset())
    if isinstance(ops.get(op_id), BranchRegion) and port:
        return needed | {(op_id, port)}
    return needed


def _check_one_live(
    op: MergeRegion,
    edges: Iterable[TemplateEdge],
    ops: dict[str, LogicalOperator],
    arms: dict[str, Arms],
    loc: dict[str, SourceLocation],
) -> list[Diagnostic]:
    inputs = [
        (edge, _source_arms(edge.from_op, edge.from_port, ops, arms))
        for edge in edges
        if edge.to_op == op.operator_id
    ]
    for index, (left, left_arms) in enumerate(inputs):
        for right, right_arms in inputs[index + 1 :]:
            if not _exclusive(left_arms, right_arms):
                return [
                    _error(
                        "merge.not-exclusive",
                        f"one_live merge {op.operator_id!r} takes {left.from_op!r} and "
                        f"{right.from_op!r}, which can both run; one_live needs "
                        "mutually exclusive branch arms",
                        loc.get(op.operator_id),
                    )
                ]
    return []


def _check_loop(
    op: LoopContextRegion,
    template: LogicalWorkflowTemplate,
    definitions: dict[str, RegionDefinition],
    ops: dict[str, LogicalOperator],
    arms: dict[str, Arms],
    loc: dict[str, SourceLocation],
) -> list[Diagnostic]:
    location = loc.get(op.operator_id)
    diags: list[Diagnostic] = []
    ports = [port.name for port in (*op.carried, *op.invariants)]
    bound = [
        edge.to_port
        for edge in template.edges
        if edge.to_op == op.operator_id and edge.to_port is not None
    ]
    for port in ports:
        if (count := bound.count(port)) != 1:
            diags.append(
                _error(
                    "loop.input-binding",
                    f"loop input {port!r} is bound {count} times; each carried and "
                    "invariant input is bound by exactly one dependsOn entry naming "
                    "it as its input",
                    location,
                )
            )
    if extra := sorted({str(port) for port in bound if port not in ports}):
        diags.append(
            _error(
                "loop.input-binding",
                f"loop dependencies bind {', '.join(extra)}, which name no carried or "
                "invariant input",
                location,
            )
        )
    body = definitions.get(op.body_ref)
    if body is None or body.kind is not DefinitionKind.LOOP_BODY:
        return [*diags, _error("loop.unknown-body", "loop names no body", location)]
    expected = {
        **{p.name: (EntryRole.CARRIED, p.kind) for p in op.carried},
        **{p.name: (EntryRole.INVARIANT, p.kind) for p in op.invariants},
    }
    declared = {p.name: (p.role, p.kind) for p in body.inputs}
    if expected != declared:
        diags.append(
            _error(
                "loop.body-inputs",
                f"body {body.definition_id!r} declares inputs "
                f"{_describe(declared)}; loop {op.operator_id!r} carries "
                f"{_describe(expected)}",
                location,
            )
        )
    carried = {port.name for port in op.carried}
    routes = _return_routes(template, body.definition_id)
    for (kind, source, source_port), names in routes.items():
        if names != carried:
            diags.append(
                _error(
                    "loop.partial-bundle",
                    f"{kind.value} from {source!r}"
                    + (f" port {source_port!r}" if source_port else "")
                    + f" carries {', '.join(sorted(names))}; each feedback and exit "
                    f"carries every carried port ({', '.join(sorted(carried))})",
                    location,
                )
            )
    if not any(kind is BoundaryKind.EGRESS for kind, _, _ in routes):
        diags.append(
            _error(
                "loop.no-exit",
                f"body {body.definition_id!r} never routes to $egress, so the loop "
                "could never exit",
                location,
            )
        )
    return diags


def _check_return_routes(
    definition: RegionDefinition,
    routes: dict[ReturnRoute, set[str]],
    ops: dict[str, LogicalOperator],
    arms: dict[str, Arms],
    location: SourceLocation | None,
) -> list[Diagnostic]:
    """At most one return route of a definition runs at one time, and every arm of a
    branch routing them leads to one."""
    diags: list[Diagnostic] = []
    keyed = [(route, _source_arms(route[1], route[2], ops, arms)) for route in routes]
    for index, (left, left_arms) in enumerate(keyed):
        for right, right_arms in keyed[index + 1 :]:
            if not _exclusive(left_arms, right_arms):
                diags.append(
                    _error(
                        "definition.return-not-exclusive",
                        f"{BOUNDARY_NODES[left[0]]} from {left[1]!r} and "
                        f"{BOUNDARY_NODES[right[0]]} from {right[1]!r} can both "
                        "happen at one time; route them from different arms of one "
                        "branch",
                        location,
                    )
                )
    routing = {branch for _, route_arms in keyed for branch, _ in route_arms} & set(
        definition.members
    )
    for branch in sorted(routing):
        op = ops[branch]
        for port in op.outputs:
            selected = arms.get(branch, frozenset()) | {(branch, port.name)}
            if not any(
                _conflict(route_arms | selected) is None for _, route_arms in keyed
            ):
                diags.append(
                    _error(
                        "definition.unrouted-arm",
                        f"arm {port.name!r} of branch {branch!r} leads to no "
                        f"{' or '.join(sorted({BOUNDARY_NODES[r[0]] for r in routes}))}"
                        f" of {definition.definition_id!r}",
                        location,
                    )
                )
    return diags


def _describe(ports: dict[str, tuple[EntryRole, PortKind]]) -> str:
    return (
        ", ".join(f"{name} ({role.value})" for name, (role, _) in sorted(ports.items()))
        or "nothing"
    )


def _return_routes(
    template: LogicalWorkflowTemplate, definition_id: str
) -> dict[ReturnRoute, set[str]]:
    """A definition's return edges grouped by the source record carrying them out,
    with the boundary ports each route carries."""
    routes: dict[ReturnRoute, set[str]] = {}
    for edge in template.boundary_edges(
        definition_id, BoundaryKind.FEEDBACK, BoundaryKind.EGRESS, BoundaryKind.RETURN
    ):
        if edge.boundary is None or edge.to_port is None:
            continue
        routes.setdefault((edge.boundary, edge.from_op, edge.from_port), set()).add(
            edge.to_port
        )
    return routes


def _check_operator_child_entry(
    op: SpawnRegion, template: LogicalWorkflowTemplate, loc: dict[str, SourceLocation]
) -> list[Diagnostic]:
    incoming = [e for e in template.edges if e.to_op == op.operator_id]
    if len(incoming) == 1 and is_spawn_fanout_port(incoming[0].to_port):
        return []
    child = loc.get(op.child_template_ref or "")
    return [
        _error(
            "spawn.param",
            f"spawn {op.operator_id!r} fans out over exactly one unnamed input; a "
            f"child task {child.source_id if child else op.child_template_ref!r} "
            "reads no other input",
            loc.get(op.operator_id),
        )
    ]


def _check_child_entry(
    op: SpawnRegion,
    template: LogicalWorkflowTemplate,
    definitions: dict[str, RegionDefinition],
    ops: dict[str, LogicalOperator],
    arms: dict[str, Arms],
    loc: dict[str, SourceLocation],
) -> list[Diagnostic]:
    location = loc.get(op.operator_id)
    child = definitions.get(op.child_definition_ref or "")
    if child is None or child.kind is not DefinitionKind.CHILD:
        return [
            _error(
                "region.spawn-child-unresolved",
                f"child template {op.child_definition_ref!r} is not a child template",
                location,
            )
        ]
    diags: list[Diagnostic] = []
    captures = {p.name for p in child.inputs if p.role is EntryRole.CAPTURE}
    incoming = [e for e in template.edges if e.to_op == op.operator_id]
    bound = [
        e.to_port
        for e in incoming
        if e.to_port is not None and not is_spawn_fanout_port(e.to_port)
    ]
    if sorted(bound) != sorted(captures):
        diags.append(
            _error(
                "spawn.captures",
                f"spawn {op.operator_id!r} binds "
                f"{', '.join(sorted(map(str, bound))) or 'no capture'}; child "
                f"template {child.definition_id!r} captures "
                f"{', '.join(sorted(captures)) or 'nothing'}",
                location,
            )
        )
    if sum(is_spawn_fanout_port(e.to_port) for e in incoming) != 1:
        diags.append(
            _error(
                "spawn.param",
                f"spawn {op.operator_id!r} fans out over exactly one unnamed input, "
                f"which binds the param of {child.definition_id!r}",
                location,
            )
        )
    routes = _return_routes(template, child.definition_id)
    returns = {port.name for port in child.returns}
    for (_, source, port), names in routes.items():
        if names != returns:
            diags.append(
                _error(
                    "definition.partial-return",
                    f"$return from {source!r}"
                    + (f" port {port!r}" if port else "")
                    + f" carries {', '.join(sorted(names))}; each return carries "
                    f"{', '.join(sorted(returns))}",
                    loc.get(source),
                )
            )
    if not routes:
        diags.append(
            _error(
                "definition.no-return",
                f"child template {child.definition_id!r} never routes to $return",
                location,
            )
        )
    diags.extend(_check_call_returns(op, template, ops, returns, location))
    return diags


def _check_call_returns(
    op: SpawnRegion,
    template: LogicalWorkflowTemplate,
    ops: dict[str, LogicalOperator],
    returns: set[str],
    location: SourceLocation | None,
) -> list[Diagnostic]:
    """A call's declared returns are returns of the child template it calls."""
    join = next(
        (
            ops[edge.to_op]
            for edge in template.edges
            if edge.from_op == op.operator_id
            and isinstance(ops.get(edge.to_op), JoinRegion)
        ),
        None,
    )
    if not isinstance(join, JoinRegion) or join.source_ref != op.operator_id:
        return []
    declared = {port.name for port in join.outputs}
    if declared == {"out"} or not (unknown := sorted(declared - returns)):
        return []
    return [
        _error(
            "call.unknown-return",
            f"call {op.operator_id!r} returns {', '.join(unknown)}, which "
            f"{op.child_definition_ref!r} does not return",
            location,
        )
    ]


def _check_definition_nesting(
    template: LogicalWorkflowTemplate,
    ops: dict[str, LogicalOperator],
    definitions: dict[str, RegionDefinition],
    loc: dict[str, SourceLocation],
) -> list[Diagnostic]:
    """A definition reaches itself again only through a child entry, never only
    through loops, which would nest a loop inside its own body."""
    loops_in: dict[str, set[str]] = {d: set() for d in definitions}
    for definition in definitions.values():
        for member in definition.members:
            if isinstance(op := ops.get(member), LoopContextRegion):
                loops_in[definition.definition_id].add(op.body_ref)

    def _reaches(start: str) -> bool:
        pending, seen = list(loops_in.get(start, ())), set()
        while pending:
            current = pending.pop()
            if current == start:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(loops_in.get(current, ()))
        return False

    return [
        _error(
            "definition.cycle",
            f"loop body {name!r} contains a loop over itself",
            loc.get(name),
        )
        for name in sorted(definitions)
        if _reaches(name)
    ]
