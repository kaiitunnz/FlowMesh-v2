"""Episode-cut lowering: an alternative to the transparent per-operator plan.

The transparent lowering mints one physical node per logical operator. This pass
rewrites that node set into run-to-yield episodes: each dispatchable node is annotated
with the boundary that closes its episode, and a maximal chain of pure, deterministic,
local leaves fuses into one episode node whose ``fused_refs`` records the folded
operators. The logical template is untouched, so the lowering is contract-equivalent to
the transparent one: it changes only where episodes cut, never a declared output, an
effect boundary, or progress closure.
"""

from ..representations.operators import (
    AgentOperator,
    BoundaryEventKind,
    DeterminismClass,
    EffectClass,
    LeafOperator,
    LogicalOperator,
    OperatorKind,
    SpawnRegion,
)
from ..representations.plan import EpisodeBoundaryKind, EpisodeSpec, PhysicalNode
from ..representations.template import LogicalWorkflowTemplate

_REGION_BLOCKING = frozenset(
    {OperatorKind.SPAWN, OperatorKind.JOIN, OperatorKind.LOOP_CONTEXT}
)


def _boundary_for(op: LogicalOperator) -> EpisodeBoundaryKind | None:
    """The boundary that closes an operator's episode, or ``None`` for residency."""
    if isinstance(op, LeafOperator):
        if op.residency_only:
            return None
        if op.profile.effect is EffectClass.EXTERNAL_EFFECT:
            return EpisodeBoundaryKind.EFFECT
        if op.profile.effect is EffectClass.PRIVATE_STATE:
            return EpisodeBoundaryKind.DURABLE_CHECKPOINT
        if op.profile.determinism is DeterminismClass.SAMPLED:
            return EpisodeBoundaryKind.SERVICE_ISSUE
        return EpisodeBoundaryKind.TASK
    if isinstance(op, AgentOperator):
        events = set(op.boundary.events)
        if BoundaryEventKind.INVOCATION in events:
            return EpisodeBoundaryKind.SERVICE_ISSUE
        if BoundaryEventKind.YIELD in events:
            return EpisodeBoundaryKind.CONTINUATION
        return EpisodeBoundaryKind.TASK
    if op.kind in _REGION_BLOCKING:
        return EpisodeBoundaryKind.REGION_BLOCKING
    return EpisodeBoundaryKind.CONTINUATION


def _is_fusible(op: LogicalOperator | None) -> bool:
    return (
        isinstance(op, LeafOperator)
        and not op.residency_only
        and op.profile.effect is EffectClass.PURE
        and op.profile.determinism is not DeterminismClass.SAMPLED
    )


def lower_to_episodes(
    template: LogicalWorkflowTemplate, nodes: tuple[PhysicalNode, ...]
) -> tuple[PhysicalNode, ...]:
    """Rewrite transparent nodes into episode nodes with boundaries and fusion."""
    ops = {op.operator_id: op for op in template.operators}
    child_templates = {
        op.child_template_ref
        for op in template.operators
        if isinstance(op, SpawnRegion) and op.child_template_ref
    }
    succ, pred = _linear_adjacency(template, child_templates)
    fused_into = _fuse_chains(ops, child_templates, succ, pred)

    rewritten: list[PhysicalNode] = []
    for node in nodes:
        ref = node.logical_ref
        op = ops.get(ref) if ref else None
        if ref in fused_into:  # folded into an earlier chain head's episode
            continue
        boundary = _boundary_for(op) if op is not None else None
        if boundary is None:  # residency administration node, left as-is
            rewritten.append(node)
            continue
        folded = tuple(k for k, head in fused_into.items() if head == ref)
        rewritten.append(
            node.model_copy(
                update={"episode": EpisodeSpec(boundary=boundary, fused_refs=folded)}
            )
        )
    return tuple(rewritten)


def _linear_adjacency(
    template: LogicalWorkflowTemplate, child_templates: set[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Forward successor/predecessor maps over dispatchable operators only."""
    kind = {op.operator_id: op.kind for op in template.operators}
    succ: dict[str, list[str]] = {}
    pred: dict[str, list[str]] = {}
    for edge in template.edges:
        if (
            edge.feedback
            or edge.from_op in child_templates
            or edge.to_op in child_templates
        ):
            continue
        if (
            kind.get(edge.from_op) is OperatorKind.SPAWN
            and kind.get(edge.to_op) is OperatorKind.JOIN
        ):
            continue
        succ.setdefault(edge.from_op, []).append(edge.to_op)
        pred.setdefault(edge.to_op, []).append(edge.from_op)
    return succ, pred


def _fuse_chains(
    ops: dict[str, LogicalOperator],
    child_templates: set[str],
    succ: dict[str, list[str]],
    pred: dict[str, list[str]],
) -> dict[str, str]:
    """Map each fused operator to its chain head, folding maximal pure-leaf runs."""
    fused_into: dict[str, str] = {}
    for op_id, op in ops.items():
        if op_id in fused_into or op_id in child_templates or not _is_fusible(op):
            continue
        # A chain head is not the single fusible successor of a fusible predecessor.
        preds = pred.get(op_id, [])
        if (
            len(preds) == 1
            and _is_fusible(ops.get(preds[0]))
            and succ.get(preds[0]) == [op_id]
        ):
            continue
        cursor = op_id
        while (nexts := succ.get(cursor, [])) and len(nexts) == 1:
            nxt_id = nexts[0]
            if (
                not _is_fusible(ops.get(nxt_id))
                or pred.get(nxt_id, []) != [cursor]
                or nxt_id in fused_into
            ):
                break
            fused_into[nxt_id] = op_id
            cursor = nxt_id
    return fused_into
