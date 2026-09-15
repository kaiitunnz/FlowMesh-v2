"""Materializing an inference contract against the upstream snapshot pinned to a task.

The origin worker resolves a leaf's source once, before either embodiment reaches its
model: the self-contained one generates from the resolved request and the resident one
serializes that same request across its service boundary, so a replica receives prompts
rather than a projection to interpret. The raw upstream values stay on this worker —
only the resolved request and the digests recording how it was reached travel further.
"""

from shared.inference import (
    CanonicalInferenceContract,
    InferenceSourceKind,
    InputResolutionError,
    ResolvedCanonicalInferenceRequest,
    UpstreamProvenance,
    content_version,
    resolve_contract,
)

from ..base_executor import ExecutionError, ExecutorTask
from ..utils.expressions import project_expression


def resolve_task_contract(
    task: ExecutorTask,
) -> ResolvedCanonicalInferenceRequest | None:
    """The request a task's contract names, or None when it carries no contract."""
    contract = task.declared_contract
    if contract is None:
        return None
    if contract.source.kind is InferenceSourceKind.LITERAL:
        return resolve_contract(contract, None)
    return _resolve_upstream(task, contract)


def _resolve_upstream(
    task: ExecutorTask, contract: CanonicalInferenceContract
) -> ResolvedCanonicalInferenceRequest:
    source = contract.source
    context = task.spec.upstreamResults or {}
    node = source.node or ""
    upstream = context.get(node)
    if upstream is None:
        raise InputResolutionError(
            f"{source.expression} names upstream input {node!r}, which this task does "
            "not declare a dependency on"
        )
    try:
        # Tables project under their own canonical contract, so this walker admits only
        # the scalar and structured steps and a table arrives as an unprojectable input.
        projected = project_expression(source.expression, context, frames=False)
    except ExecutionError as exc:
        raise InputResolutionError(
            f"{source.expression} does not project: {exc}"
        ) from (exc)
    provenance = UpstreamProvenance(
        node=node, content_digest=content_version(upstream.model_dump(mode="json"))
    )
    return resolve_contract(contract, projected, (provenance,))
