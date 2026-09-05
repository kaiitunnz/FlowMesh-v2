"""The shared worker-side fence and outcome materialization for fabric tool egress.

Both worker-side entry points to external-tool egress validate the same fence and
materialize a successful result the same way: the server-driven attachment path
(:class:`WorkerExternalToolExecutor`, fenced by a ``RemoteToolOperationEnvelope``) and
the worker-originated path (:class:`ToolOperationExecutor`, fenced by a
``MediatedOperationPermit``). These helpers are that one egress boundary; the callers
differ only in how they source the request and frame the result.
"""

import time
from dataclasses import dataclass

from shared.outcome import FabricContentStore, OutcomeManifest
from shared.tools.contract import ToolOutcome, ToolOutcomeStatus
from shared.tools.search.schema import ToolRequest, tool_request_digest


@dataclass(frozen=True)
class ProviderBinding:
    """The provider name and optional key the local worker environment provisions."""

    provider: str
    api_key: str | None


def fence_reason(
    *,
    interface: str,
    target_id: str,
    target_generation: int,
    policy_class: str,
    deadline_epoch: float,
    request_digest: str,
    request: ToolRequest,
    worker_id: str,
    worker_generation: int,
    allowed_interfaces: frozenset[str],
    expected_policy_class: str | None,
) -> str | None:
    """Why an authorized operation fails this worker's fence, or None if it passes.

    Checks the audience (worker id + generation), the declared interface, the deadline,
    and the request integrity digest, verified over the exact request the worker will
    egress, so an altered request or digest is rejected before any provider call. The
    policy class is compared only when ``expected_policy_class`` is set; a caller with
    no independent policy expectation passes ``None`` to skip it. Provider audience and
    result-budget bounds are the caller's to apply, since they differ between the two
    fences.
    """
    if interface not in allowed_interfaces:
        return "interface"
    if target_id != worker_id:
        return "audience"
    if target_generation != worker_generation:
        return "generation"
    if expected_policy_class is not None and policy_class != expected_policy_class:
        return "policy"
    if time.time() > deadline_epoch:
        return "expired"
    if request.interface != interface:
        return "interface_mismatch"
    if (
        tool_request_digest(request.interface, request.query, request.max_results)
        != request_digest
    ):
        return "digest"
    return None


def materialize_tool_outcome(
    outcome: ToolOutcome,
    *,
    idempotency_key: str | None,
    content_store: FabricContentStore | None,
) -> OutcomeManifest | ToolOutcome:
    """A reference for a materialized successful result, else a typed inline outcome.

    A successful result always materializes through the content store so no result body
    crosses the control plane; a non-success status is a bounded inline outcome. A
    successful result the worker cannot reference — no content store or idempotency
    key — is reported as a typed unavailable outcome rather than an unbounded body.
    """
    if outcome.status is not ToolOutcomeStatus.SUCCESS:
        return outcome
    if content_store is None or idempotency_key is None:
        return ToolOutcome(
            status=ToolOutcomeStatus.UNAVAILABLE,
            value="no content store is configured to materialize the result",
        )
    return content_store.materialize(
        idempotency_key,
        outcome.model_dump_json().encode(),
        media_type="application/json",
    )
