"""The worker-side fence and outcome materialization for fabric tool egress.

The worker's :class:`MediatedEgressSidecar` validates a ``MediatedOperationPermit``
against this fence and materializes a successful result through the content store before
it leaves the worker. These helpers are that one egress boundary.
"""

import time
from dataclasses import dataclass

from shared.outcome import FabricContentStore, OutcomeManifest
from shared.tools.contract import ToolOutcome, ToolOutcomeStatus


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
    computed_digest: str,
    worker_id: str,
    worker_generation: int,
    allowed_interfaces: frozenset[str],
    expected_policy_class: str | None,
) -> str | None:
    """Why an authorized operation fails this worker's fence, or None if it passes.

    Checks the audience (worker id + generation), the declared interface, the deadline,
    and the request integrity digest. ``computed_digest`` is recomputed by the caller
    over the exact request it will egress, so an altered request or digest is rejected
    before any provider call; it also binds the interface the request was framed for.
    The policy class is compared only when ``expected_policy_class`` is set; a caller
    with no independent policy expectation passes ``None`` to skip it. Provider audience
    and result-budget bounds are the caller's, since they differ between the fences.
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
    if computed_digest != request_digest:
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
