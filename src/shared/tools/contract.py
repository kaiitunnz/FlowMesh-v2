"""The generic fabric external-tool operation contract: envelopes and outcomes.

The control path mints a one-use ``MediatedOperationPermit`` and issues a bounded
``ToolOperationEnvelope``; a worker's egress sidecar validates the permit fence and
returns a normalized ``ToolOutcome`` or ``MediatedOperationOutcome``. These are
tool-agnostic — a per-tool package (today ``shared.tools.search``) supplies the request
shape and provider egress — and they mint no identity and hold no credential.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from shared.outcome import OutcomeManifest


class ToolOperationEnvelope(BaseModel):
    """A server-issued authorization for exactly one bounded external-tool operation.

    The control path issues it after the authority and quota checks; the execution
    surface egresses only within it. ``idempotency_key`` is the fabric dedupe authority
    a re-drive reuses; the result bounds cap the single authorized operation.
    ``task_id`` names the originating episode a carriage routes to its assigned worker.
    """

    model_config = ConfigDict(frozen=True)

    interface: str
    idempotency_key: str | None
    max_results: int
    timeout_sec: float
    result_char_cap: int
    task_id: str | None = None


class MediatedOperationPermit(BaseModel):
    """A one-use control authorization for one worker-originated mediated operation.

    The control plane mints it after verifying the live continuation, effective
    authority, quota, and occurrence idempotence, and returns it to the agent's own
    worker. That worker's off-lane executor validates it before egress and rejects an
    expired, altered, wrong-audience, or over-budget operation as a fence failure. It is
    neither a ``ServiceClaim`` credit nor an endpoint credential, and it carries no
    request payload: the raw request stays in worker-private state, looked up by
    ``(agent_task_id, call_correlation)``.

    ``request_digest`` binds request integrity; ``target_id`` / ``target_generation``
    bind the audience to the agent's worker incarnation; ``permit_id`` is a one-use,
    unguessable grant a fresh same-``idempotency_key`` re-drive re-mints;
    ``deadline_epoch`` bounds its lifetime; ``invocation_id`` and ``idempotency_key``
    together are the durable identity the outcome settles against. ``subject``,
    ``policy_class``, and
    ``policy_epoch`` are declared here as a forward contract; the paths that bind real
    subjects and policy generations enforce them.

    ``credential`` is a per-call provider secret the control plane resolves for a
    workflow that pins its own model key; it rides only this one-use, audience-bound
    delivery down to the egressing worker, never travels up in a proposal, and is never
    persisted or logged. A worker without one falls back to its local environment key.
    """

    model_config = ConfigDict(frozen=True)

    permit_id: str
    agent_task_id: str
    call_correlation: str
    interface: str
    subject: str
    invocation_id: str
    idempotency_key: str | None
    request_digest: str
    target_id: str
    target_generation: int
    policy_class: str = "default"
    policy_epoch: int = 0
    deadline_epoch: float
    max_results: int
    timeout_sec: float
    result_char_cap: int
    credential: str | None = Field(default=None, repr=False)


class ToolOutcomeStatus(StrEnum):
    """The typed result class of a fabric-served tool call, all durably injected."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    QUOTA = "quota"
    UNAVAILABLE = "unavailable"


class ToolOutcome(BaseModel):
    """A normalized, typed tool outcome the broker returns for durable injection.

    ``value`` is the model-facing rendering injected at the originating call; a
    non-success status still injects a bounded ``value`` (never an empty success and
    never an agent failure). ``provenance`` carries citation for a successful call.
    """

    model_config = ConfigDict(frozen=True)

    status: ToolOutcomeStatus
    value: str
    provenance: tuple[dict[str, str], ...] = ()


class MediatedOperationOutcome(BaseModel):
    """The fenced terminal fact a worker reports for one mediated operation.

    The agent's own worker egressed the operation under its permit and reports the
    result back over the authenticated attachment. Exactly one of ``outcome`` (a bounded
    typed control datum), ``outcome_ref`` (a reference to materialized content), or
    ``error`` (a worker-fault fence failure) is set; the control plane settles the
    originating boundary from it. ``permit_id`` correlates the report to the minted
    permit; ``agent_task_id`` / ``call_correlation`` name the boundary;
    ``invocation_id`` and ``idempotency_key`` are its durable identity.
    """

    model_config = ConfigDict(frozen=True)

    permit_id: str
    agent_task_id: str
    call_correlation: str
    invocation_id: str
    idempotency_key: str | None
    outcome: ToolOutcome | None = None
    outcome_ref: OutcomeManifest | None = None
    error: str | None = None


__all__ = [
    "MediatedOperationOutcome",
    "MediatedOperationPermit",
    "ToolOperationEnvelope",
    "ToolOutcome",
    "ToolOutcomeStatus",
]
