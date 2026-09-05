"""The generic fabric external-tool operation contract: envelopes and outcomes.

The control path issues a bounded ``ToolOperationEnvelope`` (and a per-delivery
``RemoteToolOperationEnvelope`` fence); a worker executor validates the fence and
returns a normalized ``ToolOutcome``. These are tool-agnostic — a per-tool package
(today ``shared.tools.search``) supplies the request shape and provider egress — and
they mint no identity and hold no credential.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


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


class RemoteToolOperationEnvelope(BaseModel):
    """A short-lived operation fence for one bounded off-server external-tool operation.

    The control path issues it per physical delivery attempt; the worker executor
    validates it before egress and rejects an expired, altered, wrong-provider,
    wrong-audience, wrong-policy, over-budget, or replayed operation as a tool-fence
    failure — never a reachability-demoting observation. It is not a ``ServiceClaim``,
    ``RouteAuthorization``, or lease, and holds no credential. ``request_digest`` binds
    request integrity; ``provider`` and ``target_id`` / ``target_generation`` bind the
    audience, both provisioned out-of-band over the authenticated attachment and never
    guessable from a frame; ``delivery_nonce`` is a one-use, target-scoped authorization
    the executor consumes atomically before egress, so an exact replay of one authorized
    delivery is refused while a fresh same-``idempotency_key`` re-drive with a new nonce
    is accepted; ``deadline_epoch`` bounds its lifetime. ``tenant`` is carried as audit
    context only; per-tenant enforcement is deferred with per-tenant credential
    delegation. ``policy_class`` binds the operation's policy to the target's bound
    policy.
    """

    model_config = ConfigDict(frozen=True)

    interface: str
    provider: str
    idempotency_key: str | None
    request_digest: str
    target_id: str
    target_generation: int
    delivery_nonce: str
    tenant: str | None = None
    policy_class: str = "default"
    deadline_epoch: float
    max_results: int
    timeout_sec: float
    result_char_cap: int


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


__all__ = [
    "MediatedOperationPermit",
    "RemoteToolOperationEnvelope",
    "ToolOperationEnvelope",
    "ToolOutcome",
    "ToolOutcomeStatus",
]
