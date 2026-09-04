"""The shared external-tool operation contract: requests, fences, and outcomes.

The control path issues a bounded ``ToolOperationEnvelope`` (and a per-delivery
``RemoteToolOperationEnvelope`` fence); a worker executor validates the fence and
returns a normalized ``ToolOutcome``. These mint no identity and hold no credential.
"""

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

# The reserved fabric-served tool interface. Exact routing keys on this exact value.
SEARCH_INTERFACE = "search/v1"

# The keyless web-search backend selected when no provider is configured.
DEFAULT_SEARCH_PROVIDER = "duckduckgo"


class ToolRequest(BaseModel):
    """The parsed, bounds-shaped request the control path derived from a boundary."""

    model_config = ConfigDict(frozen=True)

    interface: str
    query: str
    max_results: int


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


def tool_request_digest(interface: str, query: str, max_results: int) -> str:
    """A canonical integrity digest over the bounded request the fence commits to.

    The worker executor recomputes it over the delivered request and rejects a mismatch,
    so an altered request or an altered digest fails the fence before any provider call.
    """
    raw = f"{interface}\x00{query}\x00{max_results}".encode()
    return hashlib.sha256(raw).hexdigest()


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
    "DEFAULT_SEARCH_PROVIDER",
    "SEARCH_INTERFACE",
    "RemoteToolOperationEnvelope",
    "ToolOperationEnvelope",
    "ToolOutcome",
    "ToolOutcomeStatus",
    "ToolRequest",
    "tool_request_digest",
]
