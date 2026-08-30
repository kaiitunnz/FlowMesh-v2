"""Transient control-plane DTOs for fabric-served tool dispatch.

An agent's mediated tool boundary (a model invocation, a search) is recorded durably
as a ``BoundaryEvent`` in the ledger; these are the non-persisted structures the runtime
builds from that durable fact to route the boundary to its handler and to carry a
normalized outcome back. They mint no identity and hold no credential: the engine owns
the durable invocation, and the ``grant_snapshot`` names the pinned authorization for
the broker's egress/budget accounting, never a re-authorization.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ..task.v2.representations.operators import BoundaryEventKind

# The reserved interface of a deferred managed-model invocation, distinct from a
# fabric-served tool interface. Exact routing keys on these two exact values.
MODEL_INTERFACE = "model"
SEARCH_INTERFACE = "search/v1"


class GrantSnapshot(BaseModel):
    """The pinned authorization a mediated boundary was recorded under.

    Read by the broker for egress/rate/budget accounting against the authorization the
    engine already granted; it never re-authorizes.
    """

    model_config = ConfigDict(frozen=True)

    grant_id: str | None = None
    policy_envelope: str | None = None


class ToolInvocationEnvelope(BaseModel):
    """A recorded mediated boundary, lifted from its durable ``BoundaryEvent``.

    Carries the engine identity (``invocation_id``), the correlation that survives a
    re-drive, the fabric dedupe authority (``idempotency_key``), the request payload,
    and the pinned ``grant_snapshot`` — everything a handler needs, so recovery routes a
    search to the search handler and never misroutes it to the model settler.
    """

    model_config = ConfigDict(frozen=True)

    kind: BoundaryEventKind
    interface: str
    invocation_id: str
    task_id: str
    activation_id: str
    call_correlation: str
    idempotency_key: str | None = None
    request_payload: str | None = None
    grant_snapshot: GrantSnapshot = GrantSnapshot()


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
