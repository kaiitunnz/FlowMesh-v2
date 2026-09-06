"""Transient control-plane DTOs for fabric-served tool dispatch.

An agent's mediated tool boundary (a model invocation, a search) is recorded durably
as a ``BoundaryEvent`` in the ledger; these are the non-persisted structures the runtime
builds from that durable fact to route the boundary to its handler and to carry a
normalized outcome back. They mint no identity and hold no credential: the engine owns
the durable invocation, and the ``grant_snapshot`` names the pinned authorization for
the broker's egress/budget accounting, never a re-authorization.
"""

from pydantic import BaseModel, ConfigDict

from shared.tools.contract import ToolOutcome as ToolOutcome
from shared.tools.contract import ToolOutcomeStatus as ToolOutcomeStatus
from shared.tools.facade import FacadeCallMember as FacadeCallMember
from shared.tools.facade import FacadeCompletionMode as FacadeCompletionMode
from shared.tools.facade import FacadeTurnGroup as FacadeTurnGroup
from shared.tools.model.schema import MODEL_INTERFACE as MODEL_INTERFACE
from shared.tools.search.schema import SEARCH_INTERFACE

from ..task.v2.representations.operators import BoundaryEventKind

# The interfaces the fabric serves as an injected facade tool (never the model turn).
FABRIC_TOOL_INTERFACES = frozenset({SEARCH_INTERFACE})


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
    # Set for a worker-originated boundary (the raw request stays worker-private); its
    # presence routes the boundary to the off-lane worker path rather than the in-server
    # broker.
    request_digest: str | None = None
    grant_snapshot: GrantSnapshot = GrantSnapshot()


class InputMemberPlan(BaseModel):
    """One member of an agent input the engine identifies for the runtime to resolve.

    The engine owns membership and ordering; the runtime resolves ``value_ref`` to a
    frozen value and digest. A single producer binding yields one member; a merge/join
    aggregate yields one per settled source child, ordered by the declared contract.
    """

    model_config = ConfigDict(frozen=True)

    source_operator_id: str
    source_activation_id: str
    child_index: int | None = None
    outcome: str
    value_ref_kind: str
    legacy_task_id: str | None = None
    collection_key: str | None = None
    literal: str | None = None
    ordinal: int = 0


class InputPortPlan(BaseModel):
    """The ordered members bound to one declared agent input port."""

    model_config = ConfigDict(frozen=True)

    target_port: str
    provenance: str
    members: tuple[InputMemberPlan, ...] = ()


class AgentInputPlan(BaseModel):
    """The engine's per-port input membership for an activation, to be resolved."""

    model_config = ConfigDict(frozen=True)

    activation_id: str
    ports: tuple[InputPortPlan, ...] = ()
