"""The fabric-facade contract shared between the worker facade and control.

A held facade injects an agent's pinned fabric tools into a model turn, captures the
calls the model co-emits, and reports them as a turn group; control records the group
and routes each member kind-specifically. These types cross that boundary, so they live
here rather than in either side's internals.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from shared.harness import BoundaryEventKind


class FacadeDescriptor(BaseModel):
    """A fabric-owned facade tool the facade injects for one agent.

    ``name`` is the model-facing tool name whose call the facade captures;
    ``tool_schema`` is the function-tool JSON injected into the model turn; ``kind`` and
    ``interface`` are the boundary the captured call originates. The compiler pins the
    exact set an agent may use, so the facade injects only its declared facades.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    kind: BoundaryEventKind
    interface: str | None = None
    tool_schema: str  # the injected function-tool schema, serialized


class FacadeCompletionMode(StrEnum):
    """How a facade group member completes, driving the group's resume gate.

    ``AWAIT_OUTCOME`` (a search) holds the episode until its durable outcome settles and
    injects that outcome on resume. ``ADMIT_AND_CLOSE`` (a spawn) settles at admission
    with a deterministic acceptance ack, never a child result, and never holds the gate.
    """

    ADMIT_AND_CLOSE = "admit_and_close"
    AWAIT_OUTCOME = "await_outcome"


class FacadeCallMember(BaseModel):
    """One ordered facade call captured in a model turn, of any mediated kind.

    Members share their turn group's id and the work item's single continuation; each
    keeps its own stable ``call_correlation`` (turn base plus source ``ordinal``) and
    the harness call it injects back at (``harness_call_id`` under ``tool_name``).
    ``kind`` and ``completion_mode`` select the member's kind-specific routing: a spawn
    materializes one child and acks at admission, a search defers and holds the gate.
    ``interface_or_region`` carries the search interface or the spawn's target region.
    """

    model_config = ConfigDict(frozen=True)

    ordinal: int
    kind: BoundaryEventKind
    completion_mode: FacadeCompletionMode
    call_correlation: str
    harness_call_id: str
    tool_name: str
    interface_or_region: str | None = None
    request_payload: str | None = None
    # Set for a worker-captured search member: the raw request stays worker-private and
    # its presence routes the recorded boundary to the off-lane worker egress, never the
    # in-server broker. A spawn member carries its args in ``request_payload`` instead.
    request_digest: str | None = None


class FacadeTurnGroup(BaseModel):
    """A model turn's captured facade calls, recorded before the cleaned turn returns.

    One group per turn carries every facade call the model co-emitted, ordered by source
    ``ordinal``. The fabric assigns the stable ``group_id`` and each member's
    correlation from ``(activation_id, turn_id, ordinal)``, never a harness call id, so
    a re-drive of the same turn recovers the same identities and creates no duplicate
    work. Members complete kind-specifically; only ``AWAIT_OUTCOME`` members hold the
    episode's resume gate.
    """

    model_config = ConfigDict(frozen=True)

    group_id: str
    activation_id: str
    turn_id: str
    members: tuple[FacadeCallMember, ...]
    capsule: str | None = None


__all__ = [
    "FacadeCallMember",
    "FacadeCompletionMode",
    "FacadeDescriptor",
    "FacadeTurnGroup",
]
