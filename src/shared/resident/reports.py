"""What the origin worker reports to control at each transition of a resident boundary.

Control mints the fences and owns the claim FSM; the origin worker drives the data path
and reports the two transition-gating facts back — the engine enqueue ack and the fenced
terminal outcome. Both are safe under loss: a missing report leaves the claim
credit-bearing and the boundary re-drives under the same identity, never falling through
to a wrong terminal.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from shared.outcome import OutcomeManifest


class ResidentBootstrapOutcome(StrEnum):
    """The origin worker's read of the bootstrap phase.

    ``ACKED`` records the engine enqueue ack so control accepts the claim and issues the
    route authorization; ``REJECTED`` is a definite fence rejection that
    releases the reservation; ``UNCERTAIN`` is an ambiguous or lost delivery that holds
    the credit and re-drives.
    """

    ACKED = "acked"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class ResidentBootstrapAck(BaseModel):
    """The origin worker's bootstrap-phase report for one attempt."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    call_correlation: str
    invocation_id: str
    session_id: str
    outcome: ResidentBootstrapOutcome
    rejection: str | None = None


class ResidentStreamStatus(StrEnum):
    """The origin worker's read of the response stream and materialization.

    ``SUCCESS`` carries the completed immutable manifest that lets a fenced terminal
    release the credit; ``DEFINITE_FAILURE`` is a definite engine refusal or a known
    materialization failure that settles the boundary; ``UNCERTAIN`` is a pre-manifest
    loss that holds the credit and re-drives.
    """

    SUCCESS = "success"
    DEFINITE_FAILURE = "definite_failure"
    UNCERTAIN = "uncertain"


class ResidentOpOutcome(BaseModel):
    """The origin worker's fenced terminal report for one resident boundary."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    call_correlation: str
    invocation_id: str
    session_id: str
    status: ResidentStreamStatus
    manifest: OutcomeManifest | None = None
    error: str | None = None


class ResidentStreamHead(BaseModel):
    """The engine response head a task-addressed serve client receives before its body.

    The replica sidecar reads the engine response's HTTP status and content type and the
    gated edge relays them opaquely, so the client's response carries the engine's own
    status and content type ahead of the streamed body. It is transport metadata the
    edge forwards without interpreting; the body itself streams as opaque chunks.
    """

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    session_id: str
    status: int
    content_type: str


class ResidentStreamChunk(BaseModel):
    """One authorized response frame teed to a live task-addressed serve client.

    A task-addressed serve invocation has no continuation to resume, so the gated edge's
    relay executor tees each response frame to control as it streams and control relays
    the opaque frame to the client unparsed. The payload is opaque bytes-as-text that
    control and the edge never parse; neither assembles or materializes a completion.
    """

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    session_id: str
    payload: str
