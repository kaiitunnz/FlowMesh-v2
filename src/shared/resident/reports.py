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


class ResidentStreamChunk(BaseModel):
    """One authorized response frame teed to a live ingress request's client.

    An ingress request has no continuation to resume, so the origin worker tees each
    response frame to control as it streams — in addition to assembling and
    materializing the completion — and control relays the opaque frame to the client
    unparsed. The payload is opaque bytes-as-text; control and ingress never parse it.
    """

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    session_id: str
    seq: int
    payload: str
