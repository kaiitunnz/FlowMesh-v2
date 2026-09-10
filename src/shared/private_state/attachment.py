"""Exclusive, fenced authority to materialize and write one bound generation."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PrivateStateAttachment(BaseModel):
    """Materialization and write authority for one holder over one bound generation.

    An attachment is physical execution authority: it admits no capacity, holds no
    credit, and reserves nothing. Its ``write_epoch`` is the exclusive-writer fence —
    one writable attachment exists per bound generation, and a holder whose epoch has
    been superseded can neither write nor resume.
    """

    model_config = ConfigDict(frozen=True)

    attachment_id: str
    reference_id: str
    generation: int
    worker_id: str
    incarnation: int
    write_epoch: int


class PrivateStateUnavailableReason(StrEnum):
    """Why a bound generation could not be supplied to a resuming holder."""

    OWNER_LOST = "owner_lost"
    INCARNATION_MISMATCH = "incarnation_mismatch"
    STALE_EPOCH = "stale_epoch"
    COMPONENT_MISSING = "component_missing"
    COMPONENT_MISMATCH = "component_mismatch"
    CONTAINMENT_VIOLATION = "containment_violation"


class PrivateStateUnavailable(Exception):
    """A bound generation cannot be supplied, so the continuation fails closed.

    Raised instead of resuming against a fresh, partial, or foreign generation.
    """

    def __init__(
        self, reason: PrivateStateUnavailableReason, detail: str, *, reference_id: str
    ) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail
        self.reference_id = reference_id
