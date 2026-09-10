"""Exclusive, fenced authority to materialize and write one bound generation."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .manifest import StateBundleManifest


class PrivateStateAttachment(BaseModel):
    """Materialization and write authority for one holder over one bound generation.

    An attachment is physical execution authority over one activation's state, distinct
    from the capacity admission a service claim carries. Its ``write_epoch`` is the
    exclusive-writer fence: one writable attachment exists per bound generation, and a
    holder whose epoch has been superseded can neither write nor resume.
    """

    model_config = ConfigDict(frozen=True)

    attachment_id: str
    reference_id: str
    generation: int
    worker_id: str
    incarnation: int
    write_epoch: int


class PrivateStateSealReport(BaseModel):
    """A holder's report of the generation it sealed under its attachment.

    ``write_epoch`` is the fence the seal is accepted under: a holder whose epoch has
    been superseded cannot advance the lineage.
    """

    model_config = ConfigDict(frozen=True)

    manifest: StateBundleManifest
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
    """Raised when a bound generation cannot be supplied in full.

    The continuation fails closed rather than resuming against a fresh, partial, or
    foreign generation.
    """

    def __init__(
        self, reason: PrivateStateUnavailableReason, detail: str, *, reference_id: str
    ) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail
        self.reference_id = reference_id
