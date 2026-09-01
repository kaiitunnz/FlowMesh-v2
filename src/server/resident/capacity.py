"""Pure safe-capacity accounting for conservative admission.

FlowMesh shapes which compatible invocations reach an engine and how much conservative
admission credit they consume; it does not recreate the engine's token scheduler or KV
allocator. These helpers compute, from a replica's normalized capacity report and the
outstanding credit-bearing claims, whether a further claim fits and how tightly — so
concurrent reservations cannot overcommit the same reported safe headroom.
"""

from collections.abc import Iterable

from ..utils.time import now_iso, parse_iso_ts
from .state import (
    SERVABLE_REPLICA_STATES,
    AdmissionProfile,
    ClaimCredit,
    ReplicaCapacityReport,
    ServiceClaim,
)


def default_credit(profile: AdmissionProfile) -> ClaimCredit:
    """The conservative credit a claim reserves: one admission slot plus tokens."""
    return ClaimCredit(slots=1, projected_tokens=profile.max_output_tokens)


def outstanding_slots(claims: Iterable[ServiceClaim]) -> int:
    """The admission slots held by every credit-bearing claim in the set."""
    return sum(
        (claim.credit.slots if claim.credit else 1)
        for claim in claims
        if claim.holds_credit
    )


def is_feasible(
    report: ReplicaCapacityReport,
    profile: AdmissionProfile,
    held_slots: int,
    *,
    now_ts: float | None = None,
) -> bool:
    """Whether a replica can safely admit one more claim of this profile.

    A replica must be healthy and servable, satisfy an adapter-slot constraint, and have
    safe headroom after every outstanding credit. The gate is conservative: it never
    packs past the reported safe slots even to make a denser batch.
    """
    if not report.healthy or report.state not in SERVABLE_REPLICA_STATES:
        return False
    if (
        profile.adapter_ref is not None
        and report.adapter_slots_free is not None
        and report.adapter_slots_free <= 0
    ):
        return False
    if profile.deadline_at is not None:
        reference = now_ts if now_ts is not None else parse_iso_ts(now_iso())
        if parse_iso_ts(profile.deadline_at) <= reference:
            return False
    return (report.safe.admission_slots - held_slots) >= 1


def residual_after(
    report: ReplicaCapacityReport, held_slots: int, credit: ClaimCredit
) -> float:
    """Normalized safe headroom left after a candidate's projected reservation.

    Best-fit prefers the feasible replica with the least remaining headroom, so an
    efficient batch fills before work spills to another replica.
    """
    total = max(1, report.safe.admission_slots)
    remaining = report.safe.admission_slots - held_slots - credit.slots
    return remaining / total
