"""The admission claim state machine as pure transitions over a ``ServiceClaim``.

A claim advances ``PENDING -> RESERVED -> ACCEPTED -> STREAMING -> TERMINAL`` on the
normal path. A pre-acceptance cancellation, known enqueue failure, or expiry records a
terminal transition directly; a route or incarnation loss moves a credit-bearing claim
to ``UNCERTAIN``/``RECONCILING`` and only a fenced terminal outcome releases its credit.
Each transition is validated against the source state so a derived credit can never be
released by an illegal or duplicate step.
"""

from shared.utils.ids import new_service_claim_id

from ..utils.time import now_iso
from .state import (
    ClaimCredit,
    ClaimState,
    ClaimTerminalReason,
    ServiceClaim,
)

_TERMINAL_SOURCES: dict[ClaimTerminalReason, frozenset[ClaimState]] = {
    ClaimTerminalReason.COMPLETED: frozenset(
        {
            ClaimState.ACCEPTED,
            ClaimState.STREAMING,
            ClaimState.UNCERTAIN,
            ClaimState.RECONCILING,
        }
    ),
    ClaimTerminalReason.CANCELLED: frozenset(
        {
            ClaimState.PENDING,
            ClaimState.RESERVED,
            ClaimState.ACCEPTED,
            ClaimState.STREAMING,
            ClaimState.UNCERTAIN,
            ClaimState.RECONCILING,
        }
    ),
    ClaimTerminalReason.ENQUEUE_FAILED: frozenset({ClaimState.RESERVED}),
    ClaimTerminalReason.EXPIRED: frozenset({ClaimState.PENDING, ClaimState.RESERVED}),
    ClaimTerminalReason.RECONCILED: frozenset(
        {ClaimState.UNCERTAIN, ClaimState.RECONCILING}
    ),
}

_UNCERTAIN_SOURCES: frozenset[ClaimState] = frozenset(
    {ClaimState.RESERVED, ClaimState.ACCEPTED, ClaimState.STREAMING}
)


class ClaimTransitionError(ValueError):
    """An admission claim transition was attempted from an illegal source state."""


def _touch(claim: ServiceClaim, state: ClaimState) -> None:
    claim.state = state
    claim.updated_at = now_iso()


def reserve(
    claim: ServiceClaim,
    *,
    replica_id: str,
    incarnation: int,
    report_epoch: int,
    credit: ClaimCredit,
) -> None:
    """Bind a pending claim to a selected replica incarnation and hold its credit.

    The replica incarnation and report epoch fence the reservation so a stale report or
    a superseded incarnation cannot later be mistaken for this credit.
    """
    if claim.state is not ClaimState.PENDING:
        raise ClaimTransitionError(f"cannot reserve a claim in {claim.state}")
    claim.replica_id = replica_id
    claim.incarnation = incarnation
    claim.report_epoch = report_epoch
    claim.credit = credit
    _touch(claim, ClaimState.RESERVED)


def accept(claim: ServiceClaim) -> None:
    """Record an engine enqueue acknowledgement for a reserved claim."""
    if claim.state is not ClaimState.RESERVED:
        raise ClaimTransitionError(f"cannot accept a claim in {claim.state}")
    _touch(claim, ClaimState.ACCEPTED)


def begin_stream(claim: ServiceClaim) -> None:
    """Move an accepted claim into its response stream."""
    if claim.state is not ClaimState.ACCEPTED:
        raise ClaimTransitionError(f"cannot stream a claim in {claim.state}")
    _touch(claim, ClaimState.STREAMING)


def mark_uncertain(claim: ServiceClaim) -> None:
    """Record a route or incarnation loss without releasing the held credit."""
    if claim.state not in _UNCERTAIN_SOURCES:
        raise ClaimTransitionError(f"cannot mark uncertain a claim in {claim.state}")
    _touch(claim, ClaimState.UNCERTAIN)


def begin_reconcile(claim: ServiceClaim) -> None:
    """Move an uncertain claim into active reconciliation, still holding its credit."""
    if claim.state is not ClaimState.UNCERTAIN:
        raise ClaimTransitionError(f"cannot reconcile a claim in {claim.state}")
    _touch(claim, ClaimState.RECONCILING)


def settle_terminal(claim: ServiceClaim, reason: ClaimTerminalReason) -> None:
    """Advance a claim to ``TERMINAL`` and release its derived credit.

    Only a transition legal for the reason is accepted; a terminal claim is never
    reopened. A ``COMPLETED`` or ``RECONCILED`` reason represents a fenced terminal
    outcome consumed from ``DS`` by ``invocation_id``.
    """
    if claim.state is ClaimState.TERMINAL:
        return
    if claim.state not in _TERMINAL_SOURCES[reason]:
        raise ClaimTransitionError(f"cannot settle {reason} from {claim.state}")
    claim.terminal_reason = reason
    _touch(claim, ClaimState.TERMINAL)


def release_on_ds_terminal(claim: ServiceClaim, reason: ClaimTerminalReason) -> None:
    """Release a claim's credit from a fenced DS terminal outcome.

    A terminal recorded in the ledger and consumed by ``invocation_id`` is
    authoritative, so it settles any non-terminal claim regardless of its source state —
    the sole normal path by which an accepted credit disappears. Idempotent on an
    already-terminal claim.
    """
    if claim.state is ClaimState.TERMINAL:
        return
    claim.terminal_reason = reason
    _touch(claim, ClaimState.TERMINAL)


def successor_claim(claim: ServiceClaim) -> ServiceClaim:
    """Raise a fresh successor claim for a permitted reissue.

    The successor carries a new ``claim_id`` and an incremented admission epoch under
    the same ``invocation_id``; it starts ``PENDING`` and holds no credit until it
    reserves independently.
    """
    if claim.state is not ClaimState.TERMINAL:
        raise ClaimTransitionError("cannot reissue a non-terminal claim")
    return ServiceClaim(
        claim_id=new_service_claim_id(),
        invocation_id=claim.invocation_id,
        family=claim.family,
        admission_epoch=claim.admission_epoch + 1,
    )


def new_claim(
    *, invocation_id: str, family: str, admission_epoch: int = 0
) -> ServiceClaim:
    """Raise a first pending claim for an invocation's admission."""
    return ServiceClaim(
        claim_id=new_service_claim_id(),
        invocation_id=invocation_id,
        family=family,
        admission_epoch=admission_epoch,
    )
