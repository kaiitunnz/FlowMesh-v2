"""The admission claim FSM: legal progressions and fenced terminals.

These prove a credit is held only across the credit-bearing states, that pre-acceptance
terminals release it through a recorded transition, and that a route loss holds the
credit in ``UNCERTAIN`` until a fenced terminal outcome settles it.
"""

import pytest

from server.resident import (
    ClaimCredit,
    ClaimState,
    ClaimTerminalReason,
    ClaimTransitionError,
    accept,
    begin_stream,
    mark_uncertain,
    new_claim,
    reserve,
    settle_terminal,
)

_CREDIT = ClaimCredit(slots=1, projected_tokens=64)


def _reserved():
    claim = new_claim(invocation_id="inv-a", family="fam")
    reserve(claim, replica_id="rpl-1", incarnation=1, credit=_CREDIT)
    return claim


def test_happy_path_holds_then_releases_credit():
    claim = new_claim(invocation_id="inv-a", family="fam")
    assert claim.state is ClaimState.PENDING and not claim.holds_credit

    reserve(claim, replica_id="rpl-1", incarnation=1, credit=_CREDIT)
    assert claim.state is ClaimState.RESERVED and claim.holds_credit
    assert claim.replica_id == "rpl-1" and claim.incarnation == 1

    accept(claim)
    assert claim.state is ClaimState.ACCEPTED and claim.holds_credit
    begin_stream(claim)
    assert claim.state is ClaimState.STREAMING and claim.holds_credit

    settle_terminal(claim, ClaimTerminalReason.COMPLETED)
    assert claim.state is ClaimState.TERMINAL and not claim.holds_credit
    assert claim.terminal_reason is ClaimTerminalReason.COMPLETED


@pytest.mark.parametrize(
    "reason",
    [ClaimTerminalReason.CANCELLED, ClaimTerminalReason.EXPIRED],
)
def test_pending_pre_acceptance_terminal(reason):
    claim = new_claim(invocation_id="inv-a", family="fam")
    settle_terminal(claim, reason)
    assert claim.state is ClaimState.TERMINAL and not claim.holds_credit


def test_reserved_enqueue_failure_releases_credit():
    claim = _reserved()
    settle_terminal(claim, ClaimTerminalReason.ENQUEUE_FAILED)
    assert claim.state is ClaimState.TERMINAL and not claim.holds_credit


def test_route_loss_holds_credit_until_fenced_terminal():
    claim = _reserved()
    accept(claim)
    mark_uncertain(claim)
    assert claim.state is ClaimState.UNCERTAIN and claim.holds_credit
    settle_terminal(claim, ClaimTerminalReason.COMPLETED)
    assert claim.state is ClaimState.TERMINAL and not claim.holds_credit


def test_uncertain_can_settle_completed_from_ds_outcome():
    claim = _reserved()
    accept(claim)
    begin_stream(claim)
    mark_uncertain(claim)
    settle_terminal(claim, ClaimTerminalReason.COMPLETED)
    assert claim.state is ClaimState.TERMINAL


def test_illegal_transitions_raise():
    claim = new_claim(invocation_id="inv-a", family="fam")
    with pytest.raises(ClaimTransitionError):
        accept(claim)  # cannot accept a PENDING claim
    with pytest.raises(ClaimTransitionError):
        begin_stream(claim)
    with pytest.raises(ClaimTransitionError):
        settle_terminal(claim, ClaimTerminalReason.ENQUEUE_FAILED)  # needs RESERVED
    reserve(claim, replica_id="rpl-1", incarnation=1, credit=_CREDIT)
    with pytest.raises(ClaimTransitionError):
        begin_stream(claim)  # cannot stream a RESERVED claim


def test_terminal_is_idempotent_and_not_reopened():
    claim = _reserved()
    accept(claim)
    settle_terminal(claim, ClaimTerminalReason.COMPLETED)
    settle_terminal(claim, ClaimTerminalReason.COMPLETED)  # no-op
    assert claim.state is ClaimState.TERMINAL
    with pytest.raises(ClaimTransitionError):
        mark_uncertain(claim)
