"""The Admission controller advances claims and their credits over the stores.

A claim reserves a fenced credit on a warm replica, outstanding credit cannot overcommit
a replica, an accepted credit releases only from a fenced DS terminal consumed by
``invocation_id`` (tolerant of any credit-bearing source), and a re-drive resumes the
in-flight claim rather than minting a successor before that terminal.
"""

from server.resident import (
    AdmissionController,
    ClaimState,
    ClaimTerminalReason,
)
from tests.server.resident._helpers import PROFILE, warm_stores


def _raise(ctl, invocation_id="inv-1"):
    return ctl.raise_claim(
        invocation_id=invocation_id,
        workflow_id="wfl-1",
        family="fam",
        profile=PROFILE,
    )


def test_reserve_accept_release_cycle():
    stores = warm_stores(slots=2)
    ctl = AdmissionController(stores)
    claim = _raise(ctl)

    handoff = ctl.admit(claim, PROFILE, idempotency_key="idm-x")
    assert handoff is not None
    assert handoff.replica_id == "rpl-1" and handoff.incarnation == 1
    assert claim.state is ClaimState.RESERVED
    assert stores.credit_ledger.held("rpl-1") == 1
    assert stores.demand.get(claim.claim_id).admitted is True

    ctl.on_enqueue_ack(claim)
    assert claim.state is ClaimState.STREAMING
    assert stores.credit_ledger.held("rpl-1") == 1

    ctl.on_ds_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held("rpl-1") == 0


def test_admission_never_overcommits():
    stores = warm_stores(slots=1)
    ctl = AdmissionController(stores)
    first = _raise(ctl, "inv-1")
    assert ctl.admit(first, PROFILE, idempotency_key="idm-x") is not None

    second = _raise(ctl, "inv-2")
    assert ctl.admit(second, PROFILE, idempotency_key="idm-x") is None
    assert second.state is ClaimState.PENDING


def test_denied_and_expired_terminate_pending_without_credit():
    stores = warm_stores()
    ctl = AdmissionController(stores)
    denied = _raise(ctl, "inv-1")
    ctl.on_denied(denied)
    assert denied.state is ClaimState.TERMINAL
    assert denied.terminal_reason is ClaimTerminalReason.CANCELLED
    assert stores.demand.get(denied.claim_id) is None

    expired = _raise(ctl, "inv-2")
    ctl.on_expired(expired)
    assert expired.terminal_reason is ClaimTerminalReason.EXPIRED


def test_route_loss_holds_credit_until_ds_terminal():
    stores = warm_stores()
    ctl = AdmissionController(stores)
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE, idempotency_key="idm-x")
    ctl.on_enqueue_ack(claim)

    ctl.on_route_loss(claim)
    assert claim.state is ClaimState.UNCERTAIN
    assert stores.credit_ledger.held("rpl-1") == 1  # held, not released on loss

    # Only the fenced DS terminal releases the credit.
    ctl.on_ds_terminal("inv-1", ClaimTerminalReason.FAILED)
    assert claim.state is ClaimState.TERMINAL
    assert claim.terminal_reason is ClaimTerminalReason.FAILED
    assert stores.credit_ledger.held("rpl-1") == 0


def test_redrive_resumes_the_in_flight_claim_then_a_successor_after_terminal():
    stores = warm_stores()
    ctl = AdmissionController(stores)
    first = _raise(ctl, "inv-1")
    ctl.admit(first, PROFILE, idempotency_key="idm-x")
    ctl.on_enqueue_ack(first)
    ctl.on_route_loss(first)  # a lost attempt, parked and still holding credit

    # A re-drive attaches to the in-flight claim (dedup) and can resume its replica; it
    # does not mint a successor or release the parked credit.
    assert ctl.active_claim("inv-1") is first
    assert ctl.rebuild_handoff(first, idempotency_key="idm-x") is not None
    assert stores.credit_ledger.held("rpl-1") == 1

    # Only after the fenced DS terminal releases it is a fresh claim a successor.
    ctl.on_ds_terminal("inv-1", ClaimTerminalReason.FAILED)
    assert ctl.active_claim("inv-1") is None
    assert stores.credit_ledger.held("rpl-1") == 0
    successor = _raise(ctl, "inv-1")
    assert successor.admission_epoch == first.admission_epoch + 1
    assert successor.state is ClaimState.PENDING


def test_ds_terminal_releases_a_reserved_credit_on_cancellation():
    # A cancellation while the adapter issue is in flight reaches on_ds_terminal on a
    # still-RESERVED claim; the release tolerates any credit-bearing source state.
    stores = warm_stores()
    ctl = AdmissionController(stores)
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE, idempotency_key="idm-x")
    assert claim.state is ClaimState.RESERVED

    ctl.on_ds_terminal("inv-1", ClaimTerminalReason.CANCELLED)
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held("rpl-1") == 0


def test_accept_and_authorize_issues_a_fenced_route_authorization():
    stores = warm_stores()
    ctl = AdmissionController(stores)
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE, idempotency_key="idm-x")

    auth = ctl.accept_and_authorize(
        claim,
        idempotency_key="idm-x",
        origin_id="rog-1",
        operation="inference",
        deadline_at="2026-01-01T00:00:00Z",
    )
    assert claim.state is ClaimState.ACCEPTED
    # The fence binds the accepted claim's subject, request identity, and incarnation.
    assert auth.claim_id == claim.claim_id
    assert auth.invocation_id == "inv-1"
    assert auth.idempotency_key == "idm-x"
    assert auth.origin_id == "rog-1"
    assert auth.replica_id == "rpl-1" and auth.incarnation == 1
    assert auth.route_auth_epoch == 1

    ctl.on_stream_started(claim)
    assert claim.state is ClaimState.STREAMING
    assert (
        stores.credit_ledger.held("rpl-1") == 1
    )  # authorizing the stream holds credit


def test_persist_hook_fires_on_mutation():
    stores = warm_stores()
    calls = []
    ctl = AdmissionController(stores, persist=lambda: calls.append(1))
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE, idempotency_key="idm-x")
    ctl.on_enqueue_ack(claim)
    ctl.on_ds_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    assert len(calls) >= 4
