"""The Admission controller advances claims and their credits over the stores.

A claim reserves a fenced credit on a warm replica, outstanding credit cannot overcommit
a replica, an accepted credit releases only from a fenced DS terminal consumed by
``invocation_id``, and a reissue reconciles a prior lost claim into a fresh successor
under the same invocation.
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

    handoff = ctl.admit(claim, PROFILE)
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
    assert ctl.admit(first, PROFILE) is not None

    second = _raise(ctl, "inv-2")
    assert ctl.admit(second, PROFILE) is None
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


def test_route_loss_holds_credit_then_reconciles():
    stores = warm_stores()
    ctl = AdmissionController(stores)
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE)
    ctl.on_enqueue_ack(claim)

    ctl.on_route_loss(claim)
    assert claim.state is ClaimState.UNCERTAIN
    assert stores.credit_ledger.held("rpl-1") == 1

    ctl.reconcile("inv-1")
    assert claim.state is ClaimState.TERMINAL
    assert claim.terminal_reason is ClaimTerminalReason.RECONCILED
    assert stores.credit_ledger.held("rpl-1") == 0


def test_reissue_reconciles_prior_claim_into_successor():
    stores = warm_stores()
    ctl = AdmissionController(stores)
    first = _raise(ctl, "inv-1")
    ctl.admit(first, PROFILE)
    ctl.on_enqueue_ack(first)
    ctl.on_route_loss(first)  # a lost attempt still holding credit

    successor = _raise(ctl, "inv-1")
    assert first.state is ClaimState.TERMINAL
    assert first.terminal_reason is ClaimTerminalReason.RECONCILED
    assert successor.admission_epoch == first.admission_epoch + 1
    assert successor.state is ClaimState.PENDING
    assert stores.credit_ledger.held("rpl-1") == 0


def test_persist_hook_fires_on_mutation():
    stores = warm_stores()
    calls = []
    ctl = AdmissionController(stores, persist=lambda: calls.append(1))
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE)
    ctl.on_enqueue_ack(claim)
    ctl.on_ds_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    assert len(calls) >= 4
