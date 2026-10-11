"""The Admission controller advances claims and their credits over the stores.

A claim reserves a fenced credit on a warm replica, outstanding credit cannot overcommit
a replica, an accepted credit releases only from a fenced terminal fact consumed by
``invocation_id`` (tolerant of any credit-bearing source), and a re-drive resumes the
in-flight claim rather than minting a successor before that terminal.
"""

import json
from collections.abc import Mapping

import pytest

from server.resident import (
    AdmissionController,
    ClaimState,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ResidentStores,
)
from server.resident.durability import MemoryResidentDurability
from tests.server.resident._helpers import PROFILE, warm_stores


def _raise(ctl, invocation_id="inv-1"):
    return ctl.raise_claim(
        invocation_id=invocation_id,
        subject=InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1"),
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

    ctl.accept_and_authorize(claim, idempotency_key="idm-x", origin_id="rog-1")
    ctl.on_stream_started(claim)
    assert claim.state is ClaimState.STREAMING
    assert stores.credit_ledger.held("rpl-1") == 1

    ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
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
    ctl.accept_and_authorize(claim, idempotency_key="idm-x", origin_id="rog-1")
    ctl.on_stream_started(claim)

    ctl.on_route_loss(claim)
    assert claim.state is ClaimState.UNCERTAIN
    assert stores.credit_ledger.held("rpl-1") == 1  # held, not released on loss

    # Only the fenced DS terminal releases the credit.
    ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.FAILED)
    assert claim.state is ClaimState.TERMINAL
    assert claim.terminal_reason is ClaimTerminalReason.FAILED
    assert stores.credit_ledger.held("rpl-1") == 0


def test_redrive_resumes_the_in_flight_claim_then_a_successor_after_terminal():
    stores = warm_stores()
    ctl = AdmissionController(stores)
    first = _raise(ctl, "inv-1")
    ctl.admit(first, PROFILE, idempotency_key="idm-x")
    ctl.accept_and_authorize(first, idempotency_key="idm-x", origin_id="rog-1")
    ctl.on_stream_started(first)
    ctl.on_route_loss(first)  # a lost attempt, parked and still holding credit

    # A re-drive attaches to the in-flight claim (dedup) and can resume its replica; it
    # does not mint a successor or release the parked credit.
    assert ctl.active_claim("inv-1") is first
    assert ctl.rebuild_handoff(first, idempotency_key="idm-x") is not None
    assert stores.credit_ledger.held("rpl-1") == 1

    # Only after the fenced DS terminal releases it is a fresh claim a successor.
    ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.FAILED)
    assert ctl.active_claim("inv-1") is None
    assert stores.credit_ledger.held("rpl-1") == 0
    successor = _raise(ctl, "inv-1")
    assert successor.admission_epoch == first.admission_epoch + 1
    assert successor.state is ClaimState.PENDING


def test_ds_terminal_releases_a_reserved_credit_on_cancellation():
    # A cancellation while the adapter issue is in flight settles a still-RESERVED
    # claim; the release tolerates any credit-bearing source state.
    stores = warm_stores()
    ctl = AdmissionController(stores)
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE, idempotency_key="idm-x")
    assert claim.state is ClaimState.RESERVED

    ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.CANCELLED)
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
        deadline_at="2026-01-01T00:00:00Z",
    )
    assert claim.state is ClaimState.ACCEPTED
    # The fence binds the accepted claim's subject, request identity, and incarnation.
    assert auth.claim_id == claim.claim_id
    assert auth.invocation_id == "inv-1"
    assert auth.idempotency_key == "idm-x"
    assert auth.origin_id == "rog-1"
    assert auth.replica_id == "rpl-1" and auth.incarnation == 1

    ctl.on_stream_started(claim)
    assert claim.state is ClaimState.STREAMING
    assert (
        stores.credit_ledger.held("rpl-1") == 1
    )  # authorizing the stream holds credit


class _Durability(MemoryResidentDurability):
    """Records each write's changed fields, and refuses writes while ``down``."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[dict[str, str | None]] = []
        self.down = False

    def write(self, changes: Mapping[str, str | None]) -> None:
        if self.down:
            raise ConnectionError("control redis unavailable")
        self.writes.append(dict(changes))
        super().write(changes)


def _durable_stores() -> tuple[ResidentStores, _Durability]:
    durability = _Durability()
    stores = warm_stores(durability=durability)
    stores.flush()
    durability.writes.clear()
    return stores, durability


def test_each_step_writes_only_what_it_changed():
    stores, durability = _durable_stores()
    ctl = AdmissionController(stores)
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE, idempotency_key="idm-x")
    ctl.accept_and_authorize(claim, idempotency_key="idm-x", origin_id="rog-1")
    ctl.on_stream_started(claim)
    ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)

    claim_field, request_field = f"claim:{claim.claim_id}", "request:inv-1"
    assert [sorted(write) for write in durability.writes] == [
        sorted([claim_field, request_field]),
        sorted([claim_field, "replica:rpl-1"]),
        [claim_field],
        [claim_field],
        sorted([claim_field, request_field, "replica:rpl-1"]),
    ]


def test_a_terminal_whose_persist_failed_is_finished_by_its_replay():
    stores, durability = _durable_stores()
    ctl = AdmissionController(stores)
    released = []
    ctl.set_release_hook(released.append)
    claim = _raise(ctl)
    ctl.admit(claim, PROFILE, idempotency_key="idm-x")
    ctl.accept_and_authorize(claim, idempotency_key="idm-x", origin_id="rog-1")
    durability.writes.clear()

    durability.down = True
    with pytest.raises(ConnectionError):
        ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.FAILED)
    assert claim.state is ClaimState.TERMINAL
    assert released == []
    assert not stores.durable
    durability.down = False

    # The replay persists the terminal and tells the replica once, keeping the first
    # terminal's reason.
    assert ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    assert claim.terminal_reason is ClaimTerminalReason.FAILED
    stored = json.loads(durability.live[f"claim:{claim.claim_id}"])
    assert stored["state"] == "terminal" and stored["terminal_reason"] == "failed"
    assert json.loads(durability.live["request:inv-1"])["consumed"] == "failed"
    assert released == ["rpl-1"]
    assert stores.credit_ledger.held("rpl-1") == 0

    # A terminal handed over after its consumption finished does nothing more.
    writes = len(durability.writes)
    assert not ctl.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    assert len(durability.writes) == writes
    assert released == ["rpl-1"]
