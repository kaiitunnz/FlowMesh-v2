"""Authoritative stores and their derived, non-diverging views.

These prove the Admission-credit ledger and CapacityPools are recomputed from the
authoritative ``ServiceClaim`` facts — outstanding credit cannot overcommit slots, a
terminal claim releases its debit, capacity reports are fenced by incarnation and report
epoch, and a superseded incarnation drops out of selection.
"""

from server.resident import (
    AdmissionProfile,
    ClaimCredit,
    ClaimTerminalReason,
    ReplicaCapacityReport,
    ReplicaIncarnation,
    ReplicaState,
    ResidentStores,
    SafeCapacityVector,
    ServiceFamily,
    accept,
    new_claim,
    reserve,
    settle_terminal,
)

_PROFILE = AdmissionProfile(engine_batch_key="k")
_CREDIT = ClaimCredit(slots=1)


def _stores(*, slots=2, incarnation=1):
    stores = ResidentStores()
    stores.families.register(
        ServiceFamily(family="fam", engine_batch_key="k", model_ref="m")
    )
    stores.directory.add(
        ReplicaIncarnation(
            replica_id="rpl-1",
            family="fam",
            incarnation=incarnation,
            state=ReplicaState.WARM,
            healthy=True,
        )
    )
    stores.reports.ingest(
        ReplicaCapacityReport(
            replica_id="rpl-1",
            incarnation=incarnation,
            report_epoch=1,
            state=ReplicaState.WARM,
            healthy=True,
            safe=SafeCapacityVector(admission_slots=slots),
        )
    )
    return stores


def _reserve_on(stores, replica_id="rpl-1", incarnation=1):
    claim = new_claim(invocation_id=f"inv-{len(stores.claims.all())}", family="fam")
    reserve(
        claim,
        replica_id=replica_id,
        incarnation=incarnation,
        report_epoch=1,
        credit=_CREDIT,
    )
    stores.claims.add(claim)
    return claim


def test_credit_ledger_recomputes_and_never_overcommits():
    stores = _stores(slots=2)
    assert len(stores.pools.feasible_candidates("fam", _PROFILE)) == 1

    _reserve_on(stores)
    _reserve_on(stores)
    assert stores.credit_ledger.held("rpl-1") == 2
    # Safe slots are fully consumed: no further claim is feasible.
    assert stores.pools.feasible_candidates("fam", _PROFILE) == []


def test_terminal_claim_releases_derived_credit():
    stores = _stores(slots=2)
    claim = _reserve_on(stores)
    _reserve_on(stores)
    assert stores.credit_ledger.held("rpl-1") == 2

    accept(claim)
    settle_terminal(claim, ClaimTerminalReason.COMPLETED)
    assert stores.credit_ledger.held("rpl-1") == 1
    assert len(stores.pools.feasible_candidates("fam", _PROFILE)) == 1


def test_uncertain_claim_still_holds_credit():
    stores = _stores(slots=1)
    claim = _reserve_on(stores)
    from server.resident import accept, mark_uncertain

    accept(claim)
    mark_uncertain(claim)
    assert stores.credit_ledger.held("rpl-1") == 1
    assert stores.pools.feasible_candidates("fam", _PROFILE) == []


def test_report_store_fences_stale_telemetry():
    stores = _stores()
    fresh = ReplicaCapacityReport(
        replica_id="rpl-1",
        incarnation=1,
        report_epoch=5,
        state=ReplicaState.WARM,
        healthy=True,
        safe=SafeCapacityVector(admission_slots=4),
    )
    assert stores.reports.ingest(fresh)
    stale_epoch = fresh.model_copy(update={"report_epoch": 3})
    assert not stores.reports.ingest(stale_epoch)
    stale_incarnation = fresh.model_copy(update={"incarnation": 0, "report_epoch": 99})
    assert not stores.reports.ingest(stale_incarnation)
    assert stores.reports.latest("rpl-1").report_epoch == 5


def test_superseded_incarnation_drops_from_selection():
    stores = _stores(incarnation=1)
    # Directory advances the replica to a new incarnation; the report fences the old.
    stores.directory.get("rpl-1").incarnation = 2
    assert stores.pools.feasible_candidates("fam", _PROFILE) == []
