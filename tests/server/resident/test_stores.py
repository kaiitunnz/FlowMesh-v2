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
    reserve(claim, replica_id=replica_id, incarnation=incarnation, credit=_CREDIT)
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


def test_the_claim_indexes_follow_every_transition_and_drop_empty_buckets():
    stores = _stores()
    claim = new_claim(invocation_id="inv-1", family="fam")
    stores.claims.add(claim)
    assert stores.claims.pending_for_family("fam") == [claim]
    assert stores.claims.credit_bearing_for_replica("rpl-1") == []

    reserve(claim, replica_id="rpl-1", incarnation=1, credit=_CREDIT)
    assert stores.claims.pending_for_family("fam") == []
    assert stores.claims.credit_bearing_for_replica("rpl-1") == [claim]
    assert stores.claims.credit_bearing() == [claim]
    assert stores.claims.by_replica("rpl-1") == [claim]

    settle_terminal(claim, ClaimTerminalReason.ENQUEUE_FAILED)
    assert stores.claims.credit_bearing_for_replica("rpl-1") == []
    assert stores.claims.by_replica("rpl-1") == [claim]  # still names its replica

    stores.flush()
    stores.claims.forget(claim.claim_id)
    assert stores.claims.by_invocation("inv-1") == []
    assert stores.claims.by_replica("rpl-1") == []
    index = stores.claims
    assert not (
        index._by_invocation
        or index._by_replica
        or index._credit_by_replica
        or index._pending_by_family
    )


def test_a_lookup_hands_back_a_list_the_store_never_changes():
    stores = _stores()
    claim = new_claim(invocation_id="inv-1", family="fam")
    stores.claims.add(claim)
    pending = stores.claims.pending_for_family("fam")

    reserve(claim, replica_id="rpl-1", incarnation=1, credit=_CREDIT)

    assert pending == [claim]
