"""The Lifecycle & scale manager materializes, warms, drains, and stops replicas.

Scale-from-zero registers a bounded cold start, a warm replica becomes joinable and
reports conservative capacity, policy denies over quota or an unlisted model without
allocating, a drain rejects new claims, and an idle teardown only stops a replica that
holds no admitted credit.
"""

import asyncio

from server.resident import (
    AdmissionProfile,
    ClaimCredit,
    InvocationRequest,
    InvocationSubject,
    InvocationSubjectKind,
    LifecycleScaleManager,
    ProvisioningDenialReason,
    ReplicaEndpoint,
    ReplicaState,
    ResidentPolicyLimits,
    ResidentStores,
    ServiceFamily,
    new_claim,
    reserve,
)
from tests.server.resident._helpers import PROFILE, warm_stores

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="w")
_PAST = "2000-01-01T00:00:00Z"
_FUTURE = "2999-01-01T00:00:00Z"

_FAMILY = ServiceFamily(family="fam", engine_batch_key="fam", service_ref="m")
_ENDPOINT = ReplicaEndpoint(base_url="http://replica", model="m")


def _manager(stores, **kw):
    limits = kw.pop("limits", ResidentPolicyLimits(max_replicas_per_family=1))
    return LifecycleScaleManager(
        stores,
        limits=limits,
        admission_slots=kw.pop("admission_slots", 2),
        **kw,
    )


def test_refresh_report_arms_the_adapter_slot_budget():
    stores = warm_stores()
    mgr = _manager(stores, adapter_slots=2)
    for inv, adapter in (("inv-1", "lora-a"), ("inv-2", "lora-b"), ("inv-3", "lora-a")):
        stores.invocations.put(
            InvocationRequest(
                invocation_id=inv,
                subject=_SUBJECT,
                family="fam",
                profile=AdmissionProfile(engine_batch_key="fam", adapter_ref=adapter),
            )
        )
        claim = new_claim(invocation_id=inv, family="fam", admission_epoch=0)
        reserve(claim, replica_id="rpl-1", incarnation=1, credit=ClaimCredit(slots=1))
        stores.claims.add(claim)

    mgr.refresh_report("rpl-1")
    report = stores.reports.latest("rpl-1")
    # Two distinct adapters held (the repeat shares its slot) against a budget of 2.
    assert report is not None and report.adapter_slots_free == 0


def test_plan_capacity_is_adapter_aware_at_exhaustion():
    stores = warm_stores()
    mgr = _manager(stores, adapter_slots=1)
    # Fill the single adapter slot with a held claim for lora-a on the warm replica.
    stores.invocations.put(
        InvocationRequest(
            invocation_id="inv-1",
            subject=_SUBJECT,
            family="fam",
            profile=AdmissionProfile(engine_batch_key="fam", adapter_ref="lora-a"),
        )
    )
    claim = new_claim(invocation_id="inv-1", family="fam", admission_epoch=0)
    reserve(claim, replica_id="rpl-1", incarnation=1, credit=ClaimCredit(slots=1))
    stores.claims.add(claim)

    base = AdmissionProfile(engine_batch_key="fam")
    same = AdmissionProfile(engine_batch_key="fam", adapter_ref="lora-a")
    distinct = AdmissionProfile(engine_batch_key="fam", adapter_ref="lora-b")

    assert mgr.plan_capacity(_FAMILY, base).action == "join"
    assert mgr.plan_capacity(_FAMILY, same).action == "join"
    denied = mgr.plan_capacity(_FAMILY, distinct)
    assert denied.action == "deny"
    assert denied.denial is not None
    assert denied.denial.reason is ProvisioningDenialReason.ADAPTER_SLOT_CAP


def _hold_adapter(stores, inv, adapter, replica_id="rpl-1"):
    stores.invocations.put(
        InvocationRequest(
            invocation_id=inv,
            subject=_SUBJECT,
            family="fam",
            profile=AdmissionProfile(engine_batch_key="fam", adapter_ref=adapter),
        )
    )
    claim = new_claim(invocation_id=inv, family="fam", admission_epoch=0)
    reserve(claim, replica_id=replica_id, incarnation=1, credit=ClaimCredit(slots=1))
    stores.claims.add(claim)


def test_replica_holds_adapter_reads_the_credit_bearing_set():
    stores = warm_stores()
    mgr = _manager(stores, adapter_slots=2)
    _hold_adapter(stores, "inv-1", "lora-a")
    assert mgr.replica_holds_adapter("rpl-1", "lora-a") is True
    assert mgr.replica_holds_adapter("rpl-1", "lora-b") is False


def test_adapter_slot_cap_denial_surfaces_the_co_occurring_quota_reason():
    stores = warm_stores()
    mgr = _manager(stores, adapter_slots=1)
    # The single adapter slot is full and the family is at its one-replica quota, so a
    # new distinct adapter can neither fit nor add a replica: the denial names both.
    _hold_adapter(stores, "inv-1", "lora-a")
    denied = mgr.plan_capacity(
        _FAMILY, AdmissionProfile(engine_batch_key="fam", adapter_ref="lora-b")
    )
    assert denied.action == "deny" and denied.denial is not None
    assert denied.denial.reason is ProvisioningDenialReason.ADAPTER_SLOT_CAP
    assert ProvisioningDenialReason.QUOTA_EXCEEDED.value in (denied.denial.detail or "")


def test_scale_from_zero_then_warm():
    stores = ResidentStores()
    stores.families.register(_FAMILY)

    async def materialize_fn(family, replica):
        return "tsk-serve-1"

    mgr = _manager(stores, materialize_fn=materialize_fn)
    assert mgr.plan_capacity(_FAMILY).action == "materialize"

    replica = asyncio.run(mgr.materialize(_FAMILY))
    assert replica.state is ReplicaState.MATERIALIZING
    assert replica.serve_task_id == "tsk-serve-1"
    assert stores.leases.by_family("fam")[0].replica_id == replica.replica_id
    # A cold start in progress is not a fresh materialize decision.
    assert mgr.plan_capacity(_FAMILY).action == "materialize"

    mgr.on_replica_ready(replica.replica_id, _ENDPOINT)
    assert stores.directory.get(replica.replica_id).state is ReplicaState.WARM
    assert stores.pools.feasible_candidates("fam", PROFILE)
    assert mgr.plan_capacity(_FAMILY) == mgr.plan_capacity(_FAMILY)
    assert mgr.plan_capacity(_FAMILY).action == "join"


def test_policy_denies_over_quota_and_unlisted_model():
    stores = warm_stores()  # one active replica, draining so it is not joinable
    mgr = _manager(stores, limits=ResidentPolicyLimits(max_replicas_per_family=1))
    mgr.drain("rpl-1")
    denied = mgr.plan_capacity(_FAMILY)
    assert denied.action == "deny"
    assert denied.denial.reason is ProvisioningDenialReason.QUOTA_EXCEEDED

    gated = _manager(
        ResidentStores(),
        limits=ResidentPolicyLimits(allowed_models=frozenset({"allowed"})),
    )
    decision = gated.plan_capacity(_FAMILY)
    assert decision.action == "deny"
    assert decision.denial.reason is ProvisioningDenialReason.MODEL_NOT_ALLOWED


def test_drain_rejects_new_claims():
    stores = warm_stores()
    mgr = _manager(stores)
    mgr.drain("rpl-1")
    assert stores.directory.get("rpl-1").state is ReplicaState.DRAINING
    assert stores.pools.feasible_candidates("fam", PROFILE) == []


def test_idle_teardown_only_stops_uncommitted_replica():
    stores = warm_stores()
    stopped = []

    def stop_fn(serve_task_id):
        stopped.append(serve_task_id)

    stores.directory.get("rpl-1").serve_task_id = "tsk-serve-1"
    mgr = _manager(stores, stop_fn=stop_fn)
    mgr.drain("rpl-1")
    mgr.stop("rpl-1")
    assert stores.directory.get("rpl-1").state is ReplicaState.STOPPED
    assert stopped == ["tsk-serve-1"]


def test_preempt_invalidates_incarnation_and_reaps_serve_task():
    stores = warm_stores()
    stopped = []

    def stop_fn(serve_task_id):
        stopped.append(serve_task_id)

    stores.directory.get("rpl-1").serve_task_id = "tsk-serve-1"
    mgr = _manager(stores, stop_fn=stop_fn)
    mgr.on_preempt("rpl-1")
    replica = stores.directory.get("rpl-1")
    assert replica.state is ReplicaState.PREEMPTED
    assert replica.incarnation == 2
    assert stopped == ["tsk-serve-1"]


def test_preempt_never_reaps_a_standing_serve_replica():
    # A standing serve replica is the user's own long-running task: a per-request
    # failure that reaches preempt must never invalidate the incarnation or cancel the
    # backing serve task, or one bad request would tear the endpoint down for every
    # client. It cannot re-materialize, so preempt-and-recreate is the wrong recovery.
    stores = warm_stores()
    stopped = []
    replica = stores.directory.get("rpl-1")
    replica.serve_task_id = "tsk-serve-1"
    replica.standing = True
    mgr = _manager(stores, stop_fn=lambda tid: stopped.append(tid))
    mgr.on_preempt("rpl-1")
    replica = stores.directory.get("rpl-1")
    assert replica.state is ReplicaState.WARM
    assert replica.incarnation == 1
    assert stopped == []


def test_idle_sweep_drains_then_stops_an_idle_replica():
    stores = warm_stores()
    stopped = []

    def stop_fn(serve_task_id):
        stopped.append(serve_task_id)

    replica = stores.directory.get("rpl-1")
    replica.serve_task_id = "tsk-serve-1"
    replica.last_active_at = _PAST
    mgr = _manager(stores, stop_fn=stop_fn, idle_retain_sec=30.0)

    mgr.sweep_idle()
    assert stores.directory.get("rpl-1").state is ReplicaState.DRAINING
    assert stopped == []  # drained first, stopped on the next sweep

    mgr.sweep_idle()
    assert stores.directory.get("rpl-1").state is ReplicaState.STOPPED
    assert stopped == ["tsk-serve-1"]


def test_idle_sweep_retains_a_recent_replica_and_when_disabled():
    disabled = warm_stores()
    disabled.directory.get("rpl-1").last_active_at = _PAST
    _manager(disabled, idle_retain_sec=0.0).sweep_idle()
    assert disabled.directory.get("rpl-1").state is ReplicaState.WARM

    recent = warm_stores()
    recent.directory.get("rpl-1").last_active_at = _FUTURE
    _manager(recent, idle_retain_sec=1.0).sweep_idle()
    assert recent.directory.get("rpl-1").state is ReplicaState.WARM


def test_idle_sweep_holds_a_replica_that_still_holds_credit():
    stores = warm_stores()
    stores.directory.get("rpl-1").last_active_at = _PAST
    claim = new_claim(invocation_id="inv-x", family="fam")
    reserve(claim, replica_id="rpl-1", incarnation=1, credit=ClaimCredit(slots=1))
    stores.claims.add(claim)
    _manager(stores, idle_retain_sec=1.0).sweep_idle()
    assert stores.directory.get("rpl-1").state is ReplicaState.WARM
