"""The Lifecycle & scale manager materializes, warms, drains, and stops replicas.

Scale-from-zero registers a bounded cold start, a warm replica becomes joinable and
reports conservative capacity, policy denies over quota or an unlisted model without
allocating, a drain rejects new claims, and an idle teardown only stops a replica that
holds no admitted credit.
"""

import asyncio

from server.resident import (
    LifecycleScaleManager,
    ProvisioningDenialReason,
    ReplicaEndpoint,
    ReplicaState,
    ResidentPolicyLimits,
    ResidentStores,
    ServiceFamily,
)
from tests.server.resident._helpers import PROFILE, warm_stores

_FAMILY = ServiceFamily(family="fam", engine_batch_key="fam", model_ref="m")
_ENDPOINT = ReplicaEndpoint(base_url="http://replica", model="m")


def _manager(stores, **kw):
    limits = kw.pop("limits", ResidentPolicyLimits(max_replicas_per_family=1))
    return LifecycleScaleManager(
        stores,
        limits=limits,
        admission_slots=kw.pop("admission_slots", 2),
        **kw,
    )


def test_scale_from_zero_then_warm():
    stores = ResidentStores()
    stores.families.register(_FAMILY)

    async def materialize_fn(family, replica):
        return "tsk-serve-1"

    mgr = _manager(stores, materialize_fn=materialize_fn)
    assert mgr.plan_capacity("fam", "m").action == "materialize"

    replica = asyncio.run(mgr.materialize(_FAMILY))
    assert replica.state is ReplicaState.MATERIALIZING
    assert replica.serve_task_id == "tsk-serve-1"
    assert stores.leases.by_family("fam")[0].replica_id == replica.replica_id
    # A cold start in progress is not a fresh materialize decision.
    assert mgr.plan_capacity("fam", "m").action == "materialize"

    mgr.on_replica_ready(replica.replica_id, _ENDPOINT)
    assert stores.directory.get(replica.replica_id).state is ReplicaState.WARM
    assert stores.pools.feasible_candidates("fam", PROFILE)
    assert mgr.plan_capacity("fam", "m") == mgr.plan_capacity("fam", "m")
    assert mgr.plan_capacity("fam", "m").action == "join"


def test_policy_denies_over_quota_and_unlisted_model():
    stores = warm_stores()  # one active replica, draining so it is not joinable
    mgr = _manager(stores, limits=ResidentPolicyLimits(max_replicas_per_family=1))
    mgr.drain("rpl-1")
    denied = mgr.plan_capacity("fam", "m")
    assert denied.action == "deny"
    assert denied.denial.reason is ProvisioningDenialReason.QUOTA_EXCEEDED

    gated = _manager(
        ResidentStores(),
        limits=ResidentPolicyLimits(allowed_models=frozenset({"allowed"})),
    )
    decision = gated.plan_capacity("fam", "m")
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

    async def stop_fn(serve_task_id):
        stopped.append(serve_task_id)

    stores.directory.get("rpl-1").serve_task_id = "tsk-serve-1"
    mgr = _manager(stores, stop_fn=stop_fn)
    mgr.drain("rpl-1")
    asyncio.run(mgr.stop("rpl-1"))
    assert stores.directory.get("rpl-1").state is ReplicaState.STOPPED
    assert stopped == ["tsk-serve-1"]


def test_preempt_invalidates_incarnation():
    stores = warm_stores()
    mgr = _manager(stores)
    mgr.on_preempt("rpl-1")
    replica = stores.directory.get("rpl-1")
    assert replica.state is ReplicaState.PREEMPTED
    assert replica.incarnation == 2
