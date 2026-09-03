"""Resident-capacity control drives an invocation from demand to a released credit.

With a stubbed substrate a resident invocation with no pre-enabled capacity
materializes scale-from-zero, reserves and accepts a claim, settles a real completion,
and
releases the credit only on the fenced DS terminal. A disallowed model produces a typed
denial with no allocation or credit, and a restart reconciles an in-flight claim to
uncertain.
"""

import asyncio

from server.orchestration.tool_dispatch import ToolInvocationEnvelope
from server.resident import (
    AdapterError,
    AdmissionController,
    AdmissionProfile,
    ClaimState,
    LifecycleScaleManager,
    ReplicaEndpoint,
    ReplicaState,
    ResidentCapacityControl,
    ResidentPolicyLimits,
    ResidentStores,
)
from server.task.v2.representations.operators import (
    AgentModelGatewayBinding,
    BindingProvenance,
    ModelBindingProvenance,
)
from shared.harness import BoundaryEventKind
from shared.tasks.specs import ModelBindingMode

_PROV = ModelBindingProvenance(
    mode=BindingProvenance.SOURCE,
    url=BindingProvenance.SOURCE,
    model=BindingProvenance.SOURCE,
)


def _binding(model_ref="m"):
    return AgentModelGatewayBinding(
        mode=ModelBindingMode.RESIDENT, service_model_ref=model_ref, provenance=_PROV
    )


def _env(invocation_id="inv-1"):
    return ToolInvocationEnvelope(
        kind=BoundaryEventKind.INVOCATION,
        interface="model",
        invocation_id=invocation_id,
        task_id="tsk-1",
        activation_id="act-1",
        call_correlation="c1",
        request_payload='{"prompt": "hi"}',
    )


class _FakeAdapter:
    def __init__(self, completion="OK"):
        self.completion = completion
        self.calls = []

    async def issue(self, endpoint, request_payload):
        self.calls.append(endpoint)
        return self.completion


class _FlakyAdapter:
    """Fails post-acceptance for the first ``fail_first`` calls, then succeeds."""

    def __init__(self, completion="OK", fail_first=1):
        self.completion = completion
        self.remaining_failures = fail_first
        self.calls = 0

    async def issue(self, endpoint, request_payload):
        self.calls += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise AdapterError("route lost", pre_acceptance=False)
        return self.completion


class _DeadThenLiveAdapter:
    """Unreachable for the first delivery (a dead replica), reachable thereafter."""

    def __init__(self, completion="OK"):
        self.completion = completion
        self.first = True

    async def issue(self, endpoint, request_payload):
        if self.first:
            self.first = False
            raise AdapterError(
                "connection refused", pre_acceptance=True, connection_failure=True
            )
        return self.completion


class _StatusAdapter:
    """Fails pre-acceptance with a transient HTTP status (a live replica), never a
    connection failure."""

    async def issue(self, endpoint, request_payload):
        raise AdapterError("503", pre_acceptance=True, connection_failure=False)


def _build(*, limits=None, adapter=None, materialize_fn=None):
    stores = ResidentStores()
    limits = limits or ResidentPolicyLimits()
    settled = []

    def settle_cb(task_id, call_correlation, value, *, error=None):
        settled.append((task_id, call_correlation, value, error))
        return True

    if materialize_fn is None:

        async def materialize_fn(family, replica):
            return "tsk-serve-1"

    admission = AdmissionController(stores)
    lifecycle = LifecycleScaleManager(
        stores, limits=limits, admission_slots=2, materialize_fn=materialize_fn
    )
    svc = ResidentCapacityControl(
        stores=stores,
        admission=admission,
        lifecycle=lifecycle,
        adapter=adapter or _FakeAdapter(),
        limits=limits,
        binding_resolver=lambda task_id: ("wfl-1", _binding()),
        settle_cb=settle_cb,
        redispatch_cb=lambda *a, **k: False,
        endpoint_probe=lambda serve_task_id: ReplicaEndpoint(
            base_url="http://replica", model="m"
        ),
        poll_interval_sec=0.01,
    )
    return svc, stores, settled


def test_scale_from_zero_to_completion_and_credit_release():
    svc, stores, settled = _build()
    asyncio.run(svc._serve_invocation(_env()))

    assert settled == [("tsk-1", "c1", "OK", None)]
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.STREAMING
    assert stores.credit_ledger.held(claim.replica_id) == 1
    assert stores.directory.get(claim.replica_id).state is ReplicaState.WARM

    # The fenced DS terminal — consumed by invocation_id — releases the credit.
    svc.on_invocation_terminal("inv-1")
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0


def test_disallowed_model_denies_without_allocation_or_credit():
    limits = ResidentPolicyLimits(allowed_models=frozenset({"approved-only"}))
    svc, stores, settled = _build(limits=limits)
    asyncio.run(svc._serve_invocation(_env()))

    assert len(settled) == 1 and settled[0][3] is not None
    assert "model_not_allowed" in settled[0][3]
    assert stores.directory.all() == []
    assert stores.claims.all() == []


def test_rehydrate_reconciles_in_flight_claim():
    svc, stores, _ = _build()
    asyncio.run(svc._serve_invocation(_env()))
    snapshot = stores.to_snapshot()

    fresh_svc, fresh_stores, _ = _build()
    fresh_svc.rehydrate(snapshot)
    claim = fresh_stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.UNCERTAIN
    assert fresh_stores.credit_ledger.held(claim.replica_id) == 1


def test_redrive_after_rehydrate_resumes_under_one_credit():
    # The startup ordering (bind + rehydrate the claim store, then let the runtime
    # re-drive suspended boundaries) exists so the re-drive finds the loaded in-flight
    # claim and resumes on it. If the store were empty when the boundary re-drove, the
    # loaded credit would strand and a second be admitted for the same invocation.
    svc, stores, _ = _build()
    asyncio.run(svc._serve_invocation(_env()))
    snapshot = stores.to_snapshot()

    fresh_svc, fresh_stores, fresh_settled = _build()
    fresh_svc.rehydrate(snapshot)  # loads the in-flight claim as UNCERTAIN, credit held
    resumed = fresh_stores.claims.by_invocation("inv-1")[0]
    assert resumed.state is ClaimState.UNCERTAIN
    assert fresh_stores.credit_ledger.held(resumed.replica_id) == 1

    # The runtime re-drives the suspended boundary: it resumes on the existing claim.
    asyncio.run(fresh_svc._serve_invocation(_env()))
    claims = fresh_stores.claims.by_invocation("inv-1")
    assert len(claims) == 1  # resumed, not re-admitted as a successor
    assert (
        fresh_stores.credit_ledger.held(resumed.replica_id) == 1
    )  # exactly one credit
    assert fresh_settled[-1] == ("tsk-1", "c1", "OK", None)

    # The fenced DS terminal then releases that single credit — none stranded.
    fresh_svc.on_invocation_terminal("inv-1")
    assert fresh_stores.credit_ledger.held(resumed.replica_id) == 0


def test_rehydrate_reports_a_warm_replica_so_it_is_admittable_again():
    svc, stores, _ = _build()
    asyncio.run(svc._serve_invocation(_env()))
    snapshot = stores.to_snapshot()  # capacity reports are not persisted

    fresh_svc, fresh_stores, _ = _build()
    fresh_svc.rehydrate(snapshot)
    fam = fresh_stores.directory.all()[0].family
    # Re-reported on rehydrate, the warm replica is feasible again — without the
    # re-report a resident invocation would spin to the cold-start budget forever.
    assert fresh_stores.pools.feasible_candidates(
        fam, AdmissionProfile(engine_batch_key=fam)
    )


def test_rehydrate_preempts_a_warm_replica_whose_serve_task_is_gone():
    svc, stores, _ = _build()
    asyncio.run(svc._serve_invocation(_env()))
    snapshot = stores.to_snapshot()

    fresh_svc, fresh_stores, _ = _build()
    fresh_svc._probe_endpoint = lambda serve_task_id: None  # serve task is gone
    fresh_svc.rehydrate(snapshot)
    states = {r.state for r in fresh_stores.directory.all()}
    assert ReplicaState.WARM not in states
    assert ReplicaState.PREEMPTED in states


def test_connection_failure_invalidates_replica_and_self_heals():
    svc, stores, settled = _build(adapter=_DeadThenLiveAdapter())

    # First invocation: the replica is unreachable, so it is invalidated and the
    # boundary settles an error; no servable replica or held credit remains.
    asyncio.run(svc._serve_invocation(_env("inv-1")))
    assert settled[-1][3] is not None
    assert any(r.state is ReplicaState.PREEMPTED for r in stores.directory.all())
    assert all(not c.holds_credit for c in stores.claims.all())
    fam = stores.directory.all()[0].family
    assert not stores.pools.feasible_candidates(
        fam, AdmissionProfile(engine_batch_key=fam)
    )

    # The next invocation self-heals: the family re-materializes from zero.
    asyncio.run(svc._serve_invocation(_env("inv-2")))
    assert settled[-1] == ("tsk-1", "c1", "OK", None)
    warm = [r for r in stores.directory.all() if r.state is ReplicaState.WARM]
    assert len(warm) == 1


def test_transient_pre_acceptance_status_does_not_invalidate_the_replica():
    svc, stores, settled = _build(adapter=_StatusAdapter())
    asyncio.run(svc._serve_invocation(_env()))

    # The reserved credit is released, but a live replica returning a transient
    # status is not invalidated.
    assert settled[-1][3] is not None
    assert [r for r in stores.directory.all() if r.state is ReplicaState.WARM]
    assert all(r.state is not ReplicaState.PREEMPTED for r in stores.directory.all())


def test_serve_settles_an_error_when_an_internal_path_raises():
    svc, stores, settled = _build()

    def boom(task_id):
        raise RuntimeError("resolver exploded")

    svc._resolve_binding = boom
    asyncio.run(svc._serve_invocation(_env()))
    assert len(settled) == 1
    assert settled[0][3] is not None and "resident serve error" in settled[0][3]


def test_post_acceptance_loss_holds_then_releases_on_ds_terminal():
    svc, stores, settled = _build(adapter=_FlakyAdapter(fail_first=1))
    asyncio.run(svc._serve_invocation(_env()))

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.UNCERTAIN  # held, not released on a lost route
    assert stores.credit_ledger.held(claim.replica_id) == 1
    assert settled[-1][3] is not None  # the boundary settled with an error

    # The runtime fires the terminal hook on the failure settle, releasing the credit.
    svc.on_invocation_terminal("inv-1", failed=True)
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0


def test_redrive_resumes_without_double_admit_or_release():
    svc, stores, _ = _build(adapter=_FlakyAdapter(fail_first=1))
    asyncio.run(svc._serve_invocation(_env()))  # first drive: post-acceptance loss
    first = stores.claims.by_invocation("inv-1")[0]
    assert first.state is ClaimState.UNCERTAIN
    replicas = len(stores.directory.all())
    assert stores.credit_ledger.held(first.replica_id) == 1

    # A re-drive of the same invocation resumes the parked claim on its live replica: no
    # new claim, no new materialize, and the credit is not released before a terminal.
    asyncio.run(svc._serve_invocation(_env()))
    claims = stores.claims.by_invocation("inv-1")
    assert len(claims) == 1 and claims[0] is first
    assert len(stores.directory.all()) == replicas
    assert stores.credit_ledger.held(first.replica_id) == 1

    svc.on_invocation_terminal("inv-1", failed=False)
    assert first.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(first.replica_id) == 0


def test_failed_materialize_recovers_family_and_settles():
    async def boom(family, replica):
        raise RuntimeError("cold start failed")

    svc, stores, settled = _build(materialize_fn=boom)
    asyncio.run(svc._serve_invocation(_env()))

    # The invocation settled with a typed provisioning error rather than hanging.
    assert settled[-1][3] is not None and "materialization failed" in settled[-1][3]
    # No replica is wedged MATERIALIZING, so the family can materialize again.
    assert all(
        r.state is not ReplicaState.MATERIALIZING for r in stores.directory.all()
    )
    assert svc._lifecycle.plan_capacity("m", "m").action == "materialize"
    # The claim settled without holding any credit.
    assert all(not c.holds_credit for c in stores.claims.all())
