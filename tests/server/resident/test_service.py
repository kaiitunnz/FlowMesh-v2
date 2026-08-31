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
    AdmissionController,
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

    async def issue(self, handoff, request_payload):
        self.calls.append(handoff)
        return self.completion


def _build(*, limits=None, adapter=None):
    stores = ResidentStores()
    limits = limits or ResidentPolicyLimits()
    settled = []

    def settle_cb(task_id, call_correlation, value, *, error=None):
        settled.append((task_id, call_correlation, value, error))
        return True

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
