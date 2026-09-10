"""Resident-capacity control originates a boundary and settles it from worker reports.

With a stubbed worker delivery, a resident invocation with no pre-enabled capacity
materializes scale-from-zero, reserves a claim, binds the replica sidecar, resolves the
origin fence, and relays the claim-bound handoff to the origin worker. The engine
enqueue ack accepts the claim and issues the route authorization; the fenced terminal
report settles the boundary by reference and only the DS terminal releases the credit. A
lost ack or an uncertain outcome holds the credit and re-drives; a disallowed model
denies with no allocation; and a restart reconciles an in-flight claim to uncertain.
"""

import asyncio
from typing import Any

from server.network.state import (
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteCandidate,
    RouteOrigin,
    Transport,
)
from server.orchestration.tool_dispatch import ToolInvocationEnvelope
from server.resident import (
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
from server.resident.service import ResidentWorkerDelivery
from server.resident.state import ReplicaIncarnation
from server.task.v2.representations.operators import (
    ServiceDependency,
    ServiceInterface,
)
from shared.harness import BoundaryEventKind
from shared.outcome import OutcomeManifest
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamStatus,
)


def _dependency(model_ref: str = "m") -> ServiceDependency:
    return ServiceDependency(service_ref=model_ref)


def _env(invocation_id: str = "inv-1") -> ToolInvocationEnvelope:
    return ToolInvocationEnvelope(
        kind=BoundaryEventKind.INVOCATION,
        interface="model",
        invocation_id=invocation_id,
        task_id="tsk-1",
        activation_id="act-1",
        call_correlation="c1",
        idempotency_key="idm-1",
        request_digest="sha-req",
    )


class _FakeNetwork:
    async def resolve(
        self, origin_node_id: str, listener: ReplicaListenerAdvertisement
    ) -> tuple[RouteOrigin, ResolvedRoute]:
        origin = RouteOrigin(
            origin_id="rog-1",
            endpoint_id="ep-1",
            reachability_class=ReachabilityClass.ROUTABLE,
            trust_domain="td",
        )
        route = ResolvedRoute(
            origin_id="rog-1",
            target_node_id=listener.node_id,
            listener_generation=listener.listener_generation,
            route_epoch=1,
            candidates=(RouteCandidate(transport=Transport.CONTROL_RELAY, hops=()),),
        )
        return origin, route

    def record_observations(self, origin, listener, observations) -> None:
        self.observations = list(observations)

    async def endpoint_for(self, node_id: str):
        return None


class _FakeSessions:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, str]] = {}
        self.deleted: list[str] = []

    async def update(self, session_id: str, **fields: str | int) -> None:
        self.records.setdefault(session_id, {}).update(
            {k: str(v) for k, v in fields.items()}
        )

    async def delete(self, session_id: str) -> None:
        self.deleted.append(session_id)


class _Delivery:
    """A recording worker delivery: captures relays and the session records."""

    def __init__(self, deliver: bool = True) -> None:
        self.relays: list[tuple[str, str, dict[str, Any]]] = []
        self.deliver = deliver
        self.sessions = _FakeSessions()

    def build(self) -> ResidentWorkerDelivery:
        return ResidentWorkerDelivery(
            relay=self._relay,
            origin_worker_of_task=lambda task_id: "wkr-origin",
            serve_worker_of=lambda replica: (
                "wkr-replica" if replica.serve_task_id is not None else None
            ),
            node_of_worker=lambda worker_id: "node-1" if worker_id else None,
            network=_FakeNetwork(),
            sessions=self.sessions,
        )

    def _relay(self, worker_id: str, frame_kind: str, payload: dict[str, Any]) -> bool:
        self.relays.append((worker_id, frame_kind, payload))
        return self.deliver

    def kinds(self) -> list[str]:
        return [frame_kind for _worker, frame_kind, _payload in self.relays]

    def frame(self, frame_kind: str) -> dict[str, Any]:
        return next(p for _w, k, p in self.relays if k == frame_kind)


def _build(
    *,
    limits: ResidentPolicyLimits | None = None,
    materialize_fn: Any = None,
    deliver: bool = True,
    dependency: ServiceDependency | None = None,
) -> tuple[ResidentCapacityControl, ResidentStores, list[Any], _Delivery]:
    stores = ResidentStores()
    limits = limits or ResidentPolicyLimits()
    settled: list[Any] = []
    redispatched: list[tuple[str, str]] = []

    def settle_cb(
        task_id: str,
        call_correlation: str,
        value: Any = None,
        *,
        error: str | None = None,
        ref: OutcomeManifest | None = None,
    ) -> bool:
        settled.append((task_id, call_correlation, value, error, ref))
        return True

    def redispatch_cb(task_id: str, call_correlation: str) -> bool:
        redispatched.append((task_id, call_correlation))
        return True

    if materialize_fn is None:

        async def materialize_fn(family: str, replica: ReplicaIncarnation) -> str:
            return "tsk-serve-1"

    admission = AdmissionController(stores)
    lifecycle = LifecycleScaleManager(
        stores, limits=limits, admission_slots=2, materialize_fn=materialize_fn
    )
    delivery = _Delivery(deliver=deliver)
    svc = ResidentCapacityControl(
        stores=stores,
        admission=admission,
        lifecycle=lifecycle,
        limits=limits,
        dependency_resolver=lambda task_id: ("wfl-1", dependency or _dependency()),
        settle_cb=settle_cb,
        redispatch_cb=redispatch_cb,
        endpoint_probe=lambda serve_task_id: ReplicaEndpoint(
            base_url="http://replica", model="m"
        ),
        delivery=delivery.build(),
        poll_interval_sec=0.01,
        redrive_backoff_sec=0.0,
    )
    svc._redispatched = redispatched  # type: ignore[attr-defined]
    return svc, stores, settled, delivery


def _ack(
    svc: ResidentCapacityControl,
    outcome: ResidentBootstrapOutcome,
    *,
    invocation_id: str = "inv-1",
    rejection: str | None = None,
) -> ResidentBootstrapAck:
    session_id = svc._attempts[invocation_id].session_id
    return ResidentBootstrapAck(
        task_id="tsk-1",
        call_correlation="c1",
        invocation_id=invocation_id,
        session_id=session_id,
        outcome=outcome,
        rejection=rejection,
    )


def _outcome(
    svc: ResidentCapacityControl,
    status: ResidentStreamStatus,
    *,
    invocation_id: str = "inv-1",
    manifest: OutcomeManifest | None = None,
    error: str | None = None,
) -> ResidentOpOutcome:
    session_id = svc._attempts[invocation_id].session_id
    return ResidentOpOutcome(
        task_id="tsk-1",
        call_correlation="c1",
        invocation_id=invocation_id,
        session_id=session_id,
        status=status,
        manifest=manifest,
        error=error,
    )


def test_originate_binds_sidecar_resolves_fence_and_relays_handoff():
    svc, stores, _settled, delivery = _build()
    asyncio.run(svc._originate(_env()))

    assert delivery.kinds() == ["resident_sidecar_bind", "resident_handoff"]
    bind = delivery.frame("resident_sidecar_bind")
    assert bind["engine"]["base_url"] == "http://replica"
    assert bind["engine"]["interface"] == "chat"
    handoff = delivery.frame("resident_handoff")["handoff"]
    assert handoff["origin_id"] == "rog-1"

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.RESERVED
    assert stores.credit_ledger.held(claim.replica_id) == 1
    assert stores.directory.get(claim.replica_id).state is ReplicaState.WARM

    family = stores.families.get(stores.claims.by_invocation("inv-1")[0].family)
    assert family is not None and family.interface == "chat"

    session_id = svc._attempts["inv-1"].session_id
    record = delivery.sessions.records[session_id]
    assert record["origin_worker"] == "wkr-origin"
    assert record["target_worker"] == "wkr-replica"
    assert record["invocation_id"] == "inv-1"


def test_embedding_dependency_relays_the_embedding_interface_to_the_sidecar():
    dependency = ServiceDependency(
        service_ref="m", interface=ServiceInterface.EMBEDDING
    )
    svc, stores, _settled, delivery = _build(dependency=dependency)
    asyncio.run(svc._originate(_env()))

    bind = delivery.frame("resident_sidecar_bind")
    assert bind["engine"]["interface"] == "embedding"
    family = stores.families.get(stores.claims.by_invocation("inv-1")[0].family)
    assert family is not None and family.interface == "embedding"


def test_adapter_dependency_relays_the_adapter_on_the_handoff():
    dependency = ServiceDependency(
        service_ref="m", adapter="my-lora", adapter_source="hf/my-lora"
    )
    svc, stores, _settled, delivery = _build(dependency=dependency)
    asyncio.run(svc._originate(_env()))

    handoff = delivery.frame("resident_handoff")["handoff"]
    assert handoff["adapter_name"] == "my-lora"
    assert handoff["adapter_source"] == "hf/my-lora"
    request = stores.invocations.get("inv-1")
    assert request is not None and request.profile.adapter_ref == "my-lora"


def test_ack_accepts_and_authorizes_then_terminal_releases_credit():
    svc, stores, settled, delivery = _build()
    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.STREAMING
    auth = delivery.frame("resident_authorization")["auth"]
    assert auth["origin_id"] == "rog-1" and auth["claim_id"] == claim.claim_id

    manifest = OutcomeManifest(
        content_digest="sha", size_bytes=2, media_type="text/plain"
    )
    asyncio.run(
        svc._on_outcome(_outcome(svc, ResidentStreamStatus.SUCCESS, manifest=manifest))
    )
    assert settled[-1] == ("tsk-1", "c1", None, None, manifest)

    # The fenced DS terminal, consumed by invocation_id, releases the credit and reaps.
    svc.on_invocation_terminal("inv-1")
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0
    assert "resident_reap" in delivery.kinds()


def _adapter_env(invocation_id: str) -> ToolInvocationEnvelope:
    return ToolInvocationEnvelope(
        kind=BoundaryEventKind.INVOCATION,
        interface="model",
        invocation_id=invocation_id,
        task_id=f"tsk-{invocation_id}",
        activation_id="act-1",
        call_correlation=f"c-{invocation_id}",
        idempotency_key=f"idm-{invocation_id}",
        request_digest="sha-req",
    )


def _unload_frames(delivery: _Delivery) -> list[dict[str, Any]]:
    return [p for _w, k, p in delivery.relays if k == "resident_adapter_unload"]


def test_last_holder_release_unloads_the_adapter_slot():
    dependency = ServiceDependency(
        service_ref="m", adapter="my-lora", adapter_source="hf/my-lora"
    )
    svc, stores, _settled, delivery = _build(dependency=dependency)
    asyncio.run(svc._originate(_adapter_env("inv-1")))
    replica_id = stores.claims.by_invocation("inv-1")[0].replica_id

    svc.on_invocation_terminal("inv-1")

    frames = _unload_frames(delivery)
    assert len(frames) == 1
    assert frames[0] == {"replica_id": replica_id, "adapter_name": "my-lora"}
    unload_target = next(
        w for w, k, _p in delivery.relays if k == "resident_adapter_unload"
    )
    assert unload_target == "wkr-replica"


def test_a_concurrent_same_adapter_claim_is_not_unloaded():
    dependency = ServiceDependency(
        service_ref="m", adapter="my-lora", adapter_source="hf/my-lora"
    )
    svc, stores, _settled, delivery = _build(dependency=dependency)
    asyncio.run(svc._originate(_adapter_env("inv-1")))
    asyncio.run(svc._originate(_adapter_env("inv-2")))
    # Both claims share one warm replica and hold the same adapter's single slot.
    r1 = stores.claims.by_invocation("inv-1")[0].replica_id
    r2 = stores.claims.by_invocation("inv-2")[0].replica_id
    assert r1 == r2 and stores.credit_ledger.held(r1) == 2

    # The first holder's release must NOT unload the adapter out from under its peer.
    svc.on_invocation_terminal("inv-1")
    assert _unload_frames(delivery) == []

    # Only the last holder's release frees the slot.
    svc.on_invocation_terminal("inv-2")
    frames = _unload_frames(delivery)
    assert len(frames) == 1 and frames[0]["adapter_name"] == "my-lora"


def test_a_base_claim_release_relays_no_unload():
    svc, _stores, _settled, delivery = _build()  # base dependency, no adapter
    asyncio.run(svc._originate(_env()))
    svc.on_invocation_terminal("inv-1")
    assert _unload_frames(delivery) == []


def test_rejected_ack_releases_the_reservation_and_settles_an_error():
    svc, stores, settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    asyncio.run(
        svc._on_ack(
            _ack(svc, ResidentBootstrapOutcome.REJECTED, rejection="stale_listener")
        )
    )

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0
    assert settled[-1][3] is not None and "bootstrap refused" in settled[-1][3]


def test_uncertain_ack_holds_credit_and_redrives():
    svc, stores, _settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.UNCERTAIN)))

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.UNCERTAIN
    assert stores.credit_ledger.held(claim.replica_id) == 1  # held, not released
    assert svc._redispatched == [("tsk-1", "c1")]  # type: ignore[attr-defined]


def test_uncertain_outcome_holds_credit_and_redrives():
    svc, stores, _settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    asyncio.run(
        svc._on_outcome(
            _outcome(svc, ResidentStreamStatus.UNCERTAIN, error="stream lost")
        )
    )

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.UNCERTAIN
    assert stores.credit_ledger.held(claim.replica_id) == 1
    assert svc._redispatched == [("tsk-1", "c1")]  # type: ignore[attr-defined]


def test_definite_failure_outcome_settles_an_error():
    svc, _stores, settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    asyncio.run(
        svc._on_outcome(
            _outcome(svc, ResidentStreamStatus.DEFINITE_FAILURE, error="engine refused")
        )
    )
    assert settled[-1][3] is not None and "engine refused" in settled[-1][3]


def test_stale_report_is_ignored_leaving_the_claim_credit_bearing():
    svc, stores, settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    stale = ResidentBootstrapAck(
        task_id="tsk-1",
        call_correlation="c1",
        invocation_id="inv-1",
        session_id="rly-stale",
        outcome=ResidentBootstrapOutcome.ACKED,
    )
    asyncio.run(svc._on_ack(stale))

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.RESERVED  # no transition on a stale session
    assert stores.credit_ledger.held(claim.replica_id) == 1
    assert settled == []


def test_disallowed_model_denies_without_allocation_or_credit():
    limits = ResidentPolicyLimits(allowed_models=frozenset({"approved-only"}))
    svc, stores, settled, _delivery = _build(limits=limits)
    asyncio.run(svc._originate(_env()))

    assert len(settled) == 1 and settled[0][3] is not None
    assert "model_not_allowed" in settled[0][3]
    assert stores.directory.all() == []
    assert stores.claims.all() == []


def test_failed_materialize_recovers_family_and_settles():
    async def boom(family: str, replica: ReplicaIncarnation) -> str:
        raise RuntimeError("cold start failed")

    svc, stores, settled, _delivery = _build(materialize_fn=boom)
    asyncio.run(svc._originate(_env()))

    assert settled[-1][3] is not None and "materialization failed" in settled[-1][3]
    assert all(
        r.state is not ReplicaState.MATERIALIZING for r in stores.directory.all()
    )
    assert all(not c.holds_credit for c in stores.claims.all())


def test_redrive_resumes_under_one_credit_without_double_admit():
    svc, stores, _settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.UNCERTAIN)))
    first = stores.claims.by_invocation("inv-1")[0]
    assert first.state is ClaimState.UNCERTAIN
    replicas = len(stores.directory.all())

    # A re-drive of the same invocation resumes the parked claim on its live replica:
    # no new claim, no new materialize, and the credit is not released.
    asyncio.run(svc._originate(_env()))
    claims = stores.claims.by_invocation("inv-1")
    assert len(claims) == 1 and claims[0] is first
    assert len(stores.directory.all()) == replicas
    assert stores.credit_ledger.held(first.replica_id) == 1

    svc.on_invocation_terminal("inv-1")
    assert first.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(first.replica_id) == 0


def test_originate_settles_an_error_when_an_internal_path_raises():
    svc, _stores, settled, _delivery = _build()

    def boom(task_id: str) -> Any:
        raise RuntimeError("resolver exploded")

    svc._resolve_dependency = boom  # type: ignore[method-assign]
    asyncio.run(svc._originate(_env()))
    assert len(settled) == 1
    assert settled[0][3] is not None and "resident origination error" in settled[0][3]


def test_rehydrate_reconciles_in_flight_claim():
    svc, stores, _settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    snapshot = stores.to_snapshot()

    fresh_svc, fresh_stores, _s, _d = _build()
    fresh_svc.rehydrate(snapshot)
    claim = fresh_stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.UNCERTAIN
    assert fresh_stores.credit_ledger.held(claim.replica_id) == 1


def test_rehydrate_reports_a_warm_replica_so_it_is_admittable_again():
    svc, stores, _settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    snapshot = stores.to_snapshot()

    fresh_svc, fresh_stores, _s, _d = _build()
    fresh_svc.rehydrate(snapshot)
    fam = fresh_stores.directory.all()[0].family
    assert fresh_stores.pools.feasible_candidates(
        fam, AdmissionProfile(engine_batch_key=fam)
    )


def test_rehydrate_preempts_a_warm_replica_whose_serve_task_is_gone():
    svc, stores, _settled, _delivery = _build()
    asyncio.run(svc._originate(_env()))
    snapshot = stores.to_snapshot()

    fresh_svc, fresh_stores, _s, _d = _build()
    fresh_svc._probe_endpoint = lambda serve_task_id: None  # serve task is gone
    fresh_svc.rehydrate(snapshot)
    states = {r.state for r in fresh_stores.directory.all()}
    assert ReplicaState.WARM not in states
    assert ReplicaState.PREEMPTED in states
