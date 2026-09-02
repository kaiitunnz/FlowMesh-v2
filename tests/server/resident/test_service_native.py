"""Resident-capacity control carries an invocation over the native fabric path.

With the network plane wired, the service binds a sidecar, resolves a route, and drives
the two-phase delivery through the real supervisor deputy and sidecar (a fake engine
behind it): a completion releases the credit only on the fenced DS terminal, and a
post-acceptance stream loss holds the credit as uncertain and survives a restart.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from server.network.state import (
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    RouteObservationOutcome,
    RouteOrigin,
    Transport,
)
from server.orchestration import WorkItemStatus
from server.orchestration.tool_dispatch import ToolInvocationEnvelope
from server.resident import (
    AdmissionController,
    ClaimState,
    LifecycleScaleManager,
    ReplicaEndpoint,
    ResidentCapacityControl,
    ResidentPolicyLimits,
    ResidentStores,
)
from server.resident.native import NativeTransport
from server.resident.service import NativeDeliveryDeps
from server.resident.sidecar_server import EngineResponse
from server.supervisor.services.resident_deputy import ResidentDeputyService
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
_CHUNKS = ["once ", "upon ", "a time"]


async def _good_engine(
    endpoint: ReplicaEndpoint, request: str | None
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        for part in _CHUNKS:
            yield part

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


async def _loss_engine(
    endpoint: ReplicaEndpoint, request: str | None
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        raise ConnectionError("engine dropped mid-stream")
        yield ""  # pragma: no cover - marks this an async generator

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


class _FakeResolver:
    """Resolves a worker_direct route to whatever sidecar the listener advertises."""

    async def resolve(
        self, origin_node_id: str, listener: ReplicaListenerAdvertisement
    ) -> tuple[RouteOrigin, ResolvedRoute] | None:
        origin = RouteOrigin(
            origin_id="rog-1",
            endpoint_id="e-1",
            node_id=origin_node_id,
            reachability_class=ReachabilityClass.SAME_NODE,
            trust_domain="fm",
        )
        route = ResolvedRoute(
            origin_id="rog-1",
            target_node_id=listener.node_id,
            listener_generation=listener.listener_generation,
            route_epoch=1,
            candidates=(
                RouteCandidate(
                    transport=Transport.WORKER_DIRECT,
                    hops=(
                        RouteHop(
                            transport=Transport.WORKER_DIRECT,
                            endpoint=listener.routes[0],
                        ),
                    ),
                ),
            ),
        )
        return origin, route

    def record_observations(
        self,
        origin: RouteOrigin,
        listener: ReplicaListenerAdvertisement,
        observations: list[tuple[Transport, RouteObservationOutcome]],
    ) -> None:
        return None


def _binding() -> AgentModelGatewayBinding:
    return AgentModelGatewayBinding(
        mode=ModelBindingMode.RESIDENT, service_model_ref="m", provenance=_PROV
    )


def _env(invocation_id: str = "inv-1") -> ToolInvocationEnvelope:
    return ToolInvocationEnvelope(
        kind=BoundaryEventKind.INVOCATION,
        interface="model",
        invocation_id=invocation_id,
        task_id="tsk-1",
        activation_id="act-1",
        call_correlation="c1",
        idempotency_key="idm-1",
        request_payload='{"prompt": "hi"}',
    )


def _make(
    engine: Any,
    *,
    settle_cb: Any,
    redispatch_cb: Any,
    backoff: float = 0.0,
) -> tuple[ResidentCapacityControl, ResidentStores]:
    stores = ResidentStores()
    limits = ResidentPolicyLimits()

    async def materialize_fn(family: Any, replica: Any) -> str:
        return "tsk-serve-1"

    deputy_service = ResidentDeputyService(connect_budget_sec=3.0, engine_open=engine)

    async def exec_cmd(node_id: str, command: Any, payload: dict[str, Any]):
        name = command.value
        if name == "BIND_RESIDENT_SIDECAR":
            return await deputy_service.bind_sidecar(payload)
        if name == "UNBIND_RESIDENT_SIDECAR":
            return await deputy_service.unbind_sidecar(str(payload["replica_id"]))
        if name == "DELIVER_RESIDENT_BOOTSTRAP":
            return await deputy_service.bootstrap(payload)
        if name == "DELIVER_RESIDENT_STREAM":
            return await deputy_service.stream(payload)
        return await deputy_service.cancel(payload)

    native = NativeDeliveryDeps(
        network=_FakeResolver(),
        transport=NativeTransport(exec_cmd),
        origin_node_of_task=lambda task_id: "nde-1",
        node_of_replica=lambda replica: "nde-1",
    )
    admission = AdmissionController(stores)
    lifecycle = LifecycleScaleManager(
        stores, limits=limits, admission_slots=2, materialize_fn=materialize_fn
    )
    svc = ResidentCapacityControl(
        stores=stores,
        admission=admission,
        lifecycle=lifecycle,
        adapter=_UnusedAdapter(),
        limits=limits,
        binding_resolver=lambda task_id: ("wfl-1", _binding()),
        settle_cb=settle_cb,
        redispatch_cb=redispatch_cb,
        endpoint_probe=lambda serve_task_id: ReplicaEndpoint(
            base_url="http://engine/v1", model="m"
        ),
        native_delivery=native,
        poll_interval_sec=0.01,
        redrive_backoff_sec=backoff,
    )
    return svc, stores


def _build(
    engine: Any,
) -> tuple[ResidentCapacityControl, ResidentStores, list[Any], list[Any]]:
    settled: list[Any] = []
    redispatched: list[Any] = []

    def settle_cb(
        task_id: str, call_correlation: str, value: Any, *, error: Any = None
    ):
        settled.append((task_id, call_correlation, value, error))
        return True

    def redispatch_cb(task_id: str, call_correlation: str) -> bool:
        redispatched.append((task_id, call_correlation))
        return False

    svc, stores = _make(engine, settle_cb=settle_cb, redispatch_cb=redispatch_cb)
    return svc, stores, settled, redispatched


class _UnusedAdapter:
    async def issue(
        self, endpoint: ReplicaEndpoint, request_payload: str | None
    ) -> str:
        raise AssertionError("the native path must not use the in-server adapter")


def test_native_delivery_completes_and_releases_on_ds_terminal() -> None:
    async def run() -> None:
        svc, stores, settled, _ = _build(_good_engine)
        await svc._serve_invocation(_env())

        assert settled == [("tsk-1", "c1", "".join(_CHUNKS), None)]
        claim = stores.claims.by_invocation("inv-1")[0]
        assert claim.state is ClaimState.STREAMING
        assert claim.replica_id is not None
        assert stores.credit_ledger.held(claim.replica_id) == 1
        # The replica advertised a resident listener when its sidecar was bound.
        replica = stores.directory.get(claim.replica_id)
        assert replica is not None and replica.listener is not None
        assert replica.listener.protocols == ("resident",)

        svc.on_invocation_terminal("inv-1")
        assert claim.state is ClaimState.TERMINAL
        assert stores.credit_ledger.held(claim.replica_id) == 0

    asyncio.run(run())


def test_native_stream_loss_holds_uncertain_and_requests_a_redrive() -> None:
    async def run() -> None:
        svc, stores, settled, redispatched = _build(_loss_engine)
        await svc._serve_invocation(_env())

        claim = stores.claims.by_invocation("inv-1")[0]
        rid = claim.replica_id
        assert rid is not None
        # A lost stream is uncertain: the credit is held, not released, and the boundary
        # is re-driven rather than settled with an error.
        assert claim.state is ClaimState.UNCERTAIN
        assert stores.credit_ledger.held(rid) == 1
        assert settled == []
        assert redispatched == [("tsk-1", "c1")]

    asyncio.run(run())


def test_native_uncertain_claim_survives_restart() -> None:
    async def run() -> None:
        svc, stores, settled, _ = _build(_loss_engine)
        await svc._serve_invocation(_env())
        claim = stores.claims.by_invocation("inv-1")[0]
        rid = claim.replica_id
        assert rid is not None and claim.state is ClaimState.UNCERTAIN

        # A restart rehydrates the ledger: the in-flight claim stays uncertain and holds
        # its credit until the fenced DS terminal.
        snapshot = stores.to_snapshot()
        fresh, fresh_stores, _, _ = _build(_loss_engine)
        fresh.rehydrate(snapshot)
        resumed = fresh_stores.claims.by_invocation("inv-1")[0]
        assert resumed.state is ClaimState.UNCERTAIN
        assert resumed.replica_id is not None
        assert fresh_stores.credit_ledger.held(resumed.replica_id) == 1

        # The fenced DS terminal is the sole release.
        svc.on_invocation_terminal("inv-1", failed=True)
        assert claim.state is ClaimState.TERMINAL
        assert stores.credit_ledger.held(rid) == 0

    asyncio.run(run())


def test_failed_terminal_pokes_a_native_reap_for_a_live_session() -> None:
    async def run() -> None:
        cancels: list[dict[str, Any]] = []

        async def exec_cmd(node_id: str, command: Any, payload: dict[str, Any]):
            cancels.append({"node": node_id, "cmd": command.value, **payload})
            return {}

        svc, _ = _make(
            _good_engine,
            settle_cb=lambda *a, **k: True,
            redispatch_cb=lambda *a, **k: False,
        )
        assert svc._native is not None
        deps = replace(svc._native, transport=NativeTransport(exec_cmd))
        svc.set_native_delivery(deps)
        svc.bind_loop(asyncio.get_running_loop())
        svc._live_sessions["inv-9"] = ("nde-1", "inv-9:0")

        # A fenced failure/cancel terminal reaps the still-held native session.
        svc.on_invocation_terminal("inv-9", failed=True)
        await asyncio.sleep(0.05)
        assert cancels == [
            {"node": "nde-1", "cmd": "DELIVER_RESIDENT_CANCEL", "session_id": "inv-9:0"}
        ]
        assert "inv-9" not in svc._live_sessions

    asyncio.run(run())


def test_hold_and_redrive_is_a_noop_once_the_claim_settled() -> None:
    async def run() -> None:
        svc, stores, _, redispatched = _build(_good_engine)
        await svc._serve_invocation(_env())
        claim = stores.claims.by_invocation("inv-1")[0]
        svc.on_invocation_terminal("inv-1")
        assert claim.state is ClaimState.TERMINAL and redispatched == []

        # A late transient loss racing an already-settled claim neither re-holds the
        # released credit nor re-drives the settled boundary.
        await svc._hold_and_redrive(_env(), claim, "stream loss after terminal")
        assert redispatched == []
        assert "inv-1" not in svc._transient_failures
        assert claim.state is ClaimState.TERMINAL

    asyncio.run(run())


async def _noop() -> None:
    return None


class _FlakyEngine:
    """Drops the stream ``fail_times`` times, then serves a completion."""

    def __init__(self, fail_times: int) -> None:
        self.calls = 0
        self._fail = fail_times

    async def __call__(
        self, endpoint: ReplicaEndpoint, request: str | None
    ) -> EngineResponse:
        self.calls += 1
        if self.calls <= self._fail:

            async def dropped() -> AsyncIterator[str]:
                raise ConnectionError("engine dropped mid-stream")
                yield ""  # pragma: no cover - marks this an async generator

            return EngineResponse(chunks=dropped(), aclose=_noop)

        async def served() -> AsyncIterator[str]:
            for part in _CHUNKS:
                yield part

        return EngineResponse(chunks=served(), aclose=_noop)


def test_native_stream_loss_redrives_to_completion_through_the_runtime() -> None:
    # The production seam: model_settler routes through the service, losses hold the
    # credit and re-drive through the real runtime, and a completion settles the
    # boundary and releases the credit through the fenced DS terminal — no stub.
    from server.task.models import TaskStatus
    from tests.server.task import test_v2_orchestration as v2o
    from tests.server.task.test_agent_episode_runtime import _AGENT_WF, _step
    from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

    async def run() -> None:
        runtime = v2o._runtime(v2o.FakeRegistry())
        engine = _FlakyEngine(fail_times=2)
        seen: dict[str, str] = {}
        svc, stores = _make(
            engine,
            settle_cb=runtime.settle_episode_invocation,
            redispatch_cb=runtime.redispatch_episode_invocation,
            backoff=0.001,
        )
        svc.bind_loop(asyncio.get_running_loop())
        runtime.set_resident_terminal_hook(svc.on_invocation_terminal)

        def settler(env: ToolInvocationEnvelope) -> None:
            seen.setdefault("inv", env.invocation_id)
            svc.settle(env)

        runtime.set_model_settler(settler)
        workflow_id, ids = await v2o._register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(
            [
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.INVOCATION,
                    call="m0",
                    interface="model",
                    payload="draft",
                ),
                ScriptedStep(op="complete", value_from="m0"),
            ],
            "v1",
        )
        eng = runtime.orchestration_engine(workflow_id)
        assert eng is not None

        _step(runtime, adapter, writer)  # model boundary → service → held re-drive loop
        for _ in range(1000):
            await asyncio.sleep(0.005)
            claims = stores.claims.by_invocation(seen.get("inv", ""))
            if claims and claims[0].state is ClaimState.TERMINAL:
                break
        assert engine.calls == 3  # two held re-drives, then a completion
        claims = stores.claims.by_invocation(seen["inv"])
        assert len(claims) == 1  # the same claim resumed; no successor was raised
        assert claims[0].state is ClaimState.TERMINAL
        assert claims[0].replica_id is not None
        assert stores.credit_ledger.held(claims[0].replica_id) == 0
        assert runtime._tasks[writer].status is TaskStatus.PENDING  # re-readied

        _step(runtime, adapter, writer)  # the agent injects the model value, completes
        wi = eng.work_item(writer)
        assert wi is not None and wi.status is WorkItemStatus.SETTLED

    asyncio.run(run())
