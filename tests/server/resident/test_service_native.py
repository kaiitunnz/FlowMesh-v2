"""Resident-capacity control carries an invocation over the native fabric path.

With the network plane wired, the service binds a sidecar, resolves a route, and drives
the two-phase delivery through the real supervisor deputy and sidecar (a fake engine
behind it): a completion releases the credit only on the fenced DS terminal, and a
post-acceptance stream loss holds the credit as uncertain and survives a restart.
"""

import asyncio
from collections.abc import AsyncIterator
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


def _build(engine: Any) -> tuple[ResidentCapacityControl, ResidentStores, list[Any]]:
    stores = ResidentStores()
    limits = ResidentPolicyLimits()
    settled: list[Any] = []

    def settle_cb(
        task_id: str, call_correlation: str, value: Any, *, error: Any = None
    ):
        settled.append((task_id, call_correlation, value, error))
        return True

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
        endpoint_probe=lambda serve_task_id: ReplicaEndpoint(
            base_url="http://engine/v1", model="m"
        ),
        native_delivery=native,
        poll_interval_sec=0.01,
    )
    return svc, stores, settled


class _UnusedAdapter:
    async def issue(
        self, endpoint: ReplicaEndpoint, request_payload: str | None
    ) -> str:
        raise AssertionError("the native path must not use the in-server adapter")


def test_native_delivery_completes_and_releases_on_ds_terminal() -> None:
    async def run() -> None:
        svc, stores, settled = _build(_good_engine)
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


def test_native_stream_loss_is_uncertain_and_survives_restart() -> None:
    async def run() -> None:
        svc, stores, settled = _build(_loss_engine)
        await svc._serve_invocation(_env())

        claim = stores.claims.by_invocation("inv-1")[0]
        rid = claim.replica_id
        assert rid is not None
        # held, not released on a lost stream
        assert claim.state is ClaimState.UNCERTAIN
        assert stores.credit_ledger.held(rid) == 1
        assert settled[-1][3] is not None  # the boundary settled with an error

        # A restart rehydrates the ledger: the in-flight claim stays uncertain and holds
        # its credit until the fenced DS terminal.
        snapshot = stores.to_snapshot()
        fresh, fresh_stores, _ = _build(_loss_engine)
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
