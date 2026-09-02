"""The supervisor-side resident deputy binds a sidecar and drives the two phases.

Binding a sidecar and then poking the deputy's bootstrap and stream — the shape the
node-command handlers carry — delivers a claim-gated invocation over the data-direct
channel and refuses a fence the bound sidecar's incarnation does not match.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from server.network.state import ResolvedRoute, RouteCandidate, RouteHop, Transport
from server.resident.sidecar_server import EngineResponse
from server.resident.state import AdmissionHandoff, ReplicaEndpoint, RouteAuthorization
from server.supervisor.services.resident_deputy import ResidentDeputyService

_CHUNKS = ["ready ", "set ", "go"]


async def _fake_engine(
    endpoint: ReplicaEndpoint, request: str | None
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        for part in _CHUNKS:
            yield part

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


def _handoff(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    )
    base.update(overrides)
    return AdmissionHandoff(**base).model_dump(mode="json")


def _auth() -> dict[str, Any]:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        operation="inference",
        admission_epoch=0,
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    ).model_dump(mode="json")


def _route(host: str, port: int) -> dict[str, Any]:
    return ResolvedRoute(
        origin_id="rog-1",
        target_node_id="nde-1",
        listener_generation=1,
        route_epoch=1,
        candidates=(
            RouteCandidate(
                transport=Transport.WORKER_DIRECT,
                hops=(
                    RouteHop(
                        transport=Transport.WORKER_DIRECT, endpoint=f"{host}:{port}"
                    ),
                ),
            ),
        ),
    ).model_dump(mode="json")


async def _bound_service() -> tuple[ResidentDeputyService, dict[str, Any]]:
    svc = ResidentDeputyService(connect_budget_sec=3.0, engine_open=_fake_engine)
    bind = await svc.bind_sidecar(
        {
            "replica_id": "rpl-1",
            "incarnation": 1,
            "listener_generation": 1,
            "route": "127.0.0.1:0",
            "engine": {"base_url": "http://engine/v1", "model": "m", "api_key": None},
        }
    )
    assert bind["bound"]
    return svc, bind


def test_bind_then_bootstrap_and_stream() -> None:
    async def run() -> None:
        svc, bind = await _bound_service()
        route = _route(bind["host"], bind["port"])
        boot = await svc.bootstrap(
            {
                "session_id": "s1",
                "resolved_route": route,
                "handoff": _handoff(),
                "request": '{"prompt": "hi"}',
            }
        )
        assert boot["acked"] and boot["selected_transport"] == "worker_direct"
        stream = await svc.stream({"session_id": "s1", "auth": _auth()})
        assert stream["ok"] and stream["completion"] == "".join(_CHUNKS)
        await svc.stop()

    asyncio.run(run())


def test_bound_sidecar_refuses_a_mismatched_incarnation() -> None:
    async def run() -> None:
        svc, bind = await _bound_service()
        route = _route(bind["host"], bind["port"])
        boot = await svc.bootstrap(
            {
                "session_id": "s1",
                "resolved_route": route,
                "handoff": _handoff(incarnation=9),
                "request": None,
            }
        )
        assert not boot["acked"] and boot["rejection"] == "wrong_incarnation"
        await svc.stop()

    asyncio.run(run())


def test_unbind_drops_the_sidecar() -> None:
    async def run() -> None:
        svc, bind = await _bound_service()
        assert (await svc.unbind_sidecar("rpl-1"))["unbound"]
        route = _route(bind["host"], bind["port"])
        boot = await svc.bootstrap(
            {
                "session_id": "s1",
                "resolved_route": route,
                "handoff": _handoff(),
                "request": None,
            }
        )
        # With the sidecar gone the connect is refused and no candidate acks.
        assert not boot["acked"]
        await svc.stop()

    asyncio.run(run())
