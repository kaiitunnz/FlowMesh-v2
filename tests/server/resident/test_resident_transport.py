"""The resident deputy and sidecar carry a claim-gated invocation over each transport.

Over real loopback sockets — with the substrate relays as the intermediate hops and a
fake engine behind the sidecar — a bootstrap is admitted and its response streams back;
a fence rejection is refused before the engine body is forwarded and surfaces as a
stopped delivery at the deputy; cancellation is honored only under a valid
authorization; and a pre-send connect failure falls to the next candidate.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Any

from server.network.listeners import NetworkControlRelay, NetworkPlaneListeners
from server.network.state import (
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    Transport,
)
from server.resident.deputy import ResidentInvocationDeputy
from server.resident.sidecar import SidecarClaimGate
from server.resident.sidecar_server import (
    EngineResponse,
    ResidentSidecarListener,
    ResidentSidecarServer,
)
from server.resident.state import AdmissionHandoff, ReplicaEndpoint, RouteAuthorization

_CHUNKS = ["one ", "two ", "three"]


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


async def _fake_engine(
    endpoint: ReplicaEndpoint, request: str | None
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        for part in _CHUNKS:
            yield part

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


def _gate() -> SidecarClaimGate:
    return SidecarClaimGate(replica_id="rpl-1", incarnation=1, listener_generation=1)


def _handoff(**overrides: object) -> AdmissionHandoff:
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
    return AdmissionHandoff(**base)


def _auth(**overrides: object) -> RouteAuthorization:
    base: dict[str, Any] = dict(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        operation="inference",
        admission_epoch=0,
        route_auth_epoch=1,
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    )
    base.update(overrides)
    return RouteAuthorization(**base)


class _Fixture:
    """A resident sidecar plus the substrate relay hops, all on loopback."""

    def __init__(self) -> None:
        self.sidecar_port = _free_port()
        self.relay_port = _free_port()
        self.control_port = _free_port()
        self.loads: list[str] = []
        self.sidecar = ResidentSidecarListener(
            ResidentSidecarServer(
                gate=_gate(),
                endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
                engine_open=_fake_engine,
                on_load=lambda ev: self.loads.append(ev.operation),
            ),
            route=f"127.0.0.1:{self.sidecar_port}",
        )
        self.relays = NetworkPlaneListeners(
            sidecar_url=f"127.0.0.1:{_free_port()}",
            endpoint_url=f"127.0.0.1:{self.relay_port}",
            buffer_bytes=65536,
        )
        self.control = NetworkControlRelay(
            control_relay_url=f"127.0.0.1:{self.control_port}", buffer_bytes=65536
        )

    async def __aenter__(self) -> "_Fixture":
        await self.sidecar.start()
        await self.relays.start()
        await self.control.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.relays.stop()
        await self.control.stop()
        await self.sidecar.stop()

    def _sidecar_hop(self, transport: Transport) -> RouteHop:
        return RouteHop(transport=transport, endpoint=f"127.0.0.1:{self.sidecar_port}")

    def direct(self) -> ResolvedRoute:
        return _route(
            RouteCandidate(
                transport=Transport.WORKER_DIRECT,
                hops=(self._sidecar_hop(Transport.WORKER_DIRECT),),
            )
        )

    def node_relay(self) -> ResolvedRoute:
        t = Transport.NODE_RELAY
        return _route(
            RouteCandidate(
                transport=t,
                hops=(
                    RouteHop(transport=t, endpoint=f"127.0.0.1:{self.relay_port}"),
                    self._sidecar_hop(t),
                ),
            )
        )

    def control_relay(self) -> ResolvedRoute:
        t = Transport.CONTROL_RELAY
        return _route(
            RouteCandidate(
                transport=t,
                hops=(
                    RouteHop(transport=t, endpoint=f"127.0.0.1:{self.control_port}"),
                    RouteHop(transport=t, endpoint=f"127.0.0.1:{self.relay_port}"),
                    self._sidecar_hop(t),
                ),
            )
        )


def _route(*candidates: RouteCandidate) -> ResolvedRoute:
    return ResolvedRoute(
        origin_id="rog-1",
        target_node_id="nde-1",
        listener_generation=1,
        route_epoch=1,
        candidates=candidates,
    )


async def _deliver(deputy: ResidentInvocationDeputy, route: ResolvedRoute):
    boot = await deputy.bootstrap("s1", route, _handoff(), '{"prompt": "hi"}')
    if not boot.acked:
        return boot, None
    return boot, await deputy.stream("s1", _auth())


def test_delivers_and_streams_over_each_transport() -> None:
    async def run(select) -> None:
        async with _Fixture() as fx:
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            boot, stream = await _deliver(deputy, select(fx))
            assert boot.acked and boot.selected_transport is not None
            assert stream is not None and stream.ok
            assert stream.completion == "".join(_CHUNKS)
            assert fx.loads == ["request", "stream"]

    for select in (
        _Fixture.direct,
        _Fixture.node_relay,
        _Fixture.control_relay,
    ):
        asyncio.run(run(select))


def test_bootstrap_fence_rejection_stops_before_the_engine() -> None:
    async def run() -> None:
        async with _Fixture() as fx:
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            boot = await deputy.bootstrap(
                "s1", fx.direct(), _handoff(incarnation=9), '{"prompt": "hi"}'
            )
            assert not boot.acked and not boot.uncertain
            assert boot.rejection == "wrong_incarnation"
            # The engine was never opened: no load evidence was emitted.
            assert fx.loads == []

    asyncio.run(run())


def test_stream_rejection_is_refused_before_the_body() -> None:
    async def run(auth_override, expected) -> None:
        async with _Fixture() as fx:
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            boot = await deputy.bootstrap(
                "s1", fx.direct(), _handoff(), '{"prompt": "hi"}'
            )
            assert boot.acked
            result = await deputy.stream("s1", _auth(**auth_override))
            assert not result.ok and result.completion is None
            assert result.rejection == expected
            # The bootstrap opened the engine, but no stream body was forwarded.
            assert fx.loads == ["request"]

    for override, expected in [
        ({"incarnation": 9}, "wrong_incarnation"),
        ({"listener_generation": 9}, "stale_listener"),
        ({"expires_at": "2000-01-01T00:00:00Z"}, "expired"),
        ({"tenant": "other"}, "wrong_subject"),
    ]:
        asyncio.run(run(override, expected))


def test_cancel_is_honored_only_under_a_valid_authorization() -> None:
    async def run() -> None:
        async with _Fixture() as fx:
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            await deputy.bootstrap("s1", fx.direct(), _handoff(), None)
            good = await deputy.cancel("s1", _auth())
            assert good.ok and good.cancelled

            await deputy.bootstrap("s2", fx.direct(), _handoff(), None)
            bad = await deputy.cancel("s2", _auth(incarnation=9))
            assert not bad.ok and bad.rejection == "wrong_incarnation"

    asyncio.run(run())


def test_pre_send_connect_failure_falls_to_the_next_candidate() -> None:
    async def run() -> None:
        async with _Fixture() as fx:
            dead = _free_port()
            route = _route(
                RouteCandidate(
                    transport=Transport.WORKER_DIRECT,
                    hops=(
                        RouteHop(
                            transport=Transport.WORKER_DIRECT,
                            endpoint=f"127.0.0.1:{dead}",
                        ),
                    ),
                ),
                fx.direct().candidates[0],
            )
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            boot = await deputy.bootstrap("s1", route, _handoff(), None)
            assert boot.acked and boot.selected_transport is Transport.WORKER_DIRECT
            assert boot.observations[0][1].value == "connect_failure"
            await deputy.reap("s1")

    asyncio.run(run())
