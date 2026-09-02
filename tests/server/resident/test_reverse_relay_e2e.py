"""All-NAT resident delivery over the reverse-rendezvous relay, end to end in process.

Neither node accepts an inbound connection: the origin and the target attach outward to
one shared (faked) Redis, the root bridge moves frames between their per-node streams,
and the target reaches its engine only over a loopback sidecar. A real two-phase gated
invocation — bootstrap, ack, authorized stream — completes over control_relay, the case
the removed forward-dial control relay could not serve. Cancellation reaps the sidecar.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from server.network.rendezvous import RootCursorStore, RootRendezvousBridge
from server.network.reverse_relay import RelaySessionStore, RelayStreamStore
from server.network.state import (
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    Transport,
)
from server.resident.relay_delivery import ResidentRelayEndpoint
from server.resident.sidecar import SidecarClaimGate
from server.resident.sidecar_server import (
    EngineResponse,
    ResidentSidecarListener,
    ResidentSidecarServer,
)
from server.resident.state import AdmissionHandoff, ReplicaEndpoint, RouteAuthorization
from server.supervisor.services.resident_relay_attachment import ResidentRelayAttachment
from tests.server.network._relay_fakes import FakeBinaryRedis

_CHUNKS = ["once ", "upon ", "a time"]


async def _fake_engine(
    endpoint: ReplicaEndpoint, request: str | None
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        for part in _CHUNKS:
            yield part

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


class _SlowEngine:
    """Hangs in the engine request until cancelled, recording that it was aborted."""

    def __init__(self) -> None:
        self.aborted = False

    async def __call__(
        self, endpoint: ReplicaEndpoint, request: str | None
    ) -> EngineResponse:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.aborted = True
            raise
        raise AssertionError("the slow engine should have been cancelled")


def _handoff() -> AdmissionHandoff:
    return AdmissionHandoff(
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


def _auth() -> RouteAuthorization:
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
    )


def _route(sidecar_route: str) -> ResolvedRoute:
    t = Transport.CONTROL_RELAY
    return ResolvedRoute(
        origin_id="rog-1",
        target_node_id="nde-t",
        listener_generation=1,
        route_epoch=1,
        candidates=(
            RouteCandidate(
                transport=t,
                hops=(
                    RouteHop(
                        transport=t, endpoint="", node_id="nde-o", attachment_id="a-o"
                    ),
                    RouteHop(
                        transport=t,
                        endpoint=sidecar_route,
                        node_id="nde-t",
                        attachment_id="a-t",
                    ),
                ),
            ),
        ),
    )


class _Harness:
    """Shared redis, root bridge, two attachments/endpoints, and a loopback sidecar."""

    def __init__(self, engine: Any = _fake_engine) -> None:
        self.redis = FakeBinaryRedis()
        self.loads: list[str] = []
        self.sidecar = ResidentSidecarListener(
            ResidentSidecarServer(
                gate=SidecarClaimGate(
                    replica_id="rpl-1", incarnation=1, listener_generation=1
                ),
                endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
                engine_open=engine,
                on_load=lambda ev: self.loads.append(ev.operation),
            ),
            route="127.0.0.1:0",
        )
        self.origin = ResidentRelayEndpoint(self.redis, "nde-o", recv_budget_sec=5.0)
        self.target = ResidentRelayEndpoint(self.redis, "nde-t", recv_budget_sec=5.0)
        self.bridge = RootRendezvousBridge(
            RelayStreamStore(self.redis),
            RelaySessionStore(self.redis),
            RootCursorStore(self.redis),
        )
        self.origin_attach = ResidentRelayAttachment(
            self.redis, "nde-o", self.origin, owner="o"
        )
        self.target_attach = ResidentRelayAttachment(
            self.redis, "nde-t", self.target, owner="t"
        )

    async def start(self) -> str:
        host, port = await self.sidecar.start()
        return f"{host}:{port}"

    async def pump_step(self) -> None:
        await self.bridge.pump_node("nde-o")
        await self.target_attach.pump_once()
        await self.bridge.pump_node("nde-t")
        await self.origin_attach.pump_once()
        await asyncio.sleep(0.002)


async def _drive(harness: _Harness, done: asyncio.Event) -> None:
    while not done.is_set():
        await harness.pump_step()


def test_all_nat_invocation_completes_over_control_relay() -> None:
    async def run() -> None:
        harness = _Harness()
        sidecar_route = await harness.start()
        route = _route(sidecar_route)
        done = asyncio.Event()

        async def exchange() -> Any:
            boot = await harness.origin.bootstrap(
                "s1", route=route, handoff=_handoff(), request_payload='{"p":"hi"}'
            )
            assert boot.acked and boot.selected_transport is Transport.CONTROL_RELAY
            result = await harness.origin.stream("s1", _auth())
            done.set()
            return result

        driver = asyncio.ensure_future(_drive(harness, done))
        try:
            result = await asyncio.wait_for(exchange(), timeout=10.0)
        finally:
            done.set()
            await driver
        assert result.ok and result.completion == "".join(_CHUNKS)
        # The sidecar served a claim-gated request and stream, all over the relay.
        assert harness.loads == ["request", "stream"]
        await harness.sidecar.stop()

    asyncio.run(run())


def test_cancel_over_the_relay_reaps_the_co_located_sidecar() -> None:
    async def run() -> None:
        slow = _SlowEngine()
        harness = _Harness(engine=slow)
        route = _route(await harness.start())
        done = asyncio.Event()
        driver = asyncio.ensure_future(_drive(harness, done))
        try:
            boot = await asyncio.wait_for(
                harness.origin.bootstrap(
                    "s1", route=route, handoff=_handoff(), request_payload='{"p":"hi"}'
                ),
                timeout=10.0,
            )
            assert boot.acked
            stream = asyncio.ensure_future(harness.origin.stream("s1", _auth()))
            await asyncio.sleep(0.1)  # the authorized stream reaches the hung engine
            await harness.origin.cancel("s1")
            for _ in range(200):
                if slow.aborted:
                    break
                await asyncio.sleep(0.01)
            assert slow.aborted  # the cancel reaped the sidecar and aborted the engine
            stream.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stream
        finally:
            done.set()
            await driver
        await harness.sidecar.stop()

    asyncio.run(run())
