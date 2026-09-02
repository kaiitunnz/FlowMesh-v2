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


class _FloodEngine:
    """Emits a completion far larger than one window, recording an aborted stream."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.aborted = False

    async def __call__(
        self, endpoint: ReplicaEndpoint, request: str | None
    ) -> EngineResponse:
        chunks = self._chunks
        outer = self

        async def gen() -> AsyncIterator[str]:
            try:
                for part in chunks:
                    yield part
            except GeneratorExit:
                outer.aborted = True
                raise

        async def aclose() -> None:
            outer.aborted = True

        return EngineResponse(chunks=gen(), aclose=aclose)


async def _pump_until(harness: "_Harness", ready: Any, limit: int = 80) -> None:
    for _ in range(limit):
        await harness.pump_step()
        if ready():
            return


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

    def __init__(self, engine: Any = _fake_engine, window_bytes: int = 65536) -> None:
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
        self.origin = ResidentRelayEndpoint(
            self.redis, "nde-o", recv_budget_sec=5.0, window_bytes=window_bytes
        )
        self.target = ResidentRelayEndpoint(
            self.redis, "nde-t", recv_budget_sec=5.0, window_bytes=window_bytes
        )
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
        await self.pump_producer()
        await self.origin_attach.pump_once()
        await asyncio.sleep(0.002)

    async def pump_producer(self) -> None:
        # The origin-to-target request path plus the target-to-origin forwarding, but
        # not the origin's own consume: a slow origin that never drains its down stream
        # grants nothing, so the target fills its window and backpressures.
        await self.bridge.pump_node("nde-o")
        await self.target_attach.pump_once()
        await self.bridge.pump_node("nde-t")
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


def test_a_slow_origin_backpressures_the_target_to_its_window() -> None:
    async def run() -> None:
        chunks = ["x" * 100 for _ in range(20)]
        window = 256
        harness = _Harness(engine=_FloodEngine(chunks), window_bytes=window)
        route = _route(await harness.start())
        boot = asyncio.ensure_future(
            harness.origin.bootstrap(
                "s1", route=route, handoff=_handoff(), request_payload='{"p":"hi"}'
            )
        )
        await _pump_until(harness, boot.done)
        assert (await boot).acked
        # Start the stream but pump only the producer side, never the origin's consume:
        # the target fills its window and blocks, holding no more than it was granted.
        stream_task = asyncio.ensure_future(harness.origin.stream("s1", _auth()))
        for _ in range(60):
            await harness.pump_producer()
        win = harness.target._t2o_windows.get("s1")
        assert win is not None
        assert 0 < win.in_flight <= window
        assert not stream_task.done()  # completion withheld: this is the backpressure
        # Resume the origin's consume: its grants advance the window and the stream
        # completes with every chunk reassembled, in order, with no loss.
        done = asyncio.Event()
        driver = asyncio.ensure_future(_drive(harness, done))
        try:
            result = await asyncio.wait_for(stream_task, timeout=10.0)
        finally:
            done.set()
            await driver
        assert result.ok and result.completion == "".join(chunks)
        await harness.sidecar.stop()

    asyncio.run(run())


def test_a_cancel_preempts_a_full_data_window_without_deadlock() -> None:
    async def run() -> None:
        flood = _FloodEngine(["y" * 100 for _ in range(20)])
        window = 256
        harness = _Harness(engine=flood, window_bytes=window)
        route = _route(await harness.start())
        boot = asyncio.ensure_future(
            harness.origin.bootstrap(
                "s1", route=route, handoff=_handoff(), request_payload='{"p":"hi"}'
            )
        )
        await _pump_until(harness, boot.done)
        assert (await boot).acked
        stream_task = asyncio.ensure_future(harness.origin.stream("s1", _auth()))
        for _ in range(60):
            await harness.pump_producer()
        win = harness.target._t2o_windows.get("s1")
        assert win is not None and 0 < win.in_flight <= window  # full window, blocked
        assert "s1" in harness.target._targets
        # A cancel rides the priority lane, bypasses the full data window, reaps the
        # co-located sidecar, and aborts the engine — no deadlock under a full window.
        await harness.origin.cancel("s1")
        for _ in range(200):
            await harness.pump_producer()
            if flood.aborted and "s1" not in harness.target._targets:
                break
        assert flood.aborted
        assert "s1" not in harness.target._targets
        stream_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await stream_task
        await harness.sidecar.stop()

    asyncio.run(run())
