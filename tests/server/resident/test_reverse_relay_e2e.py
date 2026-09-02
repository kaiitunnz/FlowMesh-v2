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
from server.network.reverse_relay import (
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
    RelaySessionStore,
    RelayStreamStore,
)
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
                    RouteHop(transport=t, endpoint="", node_id="nde-o"),
                    RouteHop(transport=t, endpoint=sidecar_route, node_id="nde-t"),
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
            assert boot.acked
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


def test_cancel_wakes_a_blocked_origin_stream_promptly() -> None:
    async def run() -> None:
        slow = _SlowEngine()
        harness = _Harness(engine=slow)  # recv budget is 5s
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
            await asyncio.sleep(0.1)  # the stream blocks in _recv on the hung engine
            await harness.origin.cancel("s1")
            # The cancel sentinel wakes the blocked driver: the stream returns a loss
            # well within the recv budget instead of hanging until it elapses.
            result = await asyncio.wait_for(stream, timeout=2.0)
            assert not result.ok
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


def test_completion_reaps_target_state_and_deletes_the_session_record() -> None:
    async def run() -> None:
        harness = _Harness()
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
            result = await asyncio.wait_for(harness.origin.stream("s1", _auth()), 10.0)
            assert result.ok
            # The sidecar closed after DONE, so the target self-reaps its session state
            # and the origin's terminal deleted the routing record: neither leaks.
            sessions = RelaySessionStore(harness.redis)
            for _ in range(200):
                reaped = "s1" not in harness.target._targets
                gone = await sessions.load("s1") == {}
                if reaped and gone:
                    break
                await asyncio.sleep(0.01)
            assert "s1" not in harness.target._targets
            assert "s1" not in harness.target._t2o_windows
            assert await sessions.load("s1") == {}
        finally:
            done.set()
            await driver
        await harness.sidecar.stop()

    asyncio.run(run())


def test_a_re_forwarded_data_frame_is_deduped() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        endpoint = ResidentRelayEndpoint(redis, "nde-o")
        await endpoint.open_origin(
            "s1",
            invocation_id="inv-1",
            idm="idm-1",
            origin_node="nde-o",
            target_node="nde-t",
            sidecar_route="127.0.0.1:1",
        )
        frame = RelayFrame(
            kind=RelayFrameKind.DATA,
            session_id="s1",
            invocation_id="inv-1",
            idm="idm-1",
            direction=RelayDirection.TARGET_TO_ORIGIN,
            seq=1,
            payload=b"once",
        )
        await endpoint.on_frame(frame)
        await endpoint.on_frame(frame)  # a bridge re-forward with the same seq
        # The duplicate is dropped, so the origin sees the frame exactly once.
        assert endpoint._origin["s1"].qsize() == 1

    asyncio.run(run())


def test_a_flaky_sidecar_connect_is_ridden_out() -> None:
    async def run() -> None:
        calls: list[str] = []

        async def flaky(route: str) -> Any:
            calls.append(route)
            if len(calls) < 3:
                raise ConnectionRefusedError("sidecar not bound yet")
            return ("reader", "writer")

        endpoint = ResidentRelayEndpoint(
            FakeBinaryRedis(), "nde-t", connect=flaky, connect_backoff_sec=0.0
        )
        conn = await endpoint._connect_sidecar("127.0.0.1:1")
        # A bootstrap that raced the sidecar bind retries under the bounded budget
        # rather than losing the attempt to a re-drive.
        assert conn == ("reader", "writer") and len(calls) == 3

    asyncio.run(run())


def test_uncertain_bootstrap_redrives_leave_no_leaked_session_state() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        # No attachment/bridge consumes the up stream, so no ack ever returns and each
        # bootstrap times out uncertain — the re-drive shape.
        endpoint = ResidentRelayEndpoint(redis, "nde-o", recv_budget_sec=0.05)
        sessions = RelaySessionStore(redis)
        for i in range(3):
            sid = f"rly-{i}"
            boot = await endpoint.bootstrap(
                sid,
                route=_route("127.0.0.1:1"),
                handoff=_handoff(),
                request_payload='{"p":"hi"}',
            )
            assert not boot.acked and boot.uncertain
            # The abandoned attempt is reaped: no origin state, no durable record left.
            assert sid not in endpoint._origin
            assert sid not in endpoint._o2t_windows
            assert sid not in endpoint._t2o_consumed
            assert await sessions.load(sid) == {}
        # Nothing accumulated across the re-drives.
        assert endpoint._origin == {}
        assert endpoint._o2t_windows == {}
        assert endpoint._t2o_consumed == {}

    asyncio.run(run())


def test_a_failing_sidecar_connect_does_not_block_the_pump() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()

        async def dead_connect(route: str) -> Any:
            raise ConnectionRefusedError("sidecar gone")

        endpoint = ResidentRelayEndpoint(
            redis,
            "nde-t",
            connect=dead_connect,
            connect_tries=3,
            connect_backoff_sec=0.02,
        )
        sessions = RelaySessionStore(redis)
        await sessions.update(
            "rly-1",
            invocation_id="inv-1",
            idm="idm-1",
            origin_node="nde-o",
            target_node="nde-t",
            sidecar_route="127.0.0.1:1",
        )
        frame = RelayFrame(
            kind=RelayFrameKind.DATA,
            session_id="rly-1",
            invocation_id="inv-1",
            idm="idm-1",
            direction=RelayDirection.ORIGIN_TO_TARGET,
            seq=1,
            payload=b"boot",
        )
        start = asyncio.get_event_loop().time()
        await endpoint.on_frame(frame)
        # on_frame dispatched the connect off the pump loop and returned at once, rather
        # than blocking the node's multiplexed stream for the whole connect budget.
        assert asyncio.get_event_loop().time() - start < 0.05
        assert "rly-1" in endpoint._establishing
        # Draining the abandoned connect leaves no target state behind.
        est = endpoint._establishing.get("rly-1")
        if est is not None:
            with contextlib.suppress(Exception):
                await est
        assert "rly-1" not in endpoint._targets

    asyncio.run(run())
