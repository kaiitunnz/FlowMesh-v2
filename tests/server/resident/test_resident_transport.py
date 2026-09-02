"""The resident deputy and sidecar carry a claim-gated invocation over the forward-dial
offloads (worker_direct and node_relay); the universal control_relay rides the
reverse-rendezvous relay covered by the substrate tests.

Over real loopback sockets — with the node relay as the intermediate hop and a
fake engine behind the sidecar — a bootstrap is admitted and its response streams back;
a fence rejection is refused before the engine body is forwarded and surfaces as a
stopped delivery at the deputy; cancellation is honored only under a valid
authorization; and a pre-send connect failure falls to the next candidate.
"""

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from server.network.listeners import NetworkPlaneListeners
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
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    )
    base.update(overrides)
    return RouteAuthorization(**base)


_EngineOpen = Callable[[ReplicaEndpoint, str | None], Awaitable[EngineResponse]]


class _Fixture:
    """A resident sidecar plus the substrate relay hops, all on loopback."""

    def __init__(self, engine: _EngineOpen = _fake_engine) -> None:
        self.sidecar_port = _free_port()
        self.relay_port = _free_port()
        self.loads: list[str] = []
        self.sidecar = ResidentSidecarListener(
            ResidentSidecarServer(
                gate=_gate(),
                endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
                engine_open=engine,
                on_load=lambda ev: self.loads.append(ev.operation),
            ),
            route=f"127.0.0.1:{self.sidecar_port}",
        )
        self.relays = NetworkPlaneListeners(
            sidecar_url=f"127.0.0.1:{_free_port()}",
            endpoint_url=f"127.0.0.1:{self.relay_port}",
            buffer_bytes=65536,
        )

    async def __aenter__(self) -> "_Fixture":
        await self.sidecar.start()
        await self.relays.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.relays.stop()
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

    for select in (_Fixture.direct, _Fixture.node_relay):
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


class _SlowEngine:
    """Hangs in the request until cancelled, recording that the engine was aborted."""

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

        async def chunks() -> AsyncIterator[str]:
            yield "done"

        async def aclose() -> None:
            return None

        return EngineResponse(chunks=chunks(), aclose=aclose)


def test_cancel_aborts_the_engine_and_reaps_both_ends() -> None:
    async def run() -> None:
        slow = _SlowEngine()
        async with _Fixture(engine=slow) as fx:
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            boot = await deputy.bootstrap(
                "s1", fx.direct(), _handoff(), '{"prompt": "hi"}'
            )
            assert boot.acked
            stream = asyncio.ensure_future(deputy.stream("s1", _auth()))
            await asyncio.sleep(0.1)  # let the stream reach the hung engine request
            cancel = await deputy.cancel("s1")
            assert cancel.ok and cancel.cancelled
            with contextlib.suppress(asyncio.CancelledError):
                await stream
            # Both ends are reaped and the co-located engine request was aborted.
            assert "s1" not in deputy._sessions
            for _ in range(100):
                if slow.aborted:
                    break
                await asyncio.sleep(0.01)
            assert slow.aborted

    asyncio.run(run())


class _RefusingEngine:
    """Raises an engine HTTP status error so the sidecar classifies the failure."""

    def __init__(self, status: int) -> None:
        self._status = status

    async def __call__(
        self, endpoint: ReplicaEndpoint, request: str | None
    ) -> EngineResponse:
        req = httpx.Request("POST", "http://engine/v1/chat/completions")
        resp = httpx.Response(self._status, request=req)
        raise httpx.HTTPStatusError("refused", request=req, response=resp)


def test_engine_4xx_is_definite_but_429_and_5xx_are_uncertain() -> None:
    async def run(status: int, expect_definite: bool) -> None:
        async with _Fixture(engine=_RefusingEngine(status)) as fx:
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            boot = await deputy.bootstrap("s1", fx.direct(), _handoff(), None)
            assert boot.acked
            result = await deputy.stream("s1", _auth())
            # A definite refusal lets the caller release; an uncertain one holds.
            assert not result.ok and result.definite is expect_definite

    for status, expect_definite in [
        (400, True),
        (404, True),
        (422, True),
        (429, False),
        (500, False),
        (503, False),
    ]:
        asyncio.run(run(status, expect_definite))


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


def test_stranded_session_is_reaped_after_its_ttl() -> None:
    async def run() -> None:
        async with _Fixture() as fx:
            deputy = ResidentInvocationDeputy(
                connect_budget_sec=3.0, session_ttl_sec=0.05
            )
            boot = await deputy.bootstrap("s1", fx.direct(), _handoff(), None)
            assert boot.acked and "s1" in deputy._sessions
            await asyncio.sleep(0.2)
            # No stream or cancel followed the ack: the reaper closed the held session.
            assert "s1" not in deputy._sessions

    asyncio.run(run())


def test_redrive_replaces_and_closes_the_prior_session() -> None:
    async def run() -> None:
        async with _Fixture() as fx:
            deputy = ResidentInvocationDeputy(connect_budget_sec=3.0)
            assert (await deputy.bootstrap("s1", fx.direct(), _handoff(), None)).acked
            first = deputy._sessions["s1"]
            assert (await deputy.bootstrap("s1", fx.direct(), _handoff(), None)).acked
            # A re-drive under the same session id closes the prior connection.
            assert deputy._sessions["s1"] is not first
            assert first.writer.is_closing()
            await deputy.reap("s1")

    asyncio.run(run())
