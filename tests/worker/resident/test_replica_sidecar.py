"""The replica sidecar lane gates a fence and serves its engine over the session.

An origin session wired sink-to-sink with the sidecar drives a real bootstrap, ack, and
authorized stream; a fence naming the wrong incarnation is refused before any engine
call; a definite engine 4xx is carried as a definite failure so the origin releases.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from shared.network.relay_frame import RelayFrame
from shared.resident.contracts import (
    AdmissionHandoff,
    ReplicaEndpoint,
    RouteAuthorization,
)
from shared.resident.session import ResidentRelaySession, ResidentSessionRole
from shared.resident.wire import (
    KIND_ACK,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_REJECT,
)
from worker.resident.engine import EngineResponse
from worker.resident.replica_sidecar import ResidentReplicaSidecar

_CHUNKS = ["resi", "dent ", "reply"]


async def _fake_engine(
    endpoint: ReplicaEndpoint,
    request: str | None,
    adapter_name: str | None = None,
    adapter_source: str | None = None,
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        for part in _CHUNKS:
            yield part

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


async def _failing_engine(
    endpoint: ReplicaEndpoint,
    request: str | None,
    adapter_name: str | None = None,
    adapter_source: str | None = None,
) -> EngineResponse:
    response = httpx.Response(400, request=httpx.Request("POST", "http://engine/v1"))
    raise httpx.HTTPStatusError(
        "bad request", request=response.request, response=response
    )


class _ToPeer:
    """Forwards each produced frame into a target coroutine."""

    def __init__(self) -> None:
        self.on_peer: Callable[[RelayFrame], Awaitable[None]] | None = None

    async def send(self, frame: RelayFrame) -> None:
        assert self.on_peer is not None
        await self.on_peer(frame)


def _handoff(incarnation: int = 1) -> dict:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=incarnation,
        listener_generation=1,
    ).model_dump(mode="json")


def _auth() -> dict:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    ).model_dump(mode="json")


def _harness(
    engine=_fake_engine,
) -> tuple[ResidentRelaySession, ResidentReplicaSidecar]:
    origin_sink, replica_sink = _ToPeer(), _ToPeer()
    sidecar = ResidentReplicaSidecar(sink=replica_sink, engine_open=engine)
    sidecar.bind(
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
        endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
    )
    origin = ResidentRelaySession(
        session_id="s1",
        invocation_id="inv-1",
        idm="idm-1",
        role=ResidentSessionRole.ORIGIN,
        sink=origin_sink,
    )
    origin_sink.on_peer = sidecar.on_frame
    replica_sink.on_peer = origin.on_frame
    return origin, sidecar


def test_gated_invocation_streams_from_the_engine() -> None:
    async def run() -> None:
        origin, sidecar = _harness()
        await origin.send_wire("bootstrap", handoff=_handoff(), request='{"p":"hi"}')
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_auth())
        parts: list[str] = []
        while True:
            msg = await origin.recv_wire(timeout=5.0)
            assert msg is not None
            if msg["kind"] == KIND_CHUNK:
                parts.append(str(msg["data"]))
            elif msg["kind"] == KIND_DONE:
                break
        assert "".join(parts) == "".join(_CHUNKS)
        await sidecar.aclose()

    asyncio.run(run())


def test_wrong_incarnation_is_refused_before_the_engine() -> None:
    async def run() -> None:
        origin, sidecar = _harness()
        await origin.send_wire(
            "bootstrap", handoff=_handoff(incarnation=9), request=None
        )
        reply = await origin.recv_wire(timeout=5.0)
        assert reply is not None and reply["kind"] == KIND_REJECT
        assert reply["reason"] == "wrong_incarnation"
        await sidecar.aclose()

    asyncio.run(run())


def test_definite_engine_failure_is_carried_definite() -> None:
    async def run() -> None:
        origin, sidecar = _harness(engine=_failing_engine)
        await origin.send_wire("bootstrap", handoff=_handoff(), request=None)
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_auth())
        failed = await origin.recv_wire(timeout=5.0)
        assert failed is not None and failed["kind"] == KIND_FAILED
        assert failed["definite"] is True
        await sidecar.aclose()

    asyncio.run(run())


def test_not_yet_bound_signals_a_transient_loss_not_a_definite_reject() -> None:
    async def run() -> None:
        origin_sink, replica_sink = _ToPeer(), _ToPeer()
        # No bind for rpl-1: the bind frame has not arrived yet (a cold-start race).
        sidecar = ResidentReplicaSidecar(sink=replica_sink, engine_open=_fake_engine)
        origin = ResidentRelaySession(
            session_id="s1",
            invocation_id="inv-1",
            idm="idm-1",
            role=ResidentSessionRole.ORIGIN,
            sink=origin_sink,
        )
        origin_sink.on_peer = sidecar.on_frame
        replica_sink.on_peer = origin.on_frame
        await origin.send_wire("bootstrap", handoff=_handoff(), request='{"p":"hi"}')
        reply = await origin.recv_wire(timeout=5.0)
        # Transient, not a definite fence reject: the origin holds and re-drives.
        assert reply is not None and reply["kind"] == KIND_FAILED
        assert reply["definite"] is False
        await sidecar.aclose()

    asyncio.run(run())


def test_unload_adapter_calls_the_engine_for_a_bound_replica() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def fake_unload(endpoint: ReplicaEndpoint, name: str) -> None:
            calls.append((endpoint.base_url, name))

        sink = _ToPeer()
        sidecar = ResidentReplicaSidecar(
            sink=sink, engine_open=_fake_engine, engine_unload=fake_unload
        )
        sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        )
        await sidecar.unload_adapter("rpl-1", "my-lora")
        assert calls == [("http://engine/v1", "my-lora")]

        # An unbound replica is a no-op: nothing to unload against.
        await sidecar.unload_adapter("rpl-unknown", "my-lora")
        assert calls == [("http://engine/v1", "my-lora")]

    asyncio.run(run())


def test_unload_adapter_swallows_an_engine_error() -> None:
    async def run() -> None:
        async def failing_unload(endpoint: ReplicaEndpoint, name: str) -> None:
            request = httpx.Request("POST", "http://engine/v1/unload_lora_adapter")
            raise httpx.HTTPStatusError(
                "boom",
                request=request,
                response=httpx.Response(500, request=request),
            )

        sidecar = ResidentReplicaSidecar(
            sink=_ToPeer(), engine_open=_fake_engine, engine_unload=failing_unload
        )
        sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        )
        # Best effort: a failed unload does not raise out of the lane.
        await sidecar.unload_adapter("rpl-1", "my-lora")

    asyncio.run(run())


def test_reap_invocation_tears_down_the_inflight_serve() -> None:
    async def run() -> None:
        aclosed = asyncio.Event()

        async def hanging_engine(
            endpoint: ReplicaEndpoint,
            request: str | None,
            adapter_name: str | None = None,
            adapter_source: str | None = None,
        ) -> EngineResponse:
            async def chunks() -> AsyncIterator[str]:
                await asyncio.Event().wait()  # never yields — the engine is slow
                yield ""  # pragma: no cover

            async def aclose() -> None:
                aclosed.set()

            return EngineResponse(chunks=chunks(), aclose=aclose)

        origin, sidecar = _harness(engine=hanging_engine)
        await origin.send_wire("bootstrap", handoff=_handoff(), request='{"p":"hi"}')
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_auth())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sidecar._inflight.get("inv-1") is not None:
                break
        assert sidecar._inflight.get("inv-1") is not None  # in-flight

        # A fenced terminal reaps the serve task: teardown closes the engine request.
        sidecar.reap_invocation("inv-1")
        await asyncio.wait_for(aclosed.wait(), timeout=5.0)
        await sidecar.aclose()

    asyncio.run(run())
