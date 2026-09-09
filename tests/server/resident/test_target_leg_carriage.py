"""The root's target-leg carriage: realization, fallback, and ambiguity."""

import asyncio
import ssl

import pytest

from server.network.state import RouteObservationOutcome, Transport
from server.resident.target_leg import (
    TargetLegCarriage,
    TargetLegLost,
    TargetLegSupport,
)
from shared.network.frame_stream import read_relay_frame, write_relay_frame
from shared.network.mtls import MutualTlsMaterial, client_context, server_context
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.resident.carriage import CarriageUnavailable, ResidentCarriagePlan
from tests.support.certs import new_ca

_ROOT = "root.flowmesh"


def _material(identity: str) -> MutualTlsMaterial:
    ca = _CA
    issued = ca.issue(identity)
    return MutualTlsMaterial.from_b64(
        ca_b64=ca.ca_b64,
        cert_b64=issued.cert_b64,
        key_b64=issued.key_b64,
        root_identity=_ROOT,
    )


_CA = new_ca()


def _frame(seq: int = 1, payload: bytes = b"body") -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=seq,
        payload=payload,
    )


class _Recorder:
    """A base sink plus the observation and per-leg records the carriage writes."""

    def __init__(self) -> None:
        self.base: list[RelayFrame] = []
        self.inbound: list[RelayFrame] = []
        self.observations: list[tuple[str, Transport, RouteObservationOutcome]] = []
        self.legs: list[tuple[str, str, int]] = []

    async def send(self, frame: RelayFrame) -> None:
        self.base.append(frame)

    async def deliver(self, frame: RelayFrame) -> None:
        self.inbound.append(frame)

    def observe(
        self, session_id: str, transport: Transport, outcome: RouteObservationOutcome
    ) -> None:
        self.observations.append((session_id, transport, outcome))

    def meter(self, leg: str, transport: str, payload_bytes: int) -> None:
        self.legs.append((leg, transport, payload_bytes))


def _carriage(
    recorder: _Recorder, *, ssl_context: ssl.SSLContext | None
) -> TargetLegCarriage:
    return TargetLegSupport(
        ssl_context=ssl_context,
        observe=recorder.observe,
        meter=recorder.meter,
        connect_budget_sec=2.0,
    ).carriage(recorder, recorder.deliver)


def _plan(endpoint: str, transport: str = "worker_direct") -> ResidentCarriagePlan:
    return ResidentCarriagePlan(
        session_id="rly-1",
        target_leg_transport=transport,
        target_leg_endpoint=endpoint,
    )


def test_a_plan_naming_no_offload_carries_the_relay_base() -> None:
    recorder = _Recorder()
    carriage = _carriage(recorder, ssl_context=None)
    assert carriage.select(ResidentCarriagePlan(session_id="rly-1")) is recorder


def test_an_offload_this_root_cannot_open_is_refused_not_relayed() -> None:
    recorder = _Recorder()
    carriage = _carriage(recorder, ssl_context=None)
    with pytest.raises(CarriageUnavailable):
        carriage.select(_plan("127.0.0.1:1"))


def test_a_dial_failure_falls_back_to_the_base_and_demotes_the_path() -> None:
    recorder = _Recorder()

    async def scenario() -> None:
        carriage = _carriage(recorder, ssl_context=client_context(_material(_ROOT)))
        # Port 1 is closed on the loopback interface, so the dial cannot complete.
        sink = carriage.select(_plan("127.0.0.1:1"))
        await sink.send(_frame())
        await sink.send(_frame(seq=2))

    asyncio.run(scenario())
    assert [f.seq for f in recorder.base] == [1, 2]
    assert recorder.observations == [
        ("rly-1", Transport.WORKER_DIRECT, RouteObservationOutcome.CONNECT_FAILURE)
    ]
    assert {leg for leg, _, _ in recorder.legs} == {"target"}
    assert {transport for _, transport, _ in recorder.legs} == {"control_relay"}


def test_a_trusted_leg_carries_the_frames_and_delivers_the_target_reply() -> None:
    recorder = _Recorder()
    seen: list[RelayFrame] = []

    async def scenario() -> None:
        replied = asyncio.Event()

        async def serve(reader: asyncio.StreamReader, writer) -> None:
            seen.append(await read_relay_frame(reader))
            await write_relay_frame(writer, _frame(seq=9, payload=b"reply"))
            replied.set()

        server = await asyncio.start_server(
            serve, "127.0.0.1", 0, ssl=server_context(_material("worker.flowmesh"))
        )
        port = server.sockets[0].getsockname()[1]
        carriage = _carriage(recorder, ssl_context=client_context(_material(_ROOT)))
        sink = carriage.select(_plan(f"127.0.0.1:{port}"))
        await sink.send(_frame())
        await asyncio.wait_for(replied.wait(), timeout=5)
        await asyncio.sleep(0.05)
        carriage.close("rly-1")
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
    assert [f.payload for f in seen] == [b"body"]
    assert [f.payload for f in recorder.inbound] == [b"reply"]
    assert not recorder.base
    assert (
        "rly-1",
        Transport.WORKER_DIRECT,
        RouteObservationOutcome.VERIFIED,
    ) in recorder.observations
    assert ("target", "worker_direct", len(b"body")) in recorder.legs


def test_a_leg_lost_after_delivery_is_ambiguous_rather_than_a_fallback() -> None:
    recorder = _Recorder()

    async def scenario() -> None:
        async def serve(reader: asyncio.StreamReader, writer) -> None:
            await read_relay_frame(reader)
            writer.close()

        server = await asyncio.start_server(
            serve, "127.0.0.1", 0, ssl=server_context(_material("worker.flowmesh"))
        )
        port = server.sockets[0].getsockname()[1]
        carriage = _carriage(recorder, ssl_context=client_context(_material(_ROOT)))
        sink = carriage.select(_plan(f"127.0.0.1:{port}"))
        await sink.send(_frame())
        await asyncio.sleep(0.1)
        with pytest.raises(TargetLegLost):
            for seq in range(2, 60):
                await sink.send(_frame(seq=seq, payload=b"x" * 4096))
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
    # A loss after delivery never re-carries the session over the relay base.
    assert not recorder.base
