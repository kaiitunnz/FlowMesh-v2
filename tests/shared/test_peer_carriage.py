"""The origin's dialed carriage: selection, fallback, identity, and loss."""

import asyncio
import contextlib

import pytest

from shared.network.frame_stream import read_relay_frame, write_relay_frame
from shared.network.mtls import MutualTlsMaterial, client_context, server_context
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.resident.carriage import CarriageUnavailable, ResidentCarriagePlan
from shared.resident.peer_carriage import PeerCarriage, PeerCarriageLost
from shared.schemas.network import RouteObservationOutcome, Transport
from tests.support.certs import new_ca


def _frame(payload: bytes = b"x") -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        payload=payload,
    )


class _BaseSink:
    def __init__(self) -> None:
        self.frames: list[RelayFrame] = []

    async def send(self, frame: RelayFrame) -> None:
        self.frames.append(frame)


def _carriage(base, delivered, observed):
    return PeerCarriage(
        base=base,
        deliver=lambda frame: delivered.append(frame) or asyncio.sleep(0),
        observe=lambda session, transport, outcome: observed.append(
            (session, transport, outcome)
        ),
        ssl_context=None,
        connect_budget_sec=0.5,
    )


def _plan(transport: str, endpoint: str) -> ResidentCarriagePlan:
    return ResidentCarriagePlan(
        session_id="rly-1", selected_transport=transport, selected_endpoint=endpoint
    )


def _mtls(ca, identity: str, *sans: str) -> MutualTlsMaterial:
    issued = ca.issue(identity, *sans)
    return MutualTlsMaterial.from_b64(
        ca_b64=ca.ca_b64, cert_b64=issued.cert_b64, key_b64=issued.key_b64
    )


def _dial_over_mtls(target: MutualTlsMaterial, origin: MutualTlsMaterial):
    """Dial a live mutual-TLS target on loopback: (received, relayed, observed)."""
    base = _BaseSink()
    observed: list[tuple] = []
    carriage = PeerCarriage(
        base=base,
        deliver=lambda frame: asyncio.sleep(0),
        observe=lambda session, transport, outcome: observed.append(
            (session, transport, outcome)
        ),
        ssl_context=client_context(origin),
        connect_budget_sec=2.0,
    )
    received: list[RelayFrame] = []

    async def drive() -> None:
        async def serve(reader, writer):
            received.append(await read_relay_frame(reader))
            writer.close()

        server = await asyncio.start_server(
            serve, "127.0.0.1", 0, ssl=server_context(target)
        )
        port = server.sockets[0].getsockname()[1]
        async with server:
            sink = carriage.select(_plan("worker_direct", f"127.0.0.1:{port}"))
            await sink.send(_frame(b"request"))
            await asyncio.sleep(0.05)
            carriage.close("rly-1")

    asyncio.run(drive())
    return received, base.frames, observed


def test_a_relay_plan_carries_the_base_sink() -> None:
    base = _BaseSink()
    carriage = _carriage(base, [], [])
    assert carriage.select(_plan("control_relay", "")) is base


def test_a_peer_transport_without_an_address_is_refused_not_relayed() -> None:
    # Silently relaying a selection control made would carry the attempt over a
    # transport other than the one it chose.
    carriage = _carriage(_BaseSink(), [], [])
    with pytest.raises(CarriageUnavailable):
        carriage.select(_plan("worker_direct", ""))


def test_an_unreachable_target_falls_back_to_the_relay_under_one_credit() -> None:
    base = _BaseSink()
    observed: list[tuple] = []
    carriage = _carriage(base, [], observed)

    async def drive() -> None:
        # Port 1 on loopback refuses, so the dial fails before any frame is delivered.
        sink = carriage.select(_plan("worker_direct", "127.0.0.1:1"))
        await sink.send(_frame(b"hello"))

    asyncio.run(drive())

    assert [f.payload for f in base.frames] == [b"hello"]
    assert observed and observed[0][1] is Transport.WORKER_DIRECT
    assert observed[0][2] is not RouteObservationOutcome.VERIFIED


def test_a_reachable_target_carries_the_frames_and_verifies_the_path() -> None:
    base = _BaseSink()
    delivered: list[RelayFrame] = []
    observed: list[tuple] = []
    carriage = _carriage(base, delivered, observed)
    received: list[RelayFrame] = []

    async def drive() -> None:
        async def serve(reader, writer):
            received.append(await read_relay_frame(reader))
            await write_relay_frame(writer, _frame(b"answer"))
            writer.close()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            sink = carriage.select(_plan("worker_direct", f"127.0.0.1:{port}"))
            await sink.send(_frame(b"request"))
            for _ in range(50):
                if delivered:
                    break
                await asyncio.sleep(0.02)
            carriage.close("rly-1")

    asyncio.run(drive())

    # The frames crossed the socket, and the relay base carried nothing.
    assert [f.payload for f in received] == [b"request"]
    assert [f.payload for f in delivered] == [b"answer"]
    assert base.frames == []
    assert (observed[0][1], observed[0][2]) == (
        Transport.WORKER_DIRECT,
        RouteObservationOutcome.VERIFIED,
    )


def test_a_loss_after_delivery_is_ambiguous_rather_than_relayed() -> None:
    # Switching transports mid-attempt would replay a delivery whose outcome is
    # unknown, so the loss is raised for the drive to report as uncertain.
    base = _BaseSink()
    observed: list[tuple] = []
    carriage = _carriage(base, [], observed)

    async def drive() -> None:
        async def serve(reader, writer):
            await read_relay_frame(reader)
            writer.close()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            sink = carriage.select(_plan("worker_direct", f"127.0.0.1:{port}"))
            await sink.send(_frame(b"first"))
            with pytest.raises(PeerCarriageLost):
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    await sink.send(_frame(b"again"))

    asyncio.run(drive())

    assert base.frames == []
    assert observed[-1][2] is not RouteObservationOutcome.VERIFIED


def test_releasing_an_attempt_does_not_demote_a_healthy_transport() -> None:
    # A session torn down on its terminal ends the read with the same errors a genuine
    # loss raises; only the dial's verification should remain.
    base = _BaseSink()
    observed: list[tuple] = []
    carriage = _carriage(base, [], observed)

    async def drive() -> None:
        async def serve(reader, writer):
            # Ends when the released client closes, rather than outliving the test.
            with contextlib.suppress(OSError, asyncio.IncompleteReadError):
                await reader.read()
            writer.close()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            sink = carriage.select(_plan("worker_direct", f"127.0.0.1:{port}"))
            await sink.send(_frame(b"first"))
            carriage.close("rly-1")
            await asyncio.sleep(0.05)

    asyncio.run(drive())

    assert [o[2] for o in observed] == [RouteObservationOutcome.VERIFIED]


def test_a_target_whose_certificate_covers_the_dialed_host_is_carried() -> None:
    # The identity a route names is the endpoint it dials, and an operator can only put
    # a node's own address in its certificate — never the id the fabric assigns it at
    # registration. Requiring anything else refuses every legitimate target.
    ca = new_ca()
    received, relayed, observed = _dial_over_mtls(
        _mtls(ca, "node-target", "127.0.0.1"), _mtls(ca, "node-origin", "127.0.0.1")
    )

    assert [f.payload for f in received] == [b"request"]
    assert relayed == []
    assert observed[0][2] is RouteObservationOutcome.VERIFIED


def test_a_ca_signed_target_that_is_not_the_dialed_host_falls_back() -> None:
    # The deployment CA signs every node, so CA membership alone cannot tell the
    # selected target from any other holder of a certificate.
    ca = new_ca()
    received, relayed, observed = _dial_over_mtls(
        _mtls(ca, "node-elsewhere"), _mtls(ca, "node-origin", "127.0.0.1")
    )

    assert received == []
    assert [f.payload for f in relayed] == [b"request"]
    assert observed[0][2] is RouteObservationOutcome.TLS_FAILURE
