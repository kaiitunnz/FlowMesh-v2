"""The worker's claim-gated target-leg listener: who it serves and what it carries."""

import asyncio
import socket
import ssl

import pytest

from shared.network.frame_stream import read_relay_frame, write_relay_frame
from shared.network.mtls import MutualTlsMaterial, client_context, server_context
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.resident.transport import ResidentFrameSink
from tests.support.certs import new_ca
from worker.resident.direct_listener import ResidentDirectListener

_ROOT = "root.flowmesh"
_CA = new_ca()


def _material(identity: str) -> MutualTlsMaterial:
    issued = _CA.issue(identity)
    return MutualTlsMaterial.from_b64(
        ca_b64=_CA.ca_b64,
        cert_b64=issued.cert_b64,
        key_b64=issued.key_b64,
        root_identity=_ROOT,
    )


def _frame(payload: bytes = b"bootstrap") -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        payload=payload,
    )


def _bound() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    sock.setblocking(False)
    return sock


async def _listener(delivered: list[RelayFrame]) -> tuple[ResidentDirectListener, int]:
    async def deliver(frame: RelayFrame, sink: ResidentFrameSink) -> None:
        delivered.append(frame)
        await sink.send(_frame(b"answer"))

    sock = _bound()
    port = sock.getsockname()[1]
    listener = ResidentDirectListener(
        sock=sock,
        ssl_context=server_context(_material("worker.flowmesh")),
        root_identity=_ROOT,
        deliver=deliver,
    )
    await listener.start()
    return listener, port


def test_the_root_is_served_and_answered_over_its_own_connection() -> None:
    delivered: list[RelayFrame] = []

    async def scenario() -> RelayFrame:
        listener, port = await _listener(delivered)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", port, ssl=client_context(_material(_ROOT))
        )
        await write_relay_frame(writer, _frame())
        answer = await asyncio.wait_for(read_relay_frame(reader), timeout=5)
        writer.close()
        await listener.stop()
        return answer

    answer = asyncio.run(scenario())
    assert [f.payload for f in delivered] == [b"bootstrap"]
    assert answer.payload == b"answer"


def test_a_ca_signed_dialer_that_is_not_the_root_is_refused() -> None:
    delivered: list[RelayFrame] = []

    async def scenario() -> None:
        listener, port = await _listener(delivered)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", port, ssl=client_context(_material("impostor.flowmesh"))
        )
        await write_relay_frame(writer, _frame())
        with pytest.raises(
            (asyncio.IncompleteReadError, ConnectionError, ssl.SSLError)
        ):
            await asyncio.wait_for(read_relay_frame(reader), timeout=5)
        writer.close()
        await listener.stop()

    asyncio.run(scenario())
    assert delivered == []


def test_a_dialer_presenting_no_certificate_carries_nothing() -> None:
    delivered: list[RelayFrame] = []

    async def scenario() -> None:
        listener, port = await _listener(delivered)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=context)
        with pytest.raises(
            (asyncio.IncompleteReadError, ConnectionError, ssl.SSLError, OSError)
        ):
            await write_relay_frame(writer, _frame())
            await asyncio.wait_for(read_relay_frame(reader), timeout=5)
        writer.close()
        await listener.stop()

    asyncio.run(scenario())
    assert delivered == []
