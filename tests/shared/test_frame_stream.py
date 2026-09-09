"""Relay-frame carriage over a byte stream: round trip, bounds, and framing loss."""

import asyncio

import pytest

from shared.network.frame_stream import (
    MAX_PAYLOAD_BYTES,
    FrameStreamError,
    read_relay_frame,
    split_host_port,
    write_relay_frame,
)
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind


def _frame(payload: bytes = b"body", seq: int = 1) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=seq,
        ack=7,
        payload=payload,
    )


async def _round_trip(*frames: RelayFrame) -> list[RelayFrame]:
    received: list[RelayFrame] = []
    done = asyncio.Event()

    async def serve(reader: asyncio.StreamReader, writer) -> None:
        for _ in frames:
            received.append(await read_relay_frame(reader))
        done.set()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    for frame in frames:
        await write_relay_frame(writer, frame)
    await asyncio.wait_for(done.wait(), timeout=5)
    writer.close()
    server.close()
    await server.wait_closed()
    return received


def test_frames_round_trip_with_their_payload_and_routing_fields() -> None:
    sent = [_frame(b"a" * 1024, seq=1), _frame(b"", seq=2)]
    received = asyncio.run(_round_trip(*sent))
    assert received == sent


def test_an_oversized_payload_is_refused_rather_than_written() -> None:
    async def scenario() -> None:
        sent: list[bytes] = []

        class _Writer:
            def write(self, data: bytes) -> None:
                sent.append(data)

            async def drain(self) -> None:
                return None

        with pytest.raises(FrameStreamError):
            await write_relay_frame(_Writer(), _frame(b"x" * (MAX_PAYLOAD_BYTES + 1)))
        assert not sent

    asyncio.run(scenario())


def test_a_truncated_header_ends_the_stream() -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x00\x00")
        reader.feed_eof()
        with pytest.raises(asyncio.IncompleteReadError):
            await read_relay_frame(reader)

    asyncio.run(scenario())


def test_an_undecodable_header_raises_a_framing_error() -> None:
    async def scenario() -> None:
        meta = b"not-json"
        reader = asyncio.StreamReader()
        reader.feed_data(len(meta).to_bytes(4, "big") + meta + (0).to_bytes(4, "big"))
        reader.feed_eof()
        with pytest.raises(FrameStreamError):
            await read_relay_frame(reader)

    asyncio.run(scenario())


def test_an_oversized_declared_header_raises_before_reading_it() -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data((1 << 30).to_bytes(4, "big"))
        reader.feed_eof()
        with pytest.raises(FrameStreamError):
            await read_relay_frame(reader)

    asyncio.run(scenario())


def test_split_host_port_defaults_the_host_to_loopback() -> None:
    assert split_host_port("10.0.0.4:9100") == ("10.0.0.4", 9100)
    assert split_host_port(":9100") == ("127.0.0.1", 9100)
