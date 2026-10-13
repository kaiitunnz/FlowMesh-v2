"""A queued sink answers over a connection without ever waiting on its peer.

A peer that stops reading never holds the caller; it is dropped once it lets more than
its bound queue, or once its connection takes nothing for the stall deadline.
"""

import asyncio
import contextlib
import socket
import time

import pytest

from shared.network.frame_stream import read_relay_frame
from shared.network.mtls_listener import ConnectionFrameSink, QueuedFrameSink
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind

_CHUNK = 64 * 1024


def _frame(seq: int) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        correlation_id="chg-1",
        operation_id="",
        direction=RelayDirection.TARGET_TO_ORIGIN,
        seq=seq,
        payload=b"x" * _CHUNK,
    )


async def _pair(
    limit_bytes: int, stall_sec: float
) -> tuple[QueuedFrameSink, asyncio.StreamReader, asyncio.StreamWriter, asyncio.Server]:
    """A queued sink on the server side of a loopback connection whose client never
    reads, with small socket buffers so they fill at once."""
    made: asyncio.Future[QueuedFrameSink] = asyncio.get_running_loop().create_future()

    async def serve(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, 4096
        )
        made.set_result(
            QueuedFrameSink(
                ConnectionFrameSink(writer),
                limit_bytes=limit_bytes,
                stall_sec=stall_sec,
            )
        )

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.connect(server.sockets[0].getsockname())
    sock.setblocking(False)
    reader, writer = await asyncio.open_connection(sock=sock)
    return await made, reader, writer, server


async def _ended(reader: asyncio.StreamReader, timeout: float) -> bool:
    """Whether the connection ends within ``timeout``, reading what was delivered."""
    try:
        async with asyncio.timeout(timeout):
            while await reader.read(1 << 20):
                pass
    except TimeoutError:
        return False
    except ConnectionError:
        return True
    return True


def test_sending_to_a_peer_that_stops_reading_never_waits() -> None:
    async def run() -> float:
        sink, reader, writer, server = await _pair(64 << 20, 60.0)
        started = time.monotonic()
        async with asyncio.timeout(5.0):
            for seq in range(1, 257):
                await sink.send(_frame(seq))
        elapsed = time.monotonic() - started
        sink.close()
        writer.close()
        server.close()
        return elapsed

    assert asyncio.run(run()) < 1.0


def test_a_peer_that_lets_more_than_its_bound_queue_is_dropped() -> None:
    async def run() -> bool:
        sink, reader, writer, server = await _pair(4 * _CHUNK, 60.0)
        async with asyncio.timeout(5.0):
            for seq in range(1, 65):
                await sink.send(_frame(seq))
        try:
            return await _ended(reader, 5.0)
        finally:
            sink.close()
            writer.close()
            server.close()

    assert asyncio.run(run())


def test_a_peer_whose_connection_takes_nothing_is_dropped_at_the_stall_deadline() -> (
    None
):
    async def run() -> bool:
        sink, reader, writer, server = await _pair(64 << 20, 0.3)
        async with asyncio.timeout(5.0):
            for seq in range(1, 129):
                await sink.send(_frame(seq))
        # The peer reads nothing until the deadline has passed.
        await asyncio.sleep(1.0)
        try:
            return await _ended(reader, 5.0)
        finally:
            sink.close()
            writer.close()
            server.close()

    assert asyncio.run(run())


def test_a_peer_that_keeps_reading_receives_every_frame_in_order() -> None:
    async def run() -> list[int]:
        sink, reader, writer, server = await _pair(4 * _CHUNK, 5.0)
        seqs: list[int] = []

        async def drain() -> None:
            with contextlib.suppress(asyncio.IncompleteReadError):
                while len(seqs) < 32:
                    seqs.append((await read_relay_frame(reader)).seq)

        reading = asyncio.ensure_future(drain())
        async with asyncio.timeout(5.0):
            for seq in range(1, 33):
                await sink.send(_frame(seq))
                # Keep within the bound as a reader that keeps up does.
                while len(seqs) < seq - 2:
                    await asyncio.sleep(0.001)
        await asyncio.wait_for(reading, 5.0)
        sink.close()
        writer.close()
        server.close()
        return seqs

    assert asyncio.run(run()) == list(range(1, 33))


@pytest.mark.parametrize("ending", ["overflow", "abort"])
def test_a_dropped_sink_ends_its_writer_cleanly(ending: str) -> None:
    async def run() -> BaseException | None:
        sink, reader, writer, server = await _pair(4 * _CHUNK, 60.0)
        for seq in range(1, 4):
            await sink.send(_frame(seq))
        # Let the writer block on the connection the peer is not reading.
        await asyncio.sleep(0.3)
        if ending == "overflow":
            for seq in range(10, 20):
                await sink.send(_frame(seq))
        else:
            sink.abort()
        try:
            async with asyncio.timeout(5.0):
                while not sink._task.done():
                    await asyncio.sleep(0.01)
            return sink._task.exception()
        finally:
            sink.close()
            writer.close()
            server.close()

    assert asyncio.run(run()) is None
