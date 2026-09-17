"""Contract P4: ``RelayFrame.tp`` survives all four codecs, or is absent when unset.

``RelayFrame`` is carried three ways -- Redis stream fields, the worker<->supervisor
attachment JSON, and the direct-dial byte stream ``frame_stream.py`` frames
independently -- so each codec pair is proven separately. ``frame_stream`` duplicates
the field list rather than delegating to ``to_wire``/``from_wire``, which is exactly the
kind of place a newly added field is dropped silently on the ``worker_direct`` /
``node_relay`` carriage path while the Redis and attachment codecs still pass.
"""

import asyncio

from shared.network.frame_stream import read_relay_frame, write_relay_frame
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind

_TP = "00-11111111111111111111111111111111-2222222222222222-01"


def _frame(tp: str | None) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        ack=0,
        payload=b"body",
        tp=tp,
    )


def test_fields_codec_round_trips_tp() -> None:
    frame = _frame(_TP)
    assert RelayFrame.from_fields(frame.to_fields()) == frame


def test_fields_codec_omits_tp_when_absent() -> None:
    frame = _frame(None)
    assert b"t" not in frame.to_fields()
    assert RelayFrame.from_fields(frame.to_fields()).tp is None


def test_wire_codec_round_trips_tp() -> None:
    frame = _frame(_TP)
    assert RelayFrame.from_wire(frame.to_wire()) == frame


def test_wire_codec_omits_tp_when_absent() -> None:
    frame = _frame(None)
    assert "tp" not in frame.to_wire()
    assert RelayFrame.from_wire(frame.to_wire()).tp is None


def test_worker_direct_and_node_relay_carriage_round_trips_tp() -> None:
    """The direct-dial byte stream codec (H2): its own ``_meta``/``read_relay_frame``,
    not ``to_wire``/``from_wire``, is what a ``worker_direct`` or ``node_relay`` peer
    session actually carries a frame over."""
    sent = _frame(_TP)

    async def scenario() -> RelayFrame:
        received: list[RelayFrame] = []
        done = asyncio.Event()

        async def serve(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            received.append(await read_relay_frame(reader))
            done.set()
            writer.close()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        await write_relay_frame(writer, sent)
        await asyncio.wait_for(done.wait(), timeout=5)
        writer.close()
        server.close()
        await server.wait_closed()
        return received[0]

    received = asyncio.run(scenario())
    assert received == sent
    assert received.tp == _TP


def test_worker_direct_and_node_relay_carriage_omits_tp_when_absent() -> None:
    sent = _frame(None)

    async def scenario() -> RelayFrame:
        received: list[RelayFrame] = []
        done = asyncio.Event()

        async def serve(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            received.append(await read_relay_frame(reader))
            done.set()
            writer.close()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        await write_relay_frame(writer, sent)
        await asyncio.wait_for(done.wait(), timeout=5)
        writer.close()
        server.close()
        await server.wait_closed()
        return received[0]

    received = asyncio.run(scenario())
    assert received.tp is None
