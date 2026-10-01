"""A byte stream over a framed relay session cannot survive a lost frame."""

import asyncio

import pytest

from shared.network.byte_stream import ByteStreamChannel, StreamClosed
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.network.session import RelaySessionRole


class _Wire:
    """Delivers each frame to the peer channel, dropping the data frames named."""

    def __init__(self, drop_seqs: frozenset[int] = frozenset()) -> None:
        self.peer: ByteStreamChannel | None = None
        self.drop_seqs = drop_seqs
        self.sent: list[RelayFrame] = []

    async def send(self, frame: RelayFrame) -> None:
        self.sent.append(frame)
        if frame.kind is RelayFrameKind.DATA and frame.seq in self.drop_seqs:
            return
        assert self.peer is not None
        await self.peer.on_frame(frame)


def _pair(
    drop_seqs: frozenset[int] = frozenset(),
) -> tuple[ByteStreamChannel, ByteStreamChannel, _Wire, _Wire]:
    to_target, to_origin = _Wire(drop_seqs), _Wire()
    origin = ByteStreamChannel("rly-1", RelaySessionRole.ORIGIN, to_target)
    target = ByteStreamChannel("rly-1", RelaySessionRole.TARGET, to_origin)
    to_target.peer, to_origin.peer = target, origin
    return origin, target, to_target, to_origin


def test_a_lost_frame_ends_the_stream_at_both_ends() -> None:
    async def run() -> None:
        origin, target, _, to_origin = _pair(drop_seqs=frozenset({2}))
        await origin.send(b"first")
        await origin.send(b"lost")
        await origin.send(b"third")
        assert await target.recv() == b"first"
        with pytest.raises(StreamClosed):
            await target.recv()
        await target.abort()
        assert to_origin.sent[-1].kind is RelayFrameKind.CANCEL
        assert origin.closed

    asyncio.run(run())


def test_a_reforwarded_frame_is_dropped_not_treated_as_a_gap() -> None:
    async def run() -> None:
        origin, target, to_target, _ = _pair()
        await origin.send(b"once")
        await target.on_frame(to_target.sent[0])
        await origin.send_eof()
        assert await target.recv() == b"once"
        assert await target.recv() is None
        assert not target.closed

    asyncio.run(run())


def test_a_send_blocked_on_a_full_window_returns_when_the_peer_cancels() -> None:
    async def run() -> None:
        to_target = _Wire()
        origin = ByteStreamChannel(
            "rly-1", RelaySessionRole.ORIGIN, to_target, window_bytes=1
        )
        to_target.peer = ByteStreamChannel("rly-1", RelaySessionRole.TARGET, _Wire())
        sending = asyncio.ensure_future(origin.send(b"a" * 100_000))
        await asyncio.sleep(0.01)
        assert not sending.done()
        await origin.on_frame(
            RelayFrame(
                kind=RelayFrameKind.CANCEL,
                session_id="rly-1",
                direction=RelayDirection.TARGET_TO_ORIGIN,
            )
        )
        with pytest.raises(StreamClosed):
            await asyncio.wait_for(sending, 1)

    asyncio.run(run())
