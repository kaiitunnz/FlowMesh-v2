"""One resident invocation's windowed relay session, owned by its endpoint.

Each side of an invocation runs one session: the origin — a caller worker's origin
driver or the gated serve edge in the root — that drives it, and the replica worker that
serves it. A session sends its role's data direction and reads the other, bounding its
in-flight bytes with a sender-side window and granting the peer only as fast as it
drains, so a slow consumer backpressures a fast producer end to end. The servers relay
the frames opaquely: the session owns the cursor, window, and the protocol, and the
supervisors never read them.
"""

import asyncio
from enum import Enum
from typing import Any

from shared.network.relay_frame import (
    DirectionWindow,
    RelayDirection,
    RelayFrame,
    RelayFrameKind,
)
from shared.resident.wire import decode_msg, encode_msg

from .transport import ResidentFrameSink


class ResidentSessionRole(Enum):
    """Which end of the invocation this session drives."""

    ORIGIN = "origin"
    REPLICA = "replica"


class ResidentRelaySession:
    """A worker's end of one invocation's reverse-relay data session.

    The origin role sends origin-to-target and receives target-to-origin; the replica
    role is the mirror. A received window frame advances this side's send window; a
    drained data frame emits a cumulative window grant so the peer never holds more than
    its window in flight. Cancellation wakes a blocked receiver so a phase returns.
    """

    def __init__(
        self,
        *,
        session_id: str,
        invocation_id: str,
        idm: str,
        role: ResidentSessionRole,
        sink: ResidentFrameSink,
        window_bytes: int = 65536,
    ) -> None:
        self._session_id = session_id
        self._invocation_id = invocation_id
        self._idm = idm
        self._send_dir = (
            RelayDirection.ORIGIN_TO_TARGET
            if role is ResidentSessionRole.ORIGIN
            else RelayDirection.TARGET_TO_ORIGIN
        )
        self._sink = sink
        self._window = DirectionWindow(window_bytes)
        self._recv: asyncio.Queue[bytes] = asyncio.Queue()
        self._cancelled = asyncio.Event()
        self._send_seq = 0
        self._recv_seq = 0
        self._recv_consumed = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def send_wire(self, kind: str, **fields: Any) -> None:
        """Frame and send one wire message, blocking on a full send window."""
        payload = encode_msg(kind, **fields)
        await self._window.reserve(len(payload))
        self._send_seq += 1
        await self._sink.send(
            RelayFrame(
                kind=RelayFrameKind.DATA,
                session_id=self._session_id,
                invocation_id=self._invocation_id,
                idm=self._idm,
                direction=self._send_dir,
                seq=self._send_seq,
                payload=payload,
            )
        )

    async def recv_wire(self, timeout: float) -> dict[str, Any] | None:
        """Await and decode the next wire message, or ``None`` on cancel or timeout.

        Draining a frame grants the peer its cumulative consumed bytes, so the peer's
        send window advances only as this side reads.
        """
        getter = asyncio.ensure_future(self._recv.get())
        waiter = asyncio.ensure_future(self._cancelled.wait())
        try:
            done, _ = await asyncio.wait(
                {getter, waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if getter not in done:
                return None
            payload = getter.result()
        finally:
            for task in (getter, waiter):
                if not task.done():
                    task.cancel()
        self._recv_consumed += len(payload)
        await self._grant()
        return decode_msg(payload)

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one inbound frame: a grant, a cancel, or deduplicated data."""
        if frame.kind is RelayFrameKind.WINDOW:
            await self._window.grant(frame.ack)
            return
        if frame.kind is RelayFrameKind.CANCEL:
            self._cancelled.set()
            return
        if frame.seq <= self._recv_seq:
            # A bridge re-forward re-delivers a data frame with a fresh entry id; drop
            # it by sequence so it lands once within this session's lifetime.
            return
        self._recv_seq = frame.seq
        self._recv.put_nowait(frame.payload)

    async def cancel(self) -> None:
        """Signal the peer to reap this session and wake a blocked receiver here."""
        self._cancelled.set()
        await self._sink.send(
            RelayFrame(
                kind=RelayFrameKind.CANCEL,
                session_id=self._session_id,
                invocation_id=self._invocation_id,
                idm=self._idm,
                direction=self._send_dir,
            )
        )

    async def _grant(self) -> None:
        await self._sink.send(
            RelayFrame(
                kind=RelayFrameKind.WINDOW,
                session_id=self._session_id,
                invocation_id=self._invocation_id,
                idm=self._idm,
                direction=self._send_dir,
                ack=self._recv_consumed,
            )
        )
