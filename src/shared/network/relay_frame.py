"""Reverse-relay frame codec and per-direction byte window.

Content-agnostic transport primitives shared by the servers that carry the relay and the
worker endpoints that own the protocol: a frame carries an opaque payload (the resident
fence and body ride inside it) plus routing and flow-control metadata, and a direction's
sender-side window bounds its in-flight bytes against the receiver's grants. They hold
no resident, claim, or admission concept.
"""

import asyncio
from dataclasses import dataclass
from enum import StrEnum


class RelayFrameKind(StrEnum):
    # Bulk DATA, plus the WINDOW grant and CANCEL that ride the priority control lane,
    # bypassing the byte window so a full data window cannot deadlock cancellation.
    DATA = "data"
    WINDOW = "window"
    CANCEL = "cancel"


class RelayDirection(StrEnum):
    """A frame's flow direction within a session."""

    ORIGIN_TO_TARGET = "o2t"
    TARGET_TO_ORIGIN = "t2o"


_CONTROL_KINDS = frozenset({RelayFrameKind.WINDOW, RelayFrameKind.CANCEL})


@dataclass(frozen=True)
class RelayFrame:
    """One relay frame. ``payload`` is opaque bytes the servers never read (the resident
    fence and body ride inside it); the rest are routing and flow-control metadata they
    may read. ``seq`` orders a direction's data for receiver dedup; ``ack`` carries a
    grant's cumulative byte credit."""

    kind: RelayFrameKind
    session_id: str
    invocation_id: str
    idm: str
    direction: RelayDirection
    seq: int = 0
    ack: int = 0
    payload: bytes = b""

    @property
    def is_control(self) -> bool:
        return self.kind in _CONTROL_KINDS

    def to_fields(self) -> dict[bytes, bytes]:
        fields: dict[bytes, bytes] = {
            b"k": self.kind.value.encode(),
            b"s": self.session_id.encode(),
            b"i": self.invocation_id.encode(),
            b"m": self.idm.encode(),
            b"d": self.direction.value.encode(),
            b"q": str(self.seq).encode(),
            b"a": str(self.ack).encode(),
        }
        if self.payload:
            fields[b"y"] = self.payload
        return fields

    @staticmethod
    def from_fields(fields: dict[bytes, bytes]) -> "RelayFrame":
        return RelayFrame(
            kind=RelayFrameKind(fields[b"k"].decode()),
            session_id=fields[b"s"].decode(),
            invocation_id=fields[b"i"].decode(),
            idm=fields[b"m"].decode(),
            direction=RelayDirection(fields[b"d"].decode()),
            seq=int(fields[b"q"]),
            ack=int(fields[b"a"]),
            payload=fields.get(b"y", b""),
        )


@dataclass
class WindowState:
    # A direction's byte window: the granted in-flight budget, the sent-but-unacked
    # bytes, and the cumulative bytes the receiver has confirmed. A cumulative ack is
    # idempotent — the receiver reports its running drained total and the sender frees
    # the delta since the last ack.
    granted: int
    used: int = 0
    acked: int = 0

    def can_send(self, size: int) -> bool:
        # A frame fits within the remaining window, or — when nothing is yet in flight —
        # is admitted alone so a payload larger than the whole window still makes
        # progress instead of deadlocking. Resident completions are chunked under the
        # window, so alone-admit only ever carries a single oversized control frame.
        return self.used == 0 or self.used + size <= self.granted

    def on_ack(self, cumulative: int) -> None:
        if cumulative > self.acked:
            self.used = max(0, self.used - (cumulative - self.acked))
            self.acked = cumulative


class DirectionWindow:
    """Sender-side flow control for one direction's data frames.

    A send reserves its byte size against the receiver's granted window, blocking until
    earlier bytes are acknowledged when the window is full, so a slow receiver holds no
    more than its advertised window in flight. A grant advances the cumulative ack and
    wakes a blocked sender. It carries only relay-window byte credit and never a service
    claim's admission credit.
    """

    def __init__(self, granted: int) -> None:
        self._state = WindowState(granted=granted)
        self._cond = asyncio.Condition()

    async def reserve(self, size: int) -> None:
        async with self._cond:
            while not self._state.can_send(size):
                await self._cond.wait()
            self._state.used += size

    async def grant(self, cumulative: int) -> None:
        async with self._cond:
            self._state.on_ack(cumulative)
            self._cond.notify_all()

    @property
    def in_flight(self) -> int:
        return self._state.used


__all__ = [
    "DirectionWindow",
    "RelayDirection",
    "RelayFrame",
    "RelayFrameKind",
    "WindowState",
]
