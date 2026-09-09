"""Relay-frame carriage over a byte stream.

The reverse-rendezvous relay carries frames as Redis stream fields; a forward-dialed
target leg carries the same frames over one socket instead. This is that framing: a
length-prefixed metadata header followed by the frame's opaque payload as raw bytes, so
the payload crosses without a text encoding on the high-rate path.

A reader that sees a malformed header, an oversized frame, or a truncated body raises,
and the caller closes the connection: a stream whose framing is lost cannot resync.
"""

import asyncio
import json
from typing import Protocol

from .relay_frame import RelayDirection, RelayFrame, RelayFrameKind

_LENGTH_BYTES = 4
MAX_META_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


class FrameStreamError(Exception):
    """The stream's framing is unusable and its connection must be closed."""


class FrameWriter(Protocol):
    """The stream surface a frame is written to."""

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


class FrameSink(Protocol):
    """Carries one relay frame onward, whatever transport is behind it."""

    async def send(self, frame: RelayFrame) -> None: ...


def split_host_port(endpoint: str) -> tuple[str, int]:
    """Split a ``host:port`` endpoint, defaulting the host to loopback."""
    host, _, port = endpoint.rpartition(":")
    return host or "127.0.0.1", int(port)


def _meta(frame: RelayFrame) -> bytes:
    return json.dumps(
        {
            "kind": frame.kind.value,
            "session_id": frame.session_id,
            "invocation_id": frame.invocation_id,
            "idm": frame.idm,
            "direction": frame.direction.value,
            "seq": frame.seq,
            "ack": frame.ack,
        },
        separators=(",", ":"),
    ).encode()


async def write_relay_frame(writer: FrameWriter, frame: RelayFrame) -> None:
    """Write one frame's header and payload, then flush."""
    meta = _meta(frame)
    if len(meta) > MAX_META_BYTES or len(frame.payload) > MAX_PAYLOAD_BYTES:
        raise FrameStreamError("relay frame exceeds the stream frame bounds")
    writer.write(
        len(meta).to_bytes(_LENGTH_BYTES, "big")
        + meta
        + len(frame.payload).to_bytes(_LENGTH_BYTES, "big")
        + frame.payload
    )
    await writer.drain()


async def read_relay_frame(reader: asyncio.StreamReader) -> RelayFrame:
    """Read one frame, raising once the framing or its bounds no longer hold."""
    meta_len = int.from_bytes(await reader.readexactly(_LENGTH_BYTES), "big")
    if meta_len > MAX_META_BYTES:
        raise FrameStreamError(f"relay frame header too large: {meta_len}")
    raw_meta = await reader.readexactly(meta_len)
    payload_len = int.from_bytes(await reader.readexactly(_LENGTH_BYTES), "big")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise FrameStreamError(f"relay frame payload too large: {payload_len}")
    payload = await reader.readexactly(payload_len) if payload_len else b""
    try:
        meta = json.loads(raw_meta)
        return RelayFrame(
            kind=RelayFrameKind(meta["kind"]),
            session_id=str(meta["session_id"]),
            invocation_id=str(meta["invocation_id"]),
            idm=str(meta["idm"]),
            direction=RelayDirection(meta["direction"]),
            seq=int(meta.get("seq", 0)),
            ack=int(meta.get("ack", 0)),
            payload=payload,
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise FrameStreamError("undecodable relay frame header") from exc


__all__ = [
    "MAX_META_BYTES",
    "MAX_PAYLOAD_BYTES",
    "FrameSink",
    "FrameStreamError",
    "FrameWriter",
    "read_relay_frame",
    "split_host_port",
    "write_relay_frame",
]
