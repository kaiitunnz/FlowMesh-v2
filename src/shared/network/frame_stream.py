"""Relay-frame carriage over a byte stream.

The reverse-rendezvous relay carries frames as Redis stream fields; an origin that dials
its target directly carries the same frames over one socket instead. This is that
framing: a length-prefixed metadata header followed by the frame's opaque payload as raw
bytes, so the payload crosses without a text encoding on the high-rate path.

A reader that sees a malformed header, an oversized frame, or a truncated body raises,
and the caller closes the connection: a stream whose framing is lost cannot resync.

The same framing carries a reachability probe: the listener that reads one answers it
itself, so a probe exercises a stream listener's TLS and framing without entering any
relay session. A probe is no relay frame kind, so no relay codec decodes one.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .relay_frame import RelayDirection, RelayFrame, RelayFrameKind

_LENGTH_BYTES = 4
MAX_META_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_PROBE_BYTES = 1024
_PROBE_KIND = "probe"


class FrameStreamError(Exception):
    """The stream's framing is unusable and its connection must be closed."""


@dataclass(frozen=True)
class ProbeFrame:
    """A reachability probe, answered by the stream listener that reads it."""

    payload: bytes


class FrameWriter(Protocol):
    """The stream surface a frame is written to."""

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


class FrameSink(Protocol):
    """Carries one relay frame onward, whatever transport is behind it."""

    async def send(self, frame: RelayFrame) -> None: ...


class WireFrameSink:
    """Hands each produced frame to a transport that carries it as a wire dict.

    A worker's lane sends through this: the frame leaves as an attachment event, and the
    supervisor bridges it onward without reading it.
    """

    def __init__(self, push_frame: Callable[[dict[str, Any]], None]) -> None:
        self._push_frame = push_frame

    async def send(self, frame: RelayFrame) -> None:
        self._push_frame(frame.to_wire())


def split_host_port(endpoint: str) -> tuple[str, int]:
    """Split a ``host:port`` endpoint, defaulting the host to loopback."""
    host, _, port = endpoint.rpartition(":")
    return host or "127.0.0.1", int(port)


def _meta(frame: RelayFrame) -> bytes:
    meta: dict[str, Any] = {
        "kind": frame.kind.value,
        "session_id": frame.session_id,
        "correlation_id": frame.correlation_id,
        "operation_id": frame.operation_id,
        "direction": frame.direction.value,
        "seq": frame.seq,
        "ack": frame.ack,
    }
    if frame.tp:
        meta["tp"] = frame.tp
    return json.dumps(meta, separators=(",", ":")).encode()


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


async def write_probe(writer: FrameWriter, payload: bytes) -> None:
    """Write one probe, then flush."""
    if len(payload) > MAX_PROBE_BYTES:
        raise FrameStreamError("probe payload exceeds its bound")
    meta = json.dumps({"kind": _PROBE_KIND}, separators=(",", ":")).encode()
    writer.write(
        len(meta).to_bytes(_LENGTH_BYTES, "big")
        + meta
        + len(payload).to_bytes(_LENGTH_BYTES, "big")
        + payload
    )
    await writer.drain()


async def read_relay_frame(reader: asyncio.StreamReader) -> RelayFrame:
    """Read one relay frame, raising on a probe or once the framing no longer holds."""
    frame = await read_stream_frame(reader)
    if isinstance(frame, ProbeFrame):
        raise FrameStreamError("a probe is not a relay frame")
    return frame


async def read_stream_frame(
    reader: asyncio.StreamReader,
) -> RelayFrame | ProbeFrame:
    """Read one frame or probe, raising once its framing or bounds no longer hold."""
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
        if meta["kind"] == _PROBE_KIND:
            if payload_len > MAX_PROBE_BYTES:
                raise FrameStreamError(f"probe payload too large: {payload_len}")
            return ProbeFrame(payload)
        return RelayFrame(
            kind=RelayFrameKind(meta["kind"]),
            session_id=str(meta["session_id"]),
            correlation_id=str(meta["correlation_id"]),
            operation_id=str(meta["operation_id"]),
            direction=RelayDirection(meta["direction"]),
            seq=int(meta.get("seq", 0)),
            ack=int(meta.get("ack", 0)),
            payload=payload,
            tp=meta.get("tp") or None,
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise FrameStreamError("undecodable relay frame header") from exc


__all__ = [
    "MAX_META_BYTES",
    "MAX_PAYLOAD_BYTES",
    "MAX_PROBE_BYTES",
    "FrameSink",
    "FrameStreamError",
    "FrameWriter",
    "ProbeFrame",
    "WireFrameSink",
    "read_relay_frame",
    "read_stream_frame",
    "split_host_port",
    "write_probe",
    "write_relay_frame",
]
