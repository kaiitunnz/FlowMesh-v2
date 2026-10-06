"""Relay-frame carriage over a byte stream.

The reverse-rendezvous relay carries frames as Redis stream fields; an origin that dials
its target directly carries the same frames over one socket instead. This is that
framing: a length-prefixed metadata header followed by the frame's opaque payload as raw
bytes, so the payload crosses without a text encoding on the high-rate path.

A reader that sees a malformed header, an oversized frame, or a truncated body raises,
and the caller closes the connection: a stream whose framing is lost cannot resync.

The same framing carries two exchanges a stream listener answers itself, neither a relay
frame kind, so no relay codec decodes either. A reachability probe exercises a
listener's TLS and framing without entering any relay session. A connection accept lets
a dialer learn that the target took its connection before it sends a session on it.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .relay_frame import RelayDirection, RelayFrame, RelayFrameKind

_LENGTH_BYTES = 4
MAX_META_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_PROBE_BYTES = 1024
_PROBE_KIND = "probe"
_PROBE_META = json.dumps({"kind": _PROBE_KIND}, separators=(",", ":")).encode()
_ACCEPT_KIND = "accept"
_ACCEPT_META = json.dumps({"kind": _ACCEPT_KIND}, separators=(",", ":")).encode()
_MAX_ACCEPT_BYTES = 16


class FrameStreamError(Exception):
    """The stream's framing is unusable and its connection must be closed."""


@dataclass(frozen=True)
class ProbeFrame:
    """A reachability probe, answered by the stream listener that reads it."""

    payload: bytes


class AcceptStatus(StrEnum):
    """One step of a connection accept: the dialer's request or the target's answer."""

    REQUEST = "request"
    ACCEPTED = "accepted"
    BUSY = "busy"
    REFUSED = "refused"


@dataclass(frozen=True)
class AcceptFrame:
    """A connection accept step; a stream listener answers the request it reads."""

    status: AcceptStatus


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


async def _write_framed(writer: FrameWriter, meta: bytes, payload: bytes) -> None:
    writer.write(
        len(meta).to_bytes(_LENGTH_BYTES, "big")
        + meta
        + len(payload).to_bytes(_LENGTH_BYTES, "big")
        + payload
    )
    await writer.drain()


async def write_relay_frame(writer: FrameWriter, frame: RelayFrame) -> None:
    """Write one frame's header and payload, then flush."""
    meta = _meta(frame)
    if len(meta) > MAX_META_BYTES or len(frame.payload) > MAX_PAYLOAD_BYTES:
        raise FrameStreamError("relay frame exceeds the stream frame bounds")
    await _write_framed(writer, meta, frame.payload)


async def write_probe(writer: FrameWriter, payload: bytes) -> None:
    """Write one probe, then flush."""
    if len(payload) > MAX_PROBE_BYTES:
        raise FrameStreamError("probe payload exceeds its bound")
    await _write_framed(writer, _PROBE_META, payload)


async def write_accept(writer: FrameWriter, status: AcceptStatus) -> None:
    """Write one connection accept step, then flush."""
    await _write_framed(writer, _ACCEPT_META, status.value.encode())


async def read_relay_frame(reader: asyncio.StreamReader) -> RelayFrame:
    """Read one relay frame, raising on a probe, an accept step, or broken framing."""
    frame = await read_stream_frame(reader)
    if not isinstance(frame, RelayFrame):
        raise FrameStreamError("expected a relay frame")
    return frame


async def read_stream_frame(
    reader: asyncio.StreamReader,
) -> RelayFrame | ProbeFrame | AcceptFrame:
    """Read one frame, probe or accept step, raising on broken framing or a bound."""
    meta_len = int.from_bytes(await reader.readexactly(_LENGTH_BYTES), "big")
    if meta_len > MAX_META_BYTES:
        raise FrameStreamError(f"relay frame header too large: {meta_len}")
    raw_meta = await reader.readexactly(meta_len)
    try:
        meta = json.loads(raw_meta)
        kind = meta["kind"]
    except (KeyError, ValueError, TypeError) as exc:
        raise FrameStreamError("undecodable relay frame header") from exc
    payload_len = int.from_bytes(await reader.readexactly(_LENGTH_BYTES), "big")
    if kind == _PROBE_KIND:
        if payload_len > MAX_PROBE_BYTES:
            raise FrameStreamError(f"probe payload too large: {payload_len}")
        return ProbeFrame(await reader.readexactly(payload_len))
    if kind == _ACCEPT_KIND:
        if payload_len > _MAX_ACCEPT_BYTES:
            raise FrameStreamError(f"accept step too large: {payload_len}")
        try:
            return AcceptFrame(
                AcceptStatus((await reader.readexactly(payload_len)).decode())
            )
        except ValueError as exc:
            raise FrameStreamError("undecodable accept step") from exc
    if payload_len > MAX_PAYLOAD_BYTES:
        raise FrameStreamError(f"relay frame payload too large: {payload_len}")
    payload = await reader.readexactly(payload_len) if payload_len else b""
    try:
        return RelayFrame(
            kind=RelayFrameKind(kind),
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
    "AcceptFrame",
    "AcceptStatus",
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
    "write_accept",
    "write_probe",
    "write_relay_frame",
]
