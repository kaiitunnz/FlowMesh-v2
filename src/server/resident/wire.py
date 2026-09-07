"""Socket framing for the native resident invocation over a dialed connection.

The message bodies are the shared resident wire protocol; this adds the length-framed
socket read/write the forward-dial transports use. A relay hop stays byte-transparent,
so the frames are end-to-end between the origin and the sidecar.
"""

import asyncio
from typing import Any

from shared.resident.wire import (
    KIND_ACK,
    KIND_BOOTSTRAP,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_REJECT,
    KIND_STREAM,
    decode_msg,
    encode_msg,
)

from ..network import wire as netwire

split_host_port = netwire.split_host_port

__all__ = [
    "KIND_ACK",
    "KIND_BOOTSTRAP",
    "KIND_CHUNK",
    "KIND_DONE",
    "KIND_FAILED",
    "KIND_REJECT",
    "KIND_STREAM",
    "decode_msg",
    "encode_msg",
    "read_msg",
    "split_host_port",
    "write_msg",
]


async def write_msg(writer: asyncio.StreamWriter, kind: str, **fields: Any) -> None:
    """Write one JSON control/data frame."""
    await netwire.write_frame(writer, encode_msg(kind, **fields))


async def read_msg(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one JSON frame; a malformed or non-object frame is a protocol error."""
    return decode_msg(await netwire.read_frame(reader))
