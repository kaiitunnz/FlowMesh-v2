"""Framed protocol for the feature-gated network-plane echo seam.

A request is a length-prefixed payload; a response is a status byte plus, on success,
the echoed payload. The frame is end-to-end between the origin deputy and the target
sidecar, so a relay hop stays byte-transparent. This is a test capability — it never
carries a resident request.
"""

import asyncio

STATUS_OK = b"\x00"
STATUS_APP_ERROR = b"\x01"

# A payload the sidecar answers with an application error rather than an echo, to reach
# the non-demoting outcome path (the connection succeeds; the application does not).
APP_ERROR_SENTINEL = b"__APP_ERROR__"

_HEADER_BYTES = 4
_MAX_FRAME_BYTES = 1 << 20


async def write_frame(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(len(data).to_bytes(_HEADER_BYTES, "big") + data)
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(_HEADER_BYTES)
    size = int.from_bytes(header, "big")
    if size > _MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {size}")
    return await reader.readexactly(size)
