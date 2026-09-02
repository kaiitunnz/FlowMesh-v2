"""Framed protocol for native resident invocation over the data-direct channel.

The origin deputy and the resident-facing sidecar speak this over a connection the
network plane carries (``worker_direct``, or through a target-addressed relay hop). It
is two-phase: a bootstrap delivers the claim-bound handoff and the request and receives
an enqueue acknowledgement; then, under the post-``ACCEPTED`` route authorization, the
sidecar streams the response and honors cancellation. A relay hop stays
byte-transparent, so the frames are end-to-end between deputy and sidecar.
"""

import asyncio
import json
from typing import Any

from ..network import wire as netwire

# Deputy -> sidecar.
KIND_BOOTSTRAP = "bootstrap"
KIND_STREAM = "stream"
# Sidecar -> deputy.
KIND_ACK = "ack"
KIND_CHUNK = "chunk"
KIND_DONE = "done"
KIND_REJECT = "reject"
KIND_FAILED = "failed"

split_host_port = netwire.split_host_port


def encode_msg(kind: str, **fields: Any) -> bytes:
    """Serialize one control/data message body (no length framing).

    The same body rides a length-framed socket via ``write_msg`` or a reverse-relay
    frame payload verbatim, so the sidecar validates and serves it identically.
    """
    return json.dumps({"kind": kind, **fields}).encode()


def decode_msg(raw: bytes) -> dict[str, Any]:
    """Parse one message body; a malformed or non-object body is a protocol error."""
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError("malformed resident wire frame")
    return payload


async def write_msg(writer: asyncio.StreamWriter, kind: str, **fields: Any) -> None:
    """Write one JSON control/data frame."""
    await netwire.write_frame(writer, encode_msg(kind, **fields))


async def read_msg(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one JSON frame; a malformed or non-object frame is a protocol error."""
    return decode_msg(await netwire.read_frame(reader))
