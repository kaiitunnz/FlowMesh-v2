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
KIND_CANCEL = "cancel"
# Sidecar -> deputy.
KIND_ACK = "ack"
KIND_CHUNK = "chunk"
KIND_DONE = "done"
KIND_REJECT = "reject"
KIND_FAILED = "failed"

split_host_port = netwire.split_host_port


async def write_msg(writer: asyncio.StreamWriter, kind: str, **fields: Any) -> None:
    """Write one JSON control/data frame."""
    await netwire.write_frame(writer, json.dumps({"kind": kind, **fields}).encode())


async def read_msg(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one JSON frame; a malformed or non-object frame is a protocol error."""
    payload = json.loads((await netwire.read_frame(reader)).decode())
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError("malformed resident wire frame")
    return payload
