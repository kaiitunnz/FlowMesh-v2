"""Framed protocol for remote external-tool carriage between the origin and a sidecar.

The in-server origin and the worker sidecar speak this over a connection the network
plane carries (``worker_direct`` / ``node_relay``, or through a target-addressed relay
hop, or as opaque payloads over the reverse-rendezvous relay). It is a single request /
reply: the origin sends one operation frame carrying the operation fence and the bounded
request; the sidecar validates the fence, egresses within it, and returns one result
frame — or a reject frame when the fence fails, before any provider call. A relay hop
stays byte-transparent, so the frames are end-to-end between origin and sidecar. It is
claim-free: no admission, credit, or resident concept rides it.
"""

import asyncio
import json
from typing import Any

from ..network import wire as netwire

# Origin -> sidecar.
KIND_OPERATION = "operation"
# Sidecar -> origin.
KIND_RESULT = "result"
KIND_REJECT = "reject"

split_host_port = netwire.split_host_port


def encode_msg(kind: str, **fields: Any) -> bytes:
    """Serialize one message body; it rides a socket frame or relay payload verbatim."""
    return json.dumps({"kind": kind, **fields}).encode()


def decode_msg(raw: bytes) -> dict[str, Any]:
    """Parse one message body; a malformed or non-object body is a protocol error."""
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError("malformed tool sidecar wire frame")
    return payload


async def write_msg(writer: asyncio.StreamWriter, kind: str, **fields: Any) -> None:
    await netwire.write_frame(writer, encode_msg(kind, **fields))


async def read_msg(reader: asyncio.StreamReader) -> dict[str, Any]:
    return decode_msg(await netwire.read_frame(reader))
