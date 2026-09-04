"""The message body codec for one remote external-tool operation.

The origin sends one operation frame carrying the operation fence and the bounded
request; the worker executor validates the fence, egresses within it, and returns one
result frame — or a reject frame when the fence fails, before any provider call. Only
the JSON body is defined here; the framing that carries it over a socket is elsewhere.
"""

import json
from typing import Any

# Origin -> executor.
KIND_OPERATION = "operation"
# Executor -> origin.
KIND_RESULT = "result"
KIND_REJECT = "reject"


def encode_msg(kind: str, **fields: Any) -> bytes:
    """Serialize one message body; it rides a socket frame or relay payload verbatim."""
    return json.dumps({"kind": kind, **fields}).encode()


def decode_msg(raw: bytes) -> dict[str, Any]:
    """Parse one message body; a malformed or non-object body is a protocol error."""
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError("malformed tool sidecar wire frame")
    return payload


__all__ = ["KIND_OPERATION", "KIND_REJECT", "KIND_RESULT", "decode_msg", "encode_msg"]
