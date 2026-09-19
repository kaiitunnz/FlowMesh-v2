"""Message bodies carried inside a relay frame's opaque payload.

A framed session's payload is one self-describing JSON object, so the servers relay it
without decoding and the receiving endpoint validates it identically on any transport. A
message that also carries raw bytes — an HTTP body, a content chunk — puts the JSON
header on the first line and appends the bytes after a single newline, so an arbitrary
binary body rides the same payload without a second encoding. A compact JSON object
never contains a literal newline (one inside a string is escaped as two characters), so
the first newline delimits the header unambiguously.
"""

import json
from typing import Any


def encode_msg(kind: str, **fields: Any) -> bytes:
    """Serialize one control/data message body (no length framing)."""
    return json.dumps({"kind": kind, **fields}).encode()


def decode_msg(raw: bytes) -> dict[str, Any]:
    """Parse one message body; a malformed or non-object body is a protocol error."""
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError("malformed relay message")
    return payload


def encode_body_msg(kind: str, body: bytes, **fields: Any) -> bytes:
    """Serialize one message whose raw body follows its JSON header line."""
    return json.dumps({"kind": kind, **fields}).encode() + b"\n" + body


def decode_body_msg(raw: bytes) -> tuple[dict[str, Any], bytes]:
    """Parse one message and its trailing raw body, which is empty when it carries none.

    It reads both framings, so one receive path handles body-carrying and header-only
    messages alike.
    """
    header, _, body = raw.partition(b"\n")
    payload = json.loads(header.decode())
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError("malformed relay message")
    return payload, body


__all__ = ["decode_body_msg", "decode_msg", "encode_body_msg", "encode_msg"]
