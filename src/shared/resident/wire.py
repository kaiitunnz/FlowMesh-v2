"""Framed protocol bodies for a resident invocation.

Two-phase between the origin worker and the replica sidecar: a bootstrap delivers the
claim-bound handoff and the request and receives an enqueue acknowledgement; then, under
the post-``ACCEPTED`` route authorization, the sidecar streams the response and honors
cancellation. Each message body is a self-describing JSON object carried verbatim as an
opaque relay payload, so the servers relay it without decoding and the sidecar validates
and serves it identically on either transport.

A task-addressed serve message additionally carries raw HTTP body bytes, which no JSON
field can hold unchanged. Those messages put the JSON header on the first line and
append the body after a single newline, so an arbitrary binary body rides the same
opaque payload without a second encoding. A compact JSON object never contains a
literal newline — one inside a string is escaped as two characters — so the first
newline delimits the header unambiguously.
"""

import hashlib
import json
from typing import Any

# Origin -> replica.
KIND_BOOTSTRAP = "bootstrap"
KIND_STREAM = "stream"
# Replica -> origin.
KIND_ACK = "ack"
KIND_HEAD = "head"
KIND_CHUNK = "chunk"
KIND_DONE = "done"
KIND_REJECT = "reject"
KIND_FAILED = "failed"


def encode_msg(kind: str, **fields: Any) -> bytes:
    """Serialize one control/data message body (no length framing)."""
    return json.dumps({"kind": kind, **fields}).encode()


def decode_msg(raw: bytes) -> dict[str, Any]:
    """Parse one message body; a malformed or non-object body is a protocol error."""
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError("malformed resident wire frame")
    return payload


def resident_request_digest(request_payload: str) -> str:
    """A canonical digest of a captured resident request.

    It marks the boundary as worker-originated so the raw request stays worker-private
    behind it; the target-side fence is the claim incarnation, not this digest.
    """
    return hashlib.sha256(request_payload.encode()).hexdigest()


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
        raise ValueError("malformed resident wire frame")
    return payload, body
