"""The message kinds and request digest of the resident two-phase protocol.

Two-phase between the origin worker and the replica sidecar: a bootstrap delivers the
claim-bound handoff and the request and receives an enqueue acknowledgement; then, under
the post-``ACCEPTED`` route authorization, the sidecar streams the response and honors
cancellation. Each message rides a relay frame's opaque payload in the shared network
message framing; a task-addressed serve message additionally carries raw HTTP body bytes
after its header.
"""

import hashlib

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


def resident_request_digest(request_payload: str) -> str:
    """A canonical digest of a captured resident request.

    It marks the boundary as worker-originated so the raw request stays worker-private
    behind it; the target-side fence is the claim incarnation, not this digest.
    """
    return hashlib.sha256(request_payload.encode()).hexdigest()
