"""The message kinds one content hydration transfer exchanges.

A requester opens the transfer by presenting its grant; the holder either refuses with a
typed reason or answers with the object's size, its bytes in windowed chunks, and a
terminal. The bytes ride a relay frame's opaque payload, so whatever carries the
transfer moves them without reading them, and the requester verifies the assembled
object against its reference before anything uses it.
"""

# Requester -> holder.
KIND_FETCH = "content_fetch"
# Holder -> requester.
KIND_HEAD = "content_head"
KIND_CHUNK = "content_chunk"
KIND_DONE = "content_done"
KIND_REJECT = "content_reject"

# How much of an object rides one frame. The session's window bounds what is in flight;
# this bounds one frame so a large object streams rather than arriving as one payload.
CHUNK_BYTES = 64 * 1024

__all__ = [
    "CHUNK_BYTES",
    "KIND_CHUNK",
    "KIND_DONE",
    "KIND_FETCH",
    "KIND_HEAD",
    "KIND_REJECT",
]
