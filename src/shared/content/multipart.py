"""The registration contract for multipart uploads to the shared store.

A writer registers each multipart upload's exact UploadId with control before it sends
any part, so control knows which uploads have a live owner and which were abandoned.
Control answers with a verdict: the upload is registered, it is doomed and the writer
starts another, or the writer owns nothing that may write in that scope.
"""

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from enum import StrEnum

from .store import ContentStoreError

# The content protocol a worker reports at registration: 1 registers every multipart
# upload before its first part.
CONTENT_PROTOCOL = 1

MULTIPART_ACK_FRAME_KIND = "content_multipart_ack"

type MultipartInitiated = Callable[[str, str, Callable[[], bool] | None], None]
"""Register one initiated upload by its object key and UploadId, given the write's
cancellation predicate if it has one; raises to refuse."""


class MultipartVerdict(StrEnum):
    """Control's answer to one upload's registration."""

    ACK = "ack"
    DOOMED = "doomed"
    """The UploadId is being aborted; the writer sends the object under a new one."""
    UNOWNED = "unowned"
    """Nothing the writer holds may own an upload in that scope."""


class MultipartDoomed(ContentStoreError):
    """Control is aborting this UploadId; the object goes up under a new one."""


class MultipartUnowned(ContentStoreError):
    """Control found no live owner for this upload, so the write is refused."""


_FINALIZATION: ContextVar[str | None] = ContextVar(
    "content_multipart_finalization", default=None
)


@contextlib.contextmanager
def finalizing(idempotency_key: str) -> Iterator[None]:
    """Name the outcome finalization the writes in this context materialize."""
    token = _FINALIZATION.set(idempotency_key)
    try:
        yield
    finally:
        _FINALIZATION.reset(token)


def current_finalization() -> str | None:
    """The outcome finalization the current write materializes, if any."""
    return _FINALIZATION.get()


__all__ = [
    "CONTENT_PROTOCOL",
    "MULTIPART_ACK_FRAME_KIND",
    "MultipartDoomed",
    "MultipartInitiated",
    "MultipartUnowned",
    "MultipartVerdict",
    "current_finalization",
    "finalizing",
]
