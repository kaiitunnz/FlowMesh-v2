"""Why control did not bind content a producer stored, and what the producer does next.

One vocabulary answers every report naming stored objects: a worker's publication, a
staged upload, and an outcome finalization.
"""

from enum import StrEnum


class StoreRefusal(StrEnum):
    """Why control did not bind reported content now."""

    RETRY = "retry"
    """Control could not decide now; the producer reports again later."""
    BUSY = "busy"
    """The lane committing reports is full and holds the next free place for this one;
    the producer reports again shortly."""
    RETRY_RETIREMENT = "retry_retirement"
    """A named object is being removed from the store; the producer waits, storing
    nothing, and reports again."""
    REMATERIALIZE = "rematerialize"
    """A named object is not in the store; the producer stores it again from its copy,
    then reports again."""


__all__ = ["StoreRefusal"]
