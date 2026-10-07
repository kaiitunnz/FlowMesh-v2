"""Reading a producer's result to fan it out."""

import time
from dataclasses import dataclass

from shared.content import ContentReference
from shared.schemas.result.binding import collection_elements
from shared.tasks.result_binding import ResultBinding

from ..results import ResultReader, ResultUnavailable, ResultUnreadable

# A spawn producer's fan-out read retries off the lock before the workflow fails.
_FANOUT_READ_ATTEMPTS = 3
_FANOUT_READ_BACKOFF_SEC = 0.2


@dataclass(frozen=True)
class FanoutRead:
    """How many elements a spawn producer's collection holds, or why it went unread."""

    count: int = 0
    # The stored result the count was read from; None for a producer that skipped.
    reference: ContentReference | None = None
    error: str | None = None
    # The store could not be reached; the collection is still there to read later.
    unavailable: bool = False


def read_fanout(
    results: ResultReader, task_id: str, binding: ResultBinding | None
) -> FanoutRead:
    """Count a producer's collection off the lock, retrying a store that is away."""
    if binding is not None and binding.skip is not None:
        return FanoutRead()
    if binding is None or binding.reference is None:
        return FanoutRead(error=f"fan-out producer {task_id} has no bound result")
    for attempt in range(_FANOUT_READ_ATTEMPTS):
        try:
            envelope = results.read(binding)
        except ResultUnreadable as exc:
            return FanoutRead(
                error=f"fan-out producer {task_id} result is unreadable: {exc}"
            )
        except ResultUnavailable as exc:
            if attempt + 1 == _FANOUT_READ_ATTEMPTS:
                return FanoutRead(error=str(exc), unavailable=True)
            time.sleep(_FANOUT_READ_BACKOFF_SEC)
            continue
        return FanoutRead(
            count=len(collection_elements(envelope)), reference=binding.reference
        )
    raise AssertionError("unreachable")
