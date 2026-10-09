"""Reading the stored values that route a branch or fan out a spawn.

A branch's selector and a spawn's collection are read off the runtime lock from the
result envelope of the task whose value reaches them, and applied under it only while
the occurrence still waits on them.
"""

from dataclasses import dataclass
from typing import Any

from shared.schemas.result.binding import (
    collection_element,
    collection_elements,
    dig,
)
from shared.tasks.result_binding import ResultBinding

from ...orchestration.state import ValueRef
from ..results import ResultReader, ResultUnavailable, ResultUnreadable


@dataclass(frozen=True)
class ControlRead:
    """A value read for a control occurrence, or why it went unread."""

    value: Any = None
    # The elements of the value's collection, when it is a whole result.
    elements: int | None = None
    error: str | None = None
    # The store could not be reached; the value is still there to read later.
    unavailable: bool = False


def read_control_value(
    results: ResultReader, value_ref: ValueRef, binding: ResultBinding | None
) -> ControlRead:
    """Read the value a control occurrence routes or fans out over, off the lock.

    The value is the producer's result, one element of its collection, or a part of
    either a projection selects. A skipped producer delivers no value.
    """
    if value_ref.kind == "empty" or (binding is not None and binding.skip is not None):
        return ControlRead(elements=0)
    if value_ref.kind != "legacy_task_result":
        return ControlRead(error=f"a {value_ref.kind} value is not a task result")
    if binding is None or binding.reference is None:
        return ControlRead(
            error=f"task {value_ref.legacy_task_id} has no bound result to read"
        )
    try:
        envelope = results.read(binding)
    except ResultUnreadable as exc:
        return ControlRead(error=f"the result is unreadable: {exc}")
    except ResultUnavailable as exc:
        # The re-drive scheduler reads it again later.
        return ControlRead(error=str(exc), unavailable=True)
    if value_ref.collection_key is not None:
        try:
            start = collection_element(envelope, int(value_ref.collection_key))
        except IndexError as exc:
            return ControlRead(error=str(exc))
    elif not value_ref.projection:
        return ControlRead(
            value=envelope.result,
            elements=len(collection_elements(envelope)),
        )
    else:
        start = envelope.result
    value = dig(start, value_ref.projection)
    return ControlRead(
        value=value, elements=len(value) if isinstance(value, list) else None
    )
