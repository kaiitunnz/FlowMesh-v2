"""How a result binding reads as an envelope, and how a consumer projects one value.

Control reads through a binding under its own store authority; a worker hydrates the
same binding through its content plane. Both project with the functions here, so a value
reads the same wherever it is read.
"""

import json
from typing import Any

from shared.tasks.result_binding import ResultBinding

from ._base import BaseExecutorResult
from .catalog import ResultEnvelope


def skip_envelope(binding: ResultBinding) -> ResultEnvelope:
    """The envelope a task that settled without running reads as."""
    envelope = ResultEnvelope(
        task_id=binding.task_id, result=BaseExecutorResult(), metadata=binding.skip
    )
    if binding.settled_at is not None:
        envelope.received_at = binding.settled_at
    return envelope


def skip_envelope_bytes(binding: ResultBinding) -> bytes:
    """The stored form of a skipped task's envelope."""
    return skip_envelope(binding).model_dump_json(indent=2).encode("utf-8")


def result_collection(payload: dict[str, Any]) -> list[Any] | None:
    """A result's collection: its ``fanout`` list, else its ``items`` list."""
    collection = payload.get("fanout")
    if not isinstance(collection, list):
        collection = payload.get("items")
    return collection if isinstance(collection, list) else None


def collection_elements(envelope: ResultEnvelope) -> list[Any]:
    """A result's collection with each element's carried value unwrapped.

    An echo item is ``{output: v}``, so an element is its ``output`` when it has one,
    which is a valid child input. A result with no collection
    has no elements.
    """
    collection = result_collection(envelope.result.model_dump()) or []
    return [_unwrapped(item) for item in collection]


def collection_element(envelope: ResultEnvelope, index: int) -> Any:
    """One element of a result's collection; raises ``IndexError`` when it has none."""
    elements = collection_elements(envelope)
    if index < 0 or index >= len(elements):
        raise IndexError(
            f"task {envelope.task_id} has {len(elements)} collection elements and "
            f"none at {index}"
        )
    return elements[index]


def value_text(envelope: ResultEnvelope, element: int | None) -> str | None:
    """The string a consumer reads as one value of a result, or None when absent.

    An element is one collection member; otherwise the value is the result's ``value``,
    or the whole result when it declares none.
    """
    payload = envelope.result.model_dump()
    if element is not None:
        collection = result_collection(payload)
        if collection is None or element < 0 or element >= len(collection):
            return None
        return _stringify(_unwrapped(collection[element]))
    value = payload.get("value")
    return _stringify(value if value is not None else payload)


def _unwrapped(item: Any) -> Any:
    return item["output"] if isinstance(item, dict) and "output" in item else item


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


__all__ = [
    "collection_element",
    "collection_elements",
    "result_collection",
    "skip_envelope",
    "skip_envelope_bytes",
    "value_text",
]
