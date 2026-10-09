"""How a result binding reads as an envelope, and how a consumer projects one value.

Control reads through a binding under its own store authority; a worker hydrates the
same binding through its content plane. Both project with the functions here, so a value
reads the same wherever it is read.
"""

import json
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.tasks.result_binding import BindingKind, ResultBinding, ResultElementRef

from ._base import BaseExecutorResult
from .catalog import ResultEnvelope
from .routed import RoutedValue

# The outcome of an aggregate member that carries a value.
_SUCCESS = "success"


def skip_envelope(binding: ResultBinding) -> ResultEnvelope:
    """The envelope a task that settled without running reads as."""
    envelope = ResultEnvelope(
        task_id=binding.task_id or "",
        result=BaseExecutorResult(),
        metadata=binding.skip,
    )
    if binding.settled_at is not None:
        envelope.received_at = binding.settled_at
    return envelope


def skip_envelope_bytes(binding: ResultBinding) -> bytes:
    """The stored form of a skipped task's envelope."""
    return skip_envelope(binding).model_dump_json(indent=2).encode("utf-8")


class NotAResultEnvelope(ValueError):
    """Stored bytes that do not parse as a result envelope."""


def result_envelope(data: bytes, source: str) -> ResultEnvelope:
    """The result envelope stored bytes hold, naming ``source`` when they hold none."""
    try:
        return ResultEnvelope.model_validate_json(data)
    except ValidationError as exc:
        raise NotAResultEnvelope(f"{source} is not a result envelope: {exc}") from exc


def _result_collection(payload: dict[str, Any]) -> list[Any] | None:
    """A result's collection: its ``fanout`` list, else its ``items`` list."""
    collection = payload.get("fanout")
    if not isinstance(collection, list):
        collection = payload.get("items")
    return collection if isinstance(collection, list) else None


def collection_elements(envelope: ResultEnvelope) -> list[Any]:
    """A result's collection with each element's carried value unwrapped.

    An echo item is ``{output: v}``, so an element is its ``output`` when it has one. A
    result with no collection has no elements.
    """
    collection = _result_collection(envelope.result.model_dump()) or []
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


def dig(value: Any, steps: Sequence[str | int]) -> Any:
    """Walk ``steps`` into a value: a field of a mapping or model, or an index of a
    list. A step that finds nothing yields None."""
    current = value
    for step in steps:
        if isinstance(current, RoutedValue):
            current = current.routed_value
        match current:
            case dict():
                current = current.get(str(step))
            case list() if isinstance(step, int) or str(step).isdigit():
                index = int(step)
                current = current[index] if 0 <= index < len(current) else None
            case BaseModel():
                current = getattr(current, str(step), None)
            case _:
                return None
        if current is None:
            return None
    return current


def element_value(envelope: ResultEnvelope, ref: ResultElementRef) -> Any:
    """The value one element of a result reads as; raises ``IndexError`` when the
    result holds none there."""
    start = (
        collection_element(envelope, ref.element)
        if ref.element is not None
        else envelope.result
    )
    if not ref.path:
        return start
    if (value := dig(start, ref.path)) is None:
        raise IndexError(
            f"task {envelope.task_id} holds no value at {list(ref.path)!r}"
        )
    return value


def scoped_value(
    binding: ResultBinding, envelope_of: Callable[[ResultBinding], ResultEnvelope]
) -> Any:
    """The value an input binding reads as, reading each result through
    ``envelope_of``; raises ``IndexError`` when a result holds no element it selects.

    A whole result reads as the result itself. A part of one, an aggregate's members
    ``{key, outcome, value}`` (``value`` null unless the member succeeded), a bundle's
    named values, a literal, and an explicit empty (null) read as plain values, and
    ``path`` reads into whichever the rest names.
    """
    match binding.kind:
        case BindingKind.RESULT:
            envelope = envelope_of(binding)
            value: Any = (
                collection_element(envelope, binding.element)
                if binding.element is not None
                else envelope.result
            )
        case BindingKind.MEMBERS:
            value = [
                {
                    "key": member.key,
                    "outcome": member.outcome,
                    "value": (
                        _member_value(member.binding, envelope_of)
                        if member.outcome == _SUCCESS
                        else None
                    ),
                }
                for member in binding.members
            ]
        case BindingKind.BUNDLE:
            value = {
                member.key: _member_value(member.binding, envelope_of)
                for member in binding.members
            }
        case BindingKind.LITERAL:
            value = binding.literal
        case BindingKind.EMPTY:
            value = None
    return dig(value, binding.path) if binding.path else value


def _member_value(
    binding: ResultBinding | None,
    envelope_of: Callable[[ResultBinding], ResultEnvelope],
) -> Any:
    if binding is None:
        return None
    value = scoped_value(binding, envelope_of)
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def value_text(envelope: ResultEnvelope, element: int | None) -> str | None:
    """The string a consumer reads as one value of a result, or None when absent.

    An element is one collection member; otherwise the value is the result's ``value``,
    or the whole result when it declares none.
    """
    if element is not None:
        try:
            return _stringify(collection_element(envelope, element))
        except IndexError:
            return None
    payload = envelope.result.model_dump()
    value = payload.get("value")
    return _stringify(value if value is not None else payload)


def _unwrapped(item: Any) -> Any:
    return item["output"] if isinstance(item, dict) and "output" in item else item


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


__all__ = [
    "NotAResultEnvelope",
    "collection_element",
    "collection_elements",
    "result_envelope",
    "scoped_value",
    "skip_envelope",
    "skip_envelope_bytes",
    "value_text",
]
