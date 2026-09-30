"""Locating, masking, and restoring the inline credentials a task spec carries.

A spec names the fields a submission may carry a credential in; within them a
credential is any value ``find_credential_values`` finds. Each is addressed by a JSON
pointer relative to the spec, so the spec can be masked where it is stored and restored
where it runs.
"""

from collections.abc import Collection, Iterable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from ..utils.redact import (
    CredentialPath,
    find_credential_values,
    is_credential_key,
    masked,
)
from .specs import TaskSpecTemplateBase


def credential_pointer(path: CredentialPath) -> str:
    return "".join(
        "/" + str(segment).replace("~", "~0").replace("/", "~1") for segment in path
    )


def _segments(pointer: str) -> list[str]:
    return [
        segment.replace("~1", "/").replace("~0", "~")
        for segment in pointer.split("/")[1:]
    ]


def find_spec_credentials(spec: TaskSpecTemplateBase) -> dict[str, Any]:
    """Every inline credential value in ``spec``, keyed by its pointer."""
    found: dict[str, Any] = {}
    for name in type(spec).credential_fields:
        value = getattr(spec, name)
        if value is None:
            continue
        data = to_jsonable_python(value)
        if is_credential_key(name):
            found[credential_pointer((name,))] = data
            continue
        for path, credential in find_credential_values(data, (name,)):
            found[credential_pointer(path)] = credential
    return found


def find_spec_strings(
    spec: TaskSpecTemplateBase, values: Collection[str]
) -> frozenset[str]:
    """The pointers of the strings in ``spec``'s credential fields equal to one of
    ``values``."""
    if not values:
        return frozenset()
    return frozenset(
        credential_pointer(path)
        for name in type(spec).credential_fields
        if (value := getattr(spec, name)) is not None
        for path, text in _strings(to_jsonable_python(value), (name,))
        if text in values
    )


def _strings(value: Any, path: CredentialPath) -> Iterator[tuple[CredentialPath, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*path, index))


def spec_value(spec: BaseModel, pointer: str) -> Any:
    """The value at ``pointer`` in ``spec``."""
    container: Any = spec
    for segment in _segments(pointer):
        container = _child(container, segment)
    return container


def dispatched_credentials(spec: BaseModel, pointers: Iterable[str]) -> list[Any]:
    """The credential values at ``pointers`` in a dispatched spec; a pointer the spec
    does not hold is skipped."""
    values: list[Any] = []
    for pointer in pointers:
        try:
            values.append(spec_value(spec, pointer))
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            continue
    return values


def set_spec_values(spec: BaseModel, values: Mapping[str, Any]) -> None:
    """Write each value at its pointer in ``spec``, in place."""
    for pointer, value in values.items():
        *parents, last = _segments(pointer)
        container: Any = spec
        for segment in parents:
            container = _child(container, segment)
        _assign(container, last, value)


def mask_spec_values(spec: BaseModel, pointers: Mapping[str, Any]) -> None:
    """Replace each value ``pointers`` names with its credential marker, in place."""
    set_spec_values(
        spec, {pointer: masked(value) for pointer, value in pointers.items()}
    )


def _pointer_key(key: Any) -> str:
    return next(iter(to_jsonable_python({key: None})))


def _key(container: Mapping[Any, Any], segment: str) -> Any:
    """The mapping key a pointer segment names; a pointer spells every key in its JSON
    form, so ``True`` is ``true``."""
    if segment in container:
        return segment
    return next((key for key in container if _pointer_key(key) == segment), segment)


def _child(container: Any, segment: str) -> Any:
    if isinstance(container, BaseModel):
        return getattr(container, segment)
    if isinstance(container, list):
        return container[int(segment)]
    return container[_key(container, segment)]


def _assign(container: Any, segment: str, value: Any) -> None:
    if isinstance(container, BaseModel):
        setattr(container, segment, value)
    elif isinstance(container, list):
        container[int(segment)] = value
    else:
        container[_key(container, segment)] = value


__all__ = [
    "credential_pointer",
    "dispatched_credentials",
    "find_spec_credentials",
    "find_spec_strings",
    "mask_spec_values",
    "set_spec_values",
    "spec_value",
]
