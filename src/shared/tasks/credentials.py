"""Locating, masking, and restoring the inline credentials a task spec carries.

A spec names the fields a submission may carry a credential in; within them a
credential is any value ``find_credential_values`` finds. Each is addressed by a JSON
pointer relative to the spec, so the spec can be masked where it is stored and restored
where it runs.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from ..utils.redact import (
    CredentialPath,
    find_credential_values,
    is_credential_key,
    masked,
)
from .placeholders import PLACEHOLDER_PATTERN
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


def spec_value(spec: BaseModel, pointer: str) -> Any:
    """The value at ``pointer`` in ``spec``."""
    container: Any = spec
    for segment in _segments(pointer):
        container = _child(container, segment)
    return container


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


def holds_placeholder(value: Any) -> bool:
    """Whether a credential value renders from an upstream stage at dispatch."""
    if isinstance(value, str):
        return bool(PLACEHOLDER_PATTERN.search(value))
    if isinstance(value, dict):
        return any(holds_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(holds_placeholder(item) for item in value)
    return False


def _key(container: Mapping[Any, Any], segment: str) -> Any:
    """The mapping key a pointer segment names; a pointer spells every key as text."""
    if segment in container:
        return segment
    return next((key for key in container if str(key) == segment), segment)


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
    "find_spec_credentials",
    "holds_placeholder",
    "mask_spec_values",
    "set_spec_values",
    "spec_value",
]
