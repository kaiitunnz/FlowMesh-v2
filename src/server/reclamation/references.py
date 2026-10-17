"""Typed extraction of the content references a persisted model names.

Which fields of a model can hold a ``ContentReference`` is read from its annotations
once per class, so a walk visits only those fields and never interprets opaque values
or a field that merely looks like a digest. ``reference_paths`` reports the same
decision as dotted paths, the inventory a test pins every reference-bearing field
against.
"""

import threading
import types
import typing
from collections.abc import Iterable, Iterator, Mapping
from functools import cache
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from shared.content.reference import ContentReference

# The marker a path segment takes for every element of a sequence or mapping value.
ELEMENT = "[]"


def _unwrap(annotation: Any) -> Any:
    """Strip ``Annotated`` metadata and resolve a ``type`` alias to what it names."""
    while True:
        if isinstance(annotation, typing.TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is typing.Annotated:
            annotation = get_args(annotation)[0]
        else:
            return annotation


def _members(annotation: Any) -> list[tuple[str, Any]]:
    """The annotations a value of ``annotation`` may hold references through, each
    with the path segment it is reached by ("" for the value itself)."""
    annotation = _unwrap(annotation)
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return [("", arg) for arg in get_args(annotation) if arg is not type(None)]
    if origin is not None and isinstance(origin, type):
        args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
        if issubclass(origin, Mapping):
            return [(ELEMENT, args[1])] if len(args) == 2 else []
        if issubclass(origin, Iterable) and not issubclass(origin, (str, bytes)):
            return [(ELEMENT, arg) for arg in args]
    return []


_BEARING: dict[type[BaseModel], bool] = {}
_BEARING_LOCK = threading.Lock()


def _scan(annotation: Any, models: set[type[BaseModel]]) -> bool:
    """Whether ``annotation`` holds a reference other than through a model, adding
    each model it holds a value of to ``models``."""
    annotation = _unwrap(annotation)
    if isinstance(annotation, type):
        if issubclass(annotation, ContentReference):
            return True
        if issubclass(annotation, BaseModel):
            models.add(annotation)
            return False
    direct = False
    for _, arg in _members(annotation):
        direct = _scan(arg, models) or direct
    return direct


def _decide(root: type[BaseModel]) -> dict[type[BaseModel], bool]:
    """Decide every model reachable from ``root`` and not decided yet: a model bears
    references when one is reachable from its fields, through any cycle."""
    direct: dict[type[BaseModel], bool] = {}
    nested: dict[type[BaseModel], set[type[BaseModel]]] = {}
    pending = [root]
    while pending:
        model = pending.pop()
        if model in nested or model in _BEARING:
            continue
        reached: set[type[BaseModel]] = set()
        direct[model] = False
        for info in model.model_fields.values():
            direct[model] = _scan(info.annotation, reached) or direct[model]
        nested[model] = reached
        pending.extend(reached)
    bearing = {
        model
        for model, reached in nested.items()
        if direct[model] or any(_BEARING.get(other, False) for other in reached)
    }
    grew = True
    while grew:
        grew = False
        for model, reached in nested.items():
            if model not in bearing and not bearing.isdisjoint(reached):
                bearing.add(model)
                grew = True
    return {model: model in bearing for model in nested}


def _model_bears(model: type[BaseModel]) -> bool:
    """Whether any field of ``model`` can hold a reference."""
    if (known := _BEARING.get(model)) is not None:
        return known
    with _BEARING_LOCK:
        if model not in _BEARING:
            _BEARING.update(_decide(model))
        return _BEARING[model]


def _bears(annotation: Any) -> bool:
    models: set[type[BaseModel]] = set()
    return _scan(annotation, models) or any(_model_bears(m) for m in models)


@cache
def _model_fields(model: type[BaseModel]) -> tuple[str, ...]:
    """The fields of ``model`` whose annotation can hold a reference."""
    return tuple(
        name for name, info in model.model_fields.items() if _bears(info.annotation)
    )


def _walk(value: Any) -> Iterator[ContentReference]:
    if value is None:
        return
    if isinstance(value, ContentReference):
        yield value
    elif isinstance(value, BaseModel):
        for name in _model_fields(type(value)):
            yield from _walk(getattr(value, name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk(item)


def distinct(references: Iterable[ContentReference]) -> tuple[ContentReference, ...]:
    """The references, one per content identity, in first-seen order."""
    return tuple({reference.identity: reference for reference in references}.values())


def references_of(*values: Any) -> Iterator[ContentReference]:
    """Every reference the values name through their reference-bearing fields."""
    for value in values:
        yield from _walk(value)


def reference_paths(model: type[BaseModel]) -> frozenset[str]:
    """The dotted paths of ``model`` that can reach a reference, ``[]`` marking every
    element of a sequence or mapping; a path through a model refers back to it at its
    first recursion."""
    paths: set[str] = set()

    def visit(annotation: Any, prefix: str, seen: frozenset[type]) -> None:
        annotation = _unwrap(annotation)
        if isinstance(annotation, type) and issubclass(annotation, ContentReference):
            paths.add(prefix)
            return
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in seen:
                paths.add(f"{prefix}@{annotation.__name__}")
                return
            for name in _model_fields(annotation):
                info = annotation.model_fields[name]
                visit(
                    info.annotation,
                    f"{prefix}.{name}" if prefix else name,
                    seen | {annotation},
                )
            return
        for segment, arg in _members(annotation):
            visit(arg, f"{prefix}{segment}", seen)

    visit(model, "", frozenset())
    return frozenset(paths)


__all__ = ["ELEMENT", "distinct", "reference_paths", "references_of"]
