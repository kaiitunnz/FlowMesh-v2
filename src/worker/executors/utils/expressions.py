"""Projecting a value out of an upstream-result context by a dotted path.

One walker serves every consumer of an upstream projection. An expression names an
upstream input and then indexes into its value, a whole result or a value routed to the
task: each dotted token reads an attribute or key and may carry bracket indexes, and a
token applied to a list distributes over its elements. ``frames`` admits the
table-shaped steps; a consumer whose contract covers only scalar and structured values
leaves them out, so a table reaches it as an unprojectable input rather than as a
silently different value.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel

from shared.schemas.result import BaseExecutorResult
from shared.schemas.result.routed import routed_root

from ...utils.serialization import try_deserialize_dataframe
from ..base_executor import ExecutionError

_SENTINEL: Any = object()


type ReadPaths = tuple[str | int, ...] | list[ReadPaths]
"""Where a projected value was read from inside its upstream input's value: one path
for a value read whole, or one per element of a list a token distributed over."""


def project_expression(
    expr: str, context: dict[str, BaseExecutorResult], *, frames: bool = True
) -> Any:
    """Resolve ``expr`` against an upstream-result context."""
    return project_expression_paths(expr, context, frames=frames)[0]


def project_expression_paths(
    expr: str, context: dict[str, BaseExecutorResult], *, frames: bool = True
) -> tuple[Any, ReadPaths]:
    """Resolve ``expr`` against an upstream-result context, with the read paths to
    the value it projects."""
    if not expr:
        return None, ()

    parts = expr.split(".")
    root = parts[0]
    result = context.get(root)
    if result is None:
        return None, ()

    value: Any = routed_root(result)
    paths: ReadPaths = ()
    for token in parts[1:]:
        if not token:
            continue
        attr, indexes = split_indexes(token)
        if attr:
            value, distributed = _read_attr(value, attr, token, parts, frames=frames)
            paths = (
                [(*_as_path(element_paths(paths, i)), attr) for i in range(len(value))]
                if distributed
                else (*_as_path(paths), attr)
            )
        for idx in indexes:
            value, paths = _read_index(value, paths, idx, token)
        if frames:
            value = _maybe_deserialize_frames(value)
    return value, paths


def element_paths(paths: ReadPaths, index: int) -> ReadPaths:
    """The read paths to element ``index`` of a projected list."""
    return paths[index] if isinstance(paths, list) else (*paths, index)


def item_path(paths: ReadPaths, index: int) -> tuple[str | int, ...]:
    """The read path to item ``index`` of a projected list, empty for an item that is
    itself a list distributed over."""
    return _as_path(element_paths(paths, index))


def _as_path(paths: ReadPaths) -> tuple[str | int, ...]:
    return paths if isinstance(paths, tuple) else ()


def _read_attr(
    value: Any, attr: str, token: str, parts: list[str], *, frames: bool
) -> tuple[Any, bool]:
    """The value ``attr`` reads, and whether it distributed over a list."""
    if isinstance(value, dict) and attr in value:
        return value[attr], False
    if isinstance(value, list) and all(
        isinstance(v, dict) and attr in v for v in value
    ):
        return [v[attr] for v in value], True
    if (
        isinstance(value, list)
        and value
        and all(isinstance(v, BaseModel) for v in value)
    ):
        plucked = [getattr(v, attr, _SENTINEL) for v in value]
        if any(item is _SENTINEL for item in plucked):
            raise ExecutionError(
                f"{attr} not a valid attribute of {type(value[0]).__name__} "
                f"for {token}."
            )
        return plucked, True
    if (
        frames
        and isinstance(value, list)
        and all(isinstance(v, pd.DataFrame) for v in value)
    ):
        if any(attr not in v.columns for v in value):
            raise ExecutionError(
                f"{attr} not a valid column in one of the DataFrames for {token}."
            )
        return [v[attr].tolist() for v in value], True
    if frames and isinstance(value, pd.DataFrame):
        if attr not in value.columns:
            raise ExecutionError(f"{attr} not a valid column in DataFrame for {token}.")
        return value[attr].tolist(), False
    if isinstance(value, BaseModel):
        resolved = getattr(value, attr, _SENTINEL)
        if resolved is _SENTINEL:
            raise ExecutionError(
                f"{attr} not a valid attribute of {type(value).__name__} for {token}."
            )
        return resolved, False
    raise ExecutionError(
        f"{attr} in {parts} is not a valid key - {type(value).__name__}, {value}"
    )


def _read_index(
    value: Any, paths: ReadPaths, idx: int, token: str
) -> tuple[Any, ReadPaths]:
    if isinstance(value, list) and -len(value) <= idx < len(value):
        index = idx % len(value)
        return value[index], element_paths(paths, index)
    if isinstance(value, list) and all(isinstance(v, list) for v in value):
        picked: list[ReadPaths] = []
        for i, v in enumerate(value):
            if not -len(v) <= idx < len(v):
                raise ExecutionError(f"{idx} not a valid index in {token} - {len(v)}")
            picked.append(element_paths(element_paths(paths, i), idx % len(v)))
        return [v[idx] for v in value], picked
    raise ExecutionError(f"{idx} not a valid index in {token} - {len(value)}")


def _maybe_deserialize_frames(value: Any) -> Any:
    if isinstance(value, dict):
        return try_deserialize_dataframe(value)
    if isinstance(value, list) and all(isinstance(v, dict) for v in value):
        return [try_deserialize_dataframe(v) for v in value]
    return value


def split_indexes(token: str) -> tuple[str, list[int]]:
    """An expression token split into its attribute and its bracket indexes."""
    parts = token.split("[")
    attr = parts[0]
    idx_list: list[int] = []
    for part in parts[1:]:
        part = part.rstrip("]")
        if part:
            try:
                idx_list.append(int(part))
            except ValueError:
                idx_list.append(-1)
    return attr, idx_list
