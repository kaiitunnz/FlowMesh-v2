"""Projecting a value out of an upstream-result context by a dotted path.

One walker serves every consumer of an upstream projection. An expression names a root
upstream node and then indexes into its result: each dotted token reads an attribute or
key and may carry bracket indexes, and a token applied to a list distributes over its
elements. ``frames`` admits the table-shaped steps; a consumer whose contract covers
only scalar and structured values leaves them out, so a table reaches it as an
unprojectable input rather than as a silently different value.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel

from shared.schemas.result import BaseExecutorResult

from ...utils.serialization import try_deserialize_dataframe
from ..base_executor import ExecutionError

_SENTINEL: Any = object()


def project_expression(
    expr: str, context: dict[str, BaseExecutorResult], *, frames: bool = True
) -> Any:
    """Resolve ``expr`` against an upstream-result context."""
    if not expr:
        return None

    parts = expr.split(".")
    root = parts[0]
    result = context.get(root)
    if result is None:
        return None

    value: Any = result
    for token in parts[1:]:
        if not token:
            continue
        attr, indexes = split_indexes(token)
        if attr:
            value = _read_attr(value, attr, token, parts, frames=frames)
        for idx in indexes:
            value = _read_index(value, idx, token)
        if frames:
            value = _maybe_deserialize_frames(value)
    return value


def _read_attr(
    value: Any, attr: str, token: str, parts: list[str], *, frames: bool
) -> Any:
    if isinstance(value, dict) and attr in value:
        return value[attr]
    if isinstance(value, list) and all(
        isinstance(v, dict) and attr in v for v in value
    ):
        return [v[attr] for v in value]
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
        return plucked
    if (
        frames
        and isinstance(value, list)
        and all(isinstance(v, pd.DataFrame) for v in value)
    ):
        if any(attr not in v.columns for v in value):
            raise ExecutionError(
                f"{attr} not a valid column in one of the DataFrames for {token}."
            )
        return [v[attr].tolist() for v in value]
    if frames and isinstance(value, pd.DataFrame):
        if attr not in value.columns:
            raise ExecutionError(f"{attr} not a valid column in DataFrame for {token}.")
        return value[attr].tolist()
    if isinstance(value, BaseModel):
        resolved = getattr(value, attr, _SENTINEL)
        if resolved is _SENTINEL:
            raise ExecutionError(
                f"{attr} not a valid attribute of {type(value).__name__} for {token}."
            )
        return resolved
    raise ExecutionError(
        f"{attr} in {parts} is not a valid key - {type(value).__name__}, {value}"
    )


def _read_index(value: Any, idx: int, token: str) -> Any:
    if isinstance(value, list) and -len(value) <= idx < len(value):
        return value[idx]
    if isinstance(value, list) and all(isinstance(v, list) for v in value):
        return [v[idx] for v in value]
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
