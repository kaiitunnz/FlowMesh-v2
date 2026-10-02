"""Query-string filters over a declared set of fields.

An endpoint declares the fields it filters on; a query naming any other key is
rejected, so a filter only ever matches what the endpoint already serves. A value is
read by attribute, so filtering never serializes a model.

Matching:

- Values match exactly, as strings.
- Different keys combine with AND; a repeated key matches any of its values.
- A dotted key walks nested models and dicts (``hardware.cpu.model``); a path absent
  on an item leaves that key unchecked for it.
- A list field matches when it contains any value.
- A ``tags`` field holding a comma-separated string matches any of its tags.
- A null field matches ``""``, ``null`` or ``None``.
- A bool field matches ``1/true/yes/on`` or ``0/false/no/off``, case-insensitively.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from pydantic import BaseModel
from starlette.datastructures import QueryParams

PAGE_PARAMS = frozenset({"limit", "before", "after"})

_MISSING = object()
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


class InvalidQuery(ValueError):
    """A query naming a key the endpoint does not filter on."""


@dataclass(frozen=True)
class QueryFilter:
    """The filter terms of one query: each declared key and the values it accepts."""

    terms: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def parse(
        cls,
        params: QueryParams | Mapping[str, str],
        fields: Collection[str],
        reserved: Collection[str] = PAGE_PARAMS,
    ) -> Self:
        """Collect a query's terms; raises InvalidQuery for a key not in ``fields``."""
        items = (
            params.multi_items()
            if isinstance(params, QueryParams)
            else list(params.items())
        )
        terms: dict[str, list[str]] = {}
        for key, value in items:
            key = str(key)
            if key in reserved:
                continue
            if key not in fields:
                raise InvalidQuery(f"unsupported filter {key!r}")
            terms.setdefault(key, []).append(str(value))
        return cls({key: tuple(values) for key, values in terms.items()})

    def __bool__(self) -> bool:
        return bool(self.terms)

    def values(self, key: str) -> frozenset[str] | None:
        """The values a key accepts, or None when the query does not name it."""
        values = self.terms.get(key)
        return frozenset(values) if values is not None else None

    def without(self, *keys: str) -> Self:
        return type(self)(
            {key: values for key, values in self.terms.items() if key not in keys}
        )

    def matches(self, item: Any) -> bool:
        for key, values in self.terms.items():
            value = _read(item, key)
            if value is not _MISSING and not _matches(value, key, values):
                return False
        return True

    def filter[T](self, items: list[T]) -> list[T]:
        return [item for item in items if self.matches(item)] if self else items


def _read(item: Any, path: str) -> Any:
    current = item
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        elif isinstance(current, BaseModel) and (
            part in type(current).model_fields
            or part in type(current).model_computed_fields
        ):
            current = getattr(current, part)
        else:
            return _MISSING
        if current is _MISSING:
            return _MISSING
    return current


def _matches(value: Any, key: str, accepted: tuple[str, ...]) -> bool:
    if value is None:
        return any(v in ("", "null", "None") for v in accepted)
    if isinstance(value, bool):
        spellings = {v.strip().lower() for v in accepted}
        return bool(spellings & (_TRUTHY if value else _FALSY))
    if isinstance(value, (list, tuple, set, frozenset)):
        members = {str(v) for v in value}
        return any(v in members for v in accepted)
    if key == "tags" and isinstance(value, str):
        tags = {t.strip() for t in value.split(",") if t.strip()}
        return any(v in tags for v in accepted)
    return any(str(value) == v for v in accepted)


__all__ = ["PAGE_PARAMS", "InvalidQuery", "QueryFilter"]
