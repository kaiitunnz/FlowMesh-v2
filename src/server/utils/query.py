"""Query-string filters over a declared set of fields.

An endpoint declares the fields it filters on and rejects a query naming any other
key. Values are read by attribute.

Matching:

- Values match exactly, as strings.
- Different keys combine with AND; a repeated key matches any of its values.
- A dotted key walks nested models and dicts (``hardware.cpu.model``); a path that
  meets ``None`` or an absent dict key on an item reads as ``None``.
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

_NULLS = frozenset({"", "null", "None"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


class InvalidQuery(ValueError):
    """A query naming a key the endpoint does not filter on."""


@dataclass(frozen=True)
class _Accepted:
    """The values one key accepts, in every form a field's value is compared as."""

    strings: frozenset[str]
    bools: frozenset[bool]
    null: bool

    @classmethod
    def of(cls, values: frozenset[str]) -> Self:
        spellings = {value.strip().lower() for value in values}
        bools = {True} if spellings & _TRUTHY else set()
        if spellings & _FALSY:
            bools.add(False)
        return cls(values, frozenset(bools), bool(values & _NULLS))

    def matches(self, value: Any, key: str) -> bool:
        if value is None:
            return self.null
        if isinstance(value, bool):
            return value in self.bools
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(str(member) in self.strings for member in value)
        if key == "tags" and isinstance(value, str):
            tags = (tag.strip() for tag in value.split(","))
            return any(tag in self.strings for tag in tags if tag)
        return str(value) in self.strings


@dataclass(frozen=True)
class QueryFilter:
    """The filter terms of one query: each declared key and the values it accepts."""

    terms: Mapping[str, frozenset[str]] = field(default_factory=dict)
    _accepted: Mapping[str, _Accepted] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        accepted = {key: _Accepted.of(values) for key, values in self.terms.items()}
        object.__setattr__(self, "_accepted", accepted)

    @classmethod
    def parse(
        cls,
        params: QueryParams,
        fields: Collection[str],
        skip: Collection[str] = (),
    ) -> Self:
        """Collect a query's filter terms, leaving out the keys in ``skip``; raise
        InvalidQuery for any other key not in ``fields``."""
        terms: dict[str, set[str]] = {}
        for key, value in params.multi_items():
            if key in skip:
                continue
            if key not in fields:
                raise InvalidQuery(f"unsupported filter {key!r}")
            terms.setdefault(key, set()).add(value)
        return cls({key: frozenset(values) for key, values in terms.items()})

    def __bool__(self) -> bool:
        return bool(self.terms)

    def values(self, key: str) -> frozenset[str] | None:
        """Return the values a key accepts, or None when the query does not name
        it."""
        return self.terms.get(key)

    def without(self, *keys: str) -> Self:
        return type(self)(
            {key: values for key, values in self.terms.items() if key not in keys}
        )

    def matches(self, item: Any) -> bool:
        return all(
            accepted.matches(_read(item, key), key)
            for key, accepted in self._accepted.items()
        )

    def filter[T](self, items: list[T]) -> list[T]:
        return [item for item in items if self.matches(item)] if self else items


def _read(item: Any, path: str) -> Any:
    current = item
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, BaseModel) and (
            part in type(current).model_fields
            or part in type(current).model_computed_fields
        ):
            current = getattr(current, part)
        else:
            return None
        if current is None:
            return None
    return current


__all__ = ["InvalidQuery", "QueryFilter"]
