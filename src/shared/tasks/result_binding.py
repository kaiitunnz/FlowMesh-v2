"""Where a settled task's result is, and which value of it a consumer reads.

A binding says where a settled task's envelope is, or that the task settled without
running; a value reference selects what a consumer reads out of it.
"""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from shared.content import ContentReference


class BindingKind(StrEnum):
    """What value a binding carries."""

    RESULT = "result"
    """A settled task's result, or the part ``element`` and ``path`` select."""
    MEMBERS = "members"
    """An aggregate: its members in order, each with its key and outcome."""
    BUNDLE = "bundle"
    """Named values returned together, by name."""
    LITERAL = "literal"
    """An inline value."""
    EMPTY = "empty"
    """An explicitly empty value."""


class ResultMember(BaseModel):
    """One member of an aggregate binding, with the value it settled with when it
    succeeded."""

    model_config = ConfigDict(frozen=True)

    key: str
    outcome: str
    binding: "ResultBinding | None" = None


class ResultBinding(BaseModel):
    """What a value an input reads resolves to: a settled task's stored envelope or
    skip, a part of one, an aggregate or bundle of such values, an inline literal, or
    an explicit empty. ``path`` reads into whatever value the rest names."""

    model_config = ConfigDict(frozen=True)

    # The task whose result the value reads; None for a value no one task produced.
    task_id: str | None = None
    reference: ContentReference | None = None
    skip: dict[str, Any] | None = None
    settled_at: str | None = None
    kind: BindingKind = BindingKind.RESULT
    # The part of the result whose list ``element`` indexes; empty for the result's
    # own collection.
    collection: tuple[str | int, ...] = ()
    element: int | None = None
    path: tuple[str | int, ...] = ()
    members: tuple[ResultMember, ...] = ()
    literal: str | None = None

    @property
    def whole_result(self) -> bool:
        """Whether the binding reads a task's whole result."""
        return (
            self.kind is BindingKind.RESULT and self.element is None and not self.path
        )


class ResultElementRef(BaseModel):
    """One element of a producer's stored result: a member of its collection, a value
    a path reaches inside the result, or a value a path reaches inside one member."""

    model_config = ConfigDict(frozen=True)

    reference: ContentReference
    collection: tuple[str | int, ...] = ()
    element: int | None = None
    path: tuple[str | int, ...] = ()

    @model_validator(mode="after")
    def _names_an_element(self) -> Self:
        if self.element is None and not self.path:
            raise ValueError("an element names a collection member or a path")
        return self


__all__ = [
    "BindingKind",
    "ResultBinding",
    "ResultElementRef",
    "ResultMember",
]


ResultMember.model_rebuild()
