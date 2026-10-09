"""Where a settled task's result is, and which value of it a consumer reads.

A binding says where a settled task's envelope is, or that the task settled without
running; a value reference selects what a consumer reads out of it.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from shared.content import ContentReference


class ResultBinding(BaseModel):
    """What a settled task's result resolves to: a stored envelope or a skip."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    reference: ContentReference | None = None
    skip: dict[str, Any] | None = None
    settled_at: str | None = None


class ResultValueRef(BaseModel):
    """One value a consumer reads from a producer's stored result.

    ``element`` selects one member of the producer's collection; without it the value is
    the whole result.
    """

    model_config = ConfigDict(frozen=True)

    reference: ContentReference
    element: int | None = None


class ResultElementRef(BaseModel):
    """One element of a producer's stored result: a member of its collection, a value
    a path reaches inside the result, or a value a path reaches inside one member."""

    model_config = ConfigDict(frozen=True)

    reference: ContentReference
    element: int | None = None
    path: tuple[str | int, ...] = ()

    @model_validator(mode="after")
    def _names_an_element(self) -> Self:
        if self.element is None and not self.path:
            raise ValueError("an element names a collection member or a path")
        return self


__all__ = ["ResultBinding", "ResultElementRef", "ResultValueRef"]
