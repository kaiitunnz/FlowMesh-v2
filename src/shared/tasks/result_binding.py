"""Where a settled task's result is, and which value of it a consumer reads.

A result is named, never carried: a binding says where a settled task's envelope is — or
that the task settled without running — and a value reference selects what a consumer
reads out of it.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

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


__all__ = ["ResultBinding", "ResultValueRef"]
