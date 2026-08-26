"""Base executor-result model.

Kept free of any ``shared.tasks`` import so it can be bound into the package
namespace before the concrete catalog pulls ``TaskType`` (which transitively
imports ``shared.tasks.specs.common``, and that imports this base back).
``children`` is a forward reference to ``AnyExecutorResult``; the package
``__init__`` rebuilds this model once the union exists.
"""

# Necessary for the recursive ``children`` forward reference.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from ..artifact import ArtifactContext

if TYPE_CHECKING:
    from ._catalog import AnyExecutorResult


class BaseExecutorResult(BaseModel):
    """Common shape for every executor's result payload.

    ``extra="allow"`` keeps this the permissive fallback of the discriminated
    union: legacy ``results.json`` without a ``task_type`` and condition-skip
    payloads round-trip through it without losing fields.
    """

    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    ok: bool = Field(default=True, description="Whether task execution succeeded.")
    children: dict[str, SerializeAsAny[AnyExecutorResult]] = Field(
        default_factory=dict,
        exclude_if=lambda v: not v,
        description="Per-child result payloads for task merging.",
    )
    artifacts_: ArtifactContext | None = Field(
        default=None,
        alias="_artifacts",
        description="Resolution context for relative artifact refs.",
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "artifacts_" in cls.__annotations__:
            raise TypeError(
                f"{cls.__name__} may not redefine the internal "
                "BaseExecutorResult.artifacts_ field"
            )
