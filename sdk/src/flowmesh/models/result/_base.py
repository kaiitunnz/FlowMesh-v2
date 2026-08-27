# Necessary for the recursive ``children`` forward reference.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from ..artifacts import ArtifactContext

if TYPE_CHECKING:
    from .catalog import AnyExecutorResult


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseExecutorResult(BaseModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    ok: bool = True
    children: dict[str, SerializeAsAny[AnyExecutorResult]] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    artifacts_: ArtifactContext | None = Field(default=None, alias="_artifacts")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "artifacts_" in cls.__annotations__:
            raise TypeError(
                f"{cls.__name__} may not redefine the internal "
                "BaseExecutorResult.artifacts_ field"
            )


class StrictExecutorResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)
