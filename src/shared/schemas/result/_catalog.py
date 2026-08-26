"""Concrete per-task-type result models and the ``task_type`` union.

Importing ``TaskType`` here pulls ``shared.tasks`` (via ``specs.common``,
which imports ``BaseExecutorResult`` back). The package ``__init__`` imports
``_base`` first, so the base is already bound in the package namespace when
that re-entrant import happens.
"""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SerializeAsAny,
    Tag,
)

from shared.tasks.task_type import TaskType
from shared.utils.time import now_iso

from ..artifact import ArtifactRef
from ._base import BaseExecutorResult
from ._payloads import EmbeddingUsage, GenerationUsage, InferenceItem


class InferenceResult(BaseExecutorResult):
    """Text-generation inference output (vLLM / HF Transformers)."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.INFERENCE] = TaskType.INFERENCE
    model: str | None = None
    items: list[InferenceItem] = Field(default_factory=list)
    usage: GenerationUsage | None = None


class EmbeddingResult(BaseExecutorResult):
    """Embedding inference output (vLLM / HF visual-embedding)."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.EMBEDDING] = TaskType.EMBEDDING
    model: str | None = None
    embedding_file: ArtifactRef | None = None
    usage: EmbeddingUsage | None = None
    count: int | None = None
    image_group_sizes: list[int] | None = None


class DiffusionResult(BaseExecutorResult):
    """Diffusion image-generation output."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.DIFFUSION] = TaskType.DIFFUSION
    model: str | None = None
    images: list[ArtifactRef] = Field(default_factory=list)


class ServeResult(BaseExecutorResult):
    """Model-serving endpoint descriptor."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.SERVE] = TaskType.SERVE
    model: str
    port: int


_BASE_TAG = "__base__"

# Concrete result classes keyed by their ``task_type`` discriminator tag. New
# task-type families extend this map; the union and tag set derive from it.
_RESULT_CLASSES: dict[str, type[BaseExecutorResult]] = {
    TaskType.INFERENCE.value: InferenceResult,
    TaskType.EMBEDDING.value: EmbeddingResult,
    TaskType.DIFFUSION.value: DiffusionResult,
    TaskType.SERVE.value: ServeResult,
}

_RESULT_TAGS: frozenset[str] = frozenset(_RESULT_CLASSES)


def _result_discriminator(value: Any) -> str:
    """Map a raw dict or a model instance to its union tag.

    Missing or unrecognized ``task_type`` (legacy ``results.json``,
    condition-skip base payloads, future task types) falls back to the
    permissive base model.
    """
    if isinstance(value, dict):
        tag = value.get("task_type")
    else:
        tag = getattr(value, "task_type", None)
    if tag is None:
        return _BASE_TAG
    tag = str(tag)
    return tag if tag in _RESULT_TAGS else _BASE_TAG


AnyExecutorResult = Annotated[
    (
        Annotated[InferenceResult, Tag(TaskType.INFERENCE.value)]
        | Annotated[EmbeddingResult, Tag(TaskType.EMBEDDING.value)]
        | Annotated[DiffusionResult, Tag(TaskType.DIFFUSION.value)]
        | Annotated[ServeResult, Tag(TaskType.SERVE.value)]
        | Annotated[BaseExecutorResult, Tag(_BASE_TAG)]
    ),
    Discriminator(_result_discriminator),
]


class ResultEnvelope(BaseModel):
    task_id: str = Field(description="Task identifier.")
    result: SerializeAsAny[AnyExecutorResult] = Field(
        description="Result payload data."
    )
    worker_id: str | None = Field(
        default=None, description="Worker identifier submitting the result."
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional result metadata."
    )
    received_at: str = Field(
        default_factory=now_iso, description="Result receipt timestamp."
    )
