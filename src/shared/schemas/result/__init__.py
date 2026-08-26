"""Executor result schemas: base, exact per-task-type models, and the
``task_type`` discriminated union used to deserialize ``results.json``
end-to-end.

Import order matters. ``_base`` binds ``BaseExecutorResult`` into this package
namespace first, so the re-entrant ``from shared.schemas.result import
BaseExecutorResult`` performed by ``shared.tasks.specs.common`` (pulled in when
``_catalog`` imports ``TaskType``) resolves. The recursive ``children`` field
is a forward reference to ``AnyExecutorResult``; every model carrying it is
rebuilt below once the union exists.
"""

from pydantic import BaseModel

from ._base import BaseExecutorResult
from ._catalog import (
    AnyExecutorResult,
    DiffusionResult,
    EmbeddingResult,
    InferenceResult,
    ResultEnvelope,
    ServeResult,
)
from ._io import read_result, result_file_path, write_result
from ._payloads import (
    EmbeddingUsage,
    GenerationUsage,
    InferenceItem,
)

# Resolve the recursive ``children`` union on the base and every concrete
# subclass (each inherits the field and builds its own core schema).
_RESULT_MODELS: tuple[type[BaseModel], ...] = (
    BaseExecutorResult,
    InferenceResult,
    EmbeddingResult,
    DiffusionResult,
    ServeResult,
    ResultEnvelope,
)
for _model in _RESULT_MODELS:
    _model.model_rebuild()

__all__ = [
    "AnyExecutorResult",
    "BaseExecutorResult",
    "DiffusionResult",
    "EmbeddingResult",
    "EmbeddingUsage",
    "GenerationUsage",
    "InferenceItem",
    "InferenceResult",
    "ResultEnvelope",
    "ServeResult",
    "read_result",
    "result_file_path",
    "write_result",
]
