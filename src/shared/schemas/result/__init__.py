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
    AgentResult,
    AnyExecutorResult,
    APIResult,
    DataProfilingResult,
    DataRetrievalResult,
    DiffusionResult,
    DPOResult,
    EchoResult,
    EmbeddingResult,
    ImageClassificationTrainingResult,
    InferenceResult,
    LoRAResult,
    OmniResult,
    OmniText2AudioResult,
    OmniText2GeneralResult,
    OmniText2ImageResult,
    OmniText2SpeechResult,
    PPOResult,
    RAGResult,
    ResultEnvelope,
    ServeResult,
    SFTResult,
    SSHResult,
)
from ._io import read_result, result_file_path, write_result
from ._payloads import (
    AgentBatchSummary,
    AgentItem,
    AgentMetadata,
    AgentUsage,
    CostEstimates,
    DataRetrievalItem,
    EchoItem,
    EmbeddingUsage,
    GenerationUsage,
    InferenceItem,
    OmniAudioItem,
    OmniGeneralItem,
    OmniImageItem,
    OmniSpeechItem,
    RagEmbedding,
    RagHit,
    RagQdrant,
    RagQuery,
    RagSearch,
    RagUsage,
)

# Resolve the recursive ``children`` union on the base and every concrete
# subclass (each inherits the field and builds its own core schema).
_RESULT_MODELS: tuple[type[BaseModel], ...] = (
    BaseExecutorResult,
    InferenceResult,
    EmbeddingResult,
    DiffusionResult,
    ServeResult,
    SFTResult,
    LoRAResult,
    PPOResult,
    DPOResult,
    ImageClassificationTrainingResult,
    OmniText2ImageResult,
    OmniText2SpeechResult,
    OmniText2AudioResult,
    OmniText2GeneralResult,
    DataProfilingResult,
    DataRetrievalResult,
    AgentResult,
    RAGResult,
    EchoResult,
    APIResult,
    SSHResult,
    ResultEnvelope,
)
for _model in _RESULT_MODELS:
    _model.model_rebuild()

__all__ = [
    "APIResult",
    "AgentBatchSummary",
    "AgentItem",
    "AgentMetadata",
    "AgentResult",
    "AgentUsage",
    "AnyExecutorResult",
    "BaseExecutorResult",
    "CostEstimates",
    "DPOResult",
    "DataProfilingResult",
    "DataRetrievalItem",
    "DataRetrievalResult",
    "DiffusionResult",
    "EchoItem",
    "EchoResult",
    "EmbeddingResult",
    "EmbeddingUsage",
    "GenerationUsage",
    "ImageClassificationTrainingResult",
    "InferenceItem",
    "InferenceResult",
    "LoRAResult",
    "OmniAudioItem",
    "OmniGeneralItem",
    "OmniImageItem",
    "OmniResult",
    "OmniSpeechItem",
    "OmniText2AudioResult",
    "OmniText2GeneralResult",
    "OmniText2ImageResult",
    "OmniText2SpeechResult",
    "PPOResult",
    "RAGResult",
    "RagEmbedding",
    "RagHit",
    "RagQdrant",
    "RagQuery",
    "RagSearch",
    "RagUsage",
    "ResultEnvelope",
    "SFTResult",
    "SSHResult",
    "ServeResult",
    "read_result",
    "result_file_path",
    "write_result",
]
