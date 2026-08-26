"""Executor result schemas: the base model, exact per-task-type models, and the
``task_type`` discriminated union that deserializes ``results.json``.

``_base`` imports before ``_catalog``: importing ``TaskType`` re-enters this
package through ``shared.tasks.specs.common``, which needs ``BaseExecutorResult``
already bound. Models carrying the recursive ``children`` field are rebuilt below
once ``AnyExecutorResult`` exists.
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
