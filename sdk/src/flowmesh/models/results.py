"""Result-related models.

Mirrors ``shared.schemas.result``. The per-task-type subclasses and the
``AnyExecutorResult`` discriminated union let ``Results.retrieve()`` deserialize a
result into its exact subclass. Drift against the shared definitions is guarded by
``tests/sdk/test_schema_compat.py``.
"""

# Necessary for the recursive ``children`` forward reference.
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    SerializeAsAny,
    Tag,
)

from .artifacts import ArtifactContext, ArtifactRef
from .common import TaskType


class PathResponse(BaseModel):
    ok: bool
    path: str


# --------------------------------------------------------------------------- #
# Nested payload models
# --------------------------------------------------------------------------- #
class GenerationUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_requests: int
    latency_sec: float


class EmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    total_tokens: int
    num_requests: int
    latency_sec: float
    embedding_dim: int


class InferenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int | None = None
    prompt: str | None = None
    output: JsonValue = None
    finish_reason: str | list[str | None] | None = None
    metadata: dict[str, Any] | None = None


class OmniImageItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    prompt: str
    image: ArtifactRef


class OmniSpeechItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    audio: ArtifactRef


class OmniAudioItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    prompt_index: int
    waveform_index: int
    prompt: str
    audio: ArtifactRef


class OmniGeneralItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    request_id: str
    prompt: str | None = None
    audio: ArtifactRef
    text: str | None = None


class CostEstimates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    num_queries: int
    avg_estimated_cost: float
    min_estimated_cost: float
    max_estimated_cost: float
    avg_estimated_rows: float
    min_estimated_rows: int
    max_estimated_rows: int


class DataRetrievalItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int | None = None
    query: str | None = None
    description: str | None = None
    params: Any = None
    table: dict[str, str] | None = None
    rows: int | None = None
    keys: list[str] | None = None
    content: list[Any] | None = None
    run_id: str | None = None
    access_chain: Any = None
    materialized_uri: str | None = None
    size_bytes: int | None = None
    transcript_url: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    steps_taken: int | None = None
    replay_latency_ms: int | float | None = None


class AgentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    output: str
    finish_reason: str


class AgentUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_time_sec: float
    num_requests: int
    agent_config: str


class AgentBatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0


class AgentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str | None = None
    tasks_count: int | None = None
    execution_log: list[str] = Field(default_factory=list)
    error: str | None = None
    batch_summary: AgentBatchSummary | None = None


class RagQdrant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    url: str


class RagEmbedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str


class RagSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int


class RagUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_sec: float
    num_queries: int
    total_results: int


class RagHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | str | None = None
    score: float | None = None
    payload: dict[str, Any] | None = None


class RagQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    query: str
    items: list[RagHit] = Field(default_factory=list)


class EchoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: JsonValue = None


# --------------------------------------------------------------------------- #
# Base + concrete result models
# --------------------------------------------------------------------------- #
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


class InferenceResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.INFERENCE] = TaskType.INFERENCE
    model: str | None = None
    items: list[InferenceItem] = Field(default_factory=list)
    usage: GenerationUsage | None = None


class EmbeddingResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.EMBEDDING] = TaskType.EMBEDDING
    model: str | None = None
    embedding_file: ArtifactRef | None = None
    usage: EmbeddingUsage | None = None
    count: int | None = None
    image_group_sizes: list[int] | None = None


class DiffusionResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.DIFFUSION] = TaskType.DIFFUSION
    model: str | None = None
    images: list[ArtifactRef] = Field(default_factory=list)


class ServeResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.SERVE] = TaskType.SERVE
    model: str
    port: int


class _TrainingResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    training_time_seconds: float | None = None
    error_message: str | None = None
    model_name: str | None = None
    dataset_size: int = 0
    output_dir: str | None = None
    checkpoints_dir: ArtifactRef | None = None


class SFTResult(_TrainingResult):
    task_type: Literal[TaskType.SFT] = TaskType.SFT
    resume_from_path: str | None = None
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None
    spawned_torchrun: bool = False


class LoRAResult(_TrainingResult):
    task_type: Literal[TaskType.LORA_SFT] = TaskType.LORA_SFT
    resume_from_path: str | None = None
    final_lora: ArtifactRef | None = None
    final_lora_archive: ArtifactRef | None = None


class PPOResult(_TrainingResult):
    task_type: Literal[TaskType.PPO] = TaskType.PPO
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None
    spawned_torchrun: bool = False


class DPOResult(_TrainingResult):
    task_type: Literal[TaskType.DPO] = TaskType.DPO
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None
    spawned_torchrun: bool = False


class ImageClassificationTrainingResult(_TrainingResult):
    task_type: Literal[TaskType.IMAGE_CLASSIFICATION_TRAINING] = (
        TaskType.IMAGE_CLASSIFICATION_TRAINING
    )
    num_labels: int = 0
    eval_accuracy: float | None = None
    train_losses: list[float] = Field(default_factory=list)
    resume_from_path: str | None = None
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None


class OmniResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    executor: str
    mode: str
    model: str | None


class OmniText2ImageResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2IMAGE] = TaskType.OMNI_TEXT2IMAGE
    executor: str = "omni_text2image"
    mode: str = "image"
    image: ArtifactRef | None
    items: list[OmniImageItem]


class OmniText2SpeechResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2SPEECH] = TaskType.OMNI_TEXT2SPEECH
    executor: str = "omni_text2speech"
    mode: str = "tts"
    audio: ArtifactRef | None
    sample_rate: int
    storyboard: dict[str, Any] | None = None
    items: list[OmniSpeechItem]


class OmniText2AudioResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2AUDIO] = TaskType.OMNI_TEXT2AUDIO
    executor: str = "omni_text2audio"
    mode: str = "bgm"
    audio: ArtifactRef | None
    sample_rate: int
    num_waveforms: int
    audio_length: float
    storyboard: dict[str, Any] | None = None
    items: list[OmniAudioItem]


class OmniText2GeneralResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2GENERAL] = TaskType.OMNI_TEXT2GENERAL
    executor: str = "omni_text2general"
    mode: str = "narration"
    audio: ArtifactRef | None
    sample_rate: int
    storyboard: dict[str, Any] | None = None
    items: list[OmniGeneralItem]


class DataProfilingResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.DATA_PROFILING] = TaskType.DATA_PROFILING
    type: str = "sql"
    template: str | None = None
    cost_estimates: CostEstimates | None = None


class DataRetrievalResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.DATA_RETRIEVAL] = TaskType.DATA_RETRIEVAL
    type: str | None = None
    items: list[DataRetrievalItem] = Field(default_factory=list)
    count: int | None = None
    metadata: dict[str, Any] | None = None


class AgentResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.AGENT] = TaskType.AGENT
    model: str
    items: list[AgentItem] = Field(default_factory=list)
    usage: AgentUsage | None = None
    metadata: AgentMetadata | None = None
    agent_output: ArtifactRef | None = None
    batch_summary_file: ArtifactRef | None = None


class RAGResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.RAG] = TaskType.RAG
    executor: str = "rag"
    qdrant: RagQdrant
    embedding: RagEmbedding
    search: RagSearch
    queries: list[RagQuery] = Field(default_factory=list)
    usage: RagUsage | None = None


class EchoResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.ECHO] = TaskType.ECHO
    items: list[EchoItem] = Field(default_factory=list)
    count: int = 0


class APIResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.API] = TaskType.API
    executor: str
    method: str
    url: str
    status_code: int
    truncated: bool = False
    headers: dict[str, str] | None = None
    response_json: Any = Field(default=None, alias="json")
    usage: dict[str, Any] | None = None
    text: str | None = None


class SSHResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    task_type: Literal[TaskType.SSH] = TaskType.SSH
    session_id: str
    exit_code: int
    command: list[str] | None = None
    entrypoint: list[str] | None = None
    expires_at: str | None = None
    host: str | None = None
    port: int | None = None


_BASE_TAG = "__base__"

_RESULT_TAGS: frozenset[str] = frozenset(
    {
        TaskType.INFERENCE.value,
        TaskType.EMBEDDING.value,
        TaskType.DIFFUSION.value,
        TaskType.SERVE.value,
        TaskType.SFT.value,
        TaskType.LORA_SFT.value,
        TaskType.PPO.value,
        TaskType.DPO.value,
        TaskType.IMAGE_CLASSIFICATION_TRAINING.value,
        TaskType.OMNI_TEXT2IMAGE.value,
        TaskType.OMNI_TEXT2SPEECH.value,
        TaskType.OMNI_TEXT2AUDIO.value,
        TaskType.OMNI_TEXT2GENERAL.value,
        TaskType.DATA_PROFILING.value,
        TaskType.DATA_RETRIEVAL.value,
        TaskType.AGENT.value,
        TaskType.RAG.value,
        TaskType.ECHO.value,
        TaskType.API.value,
        TaskType.SSH.value,
    }
)


def _result_discriminator(value: Any) -> str:
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
        | Annotated[SFTResult, Tag(TaskType.SFT.value)]
        | Annotated[LoRAResult, Tag(TaskType.LORA_SFT.value)]
        | Annotated[PPOResult, Tag(TaskType.PPO.value)]
        | Annotated[DPOResult, Tag(TaskType.DPO.value)]
        | Annotated[
            ImageClassificationTrainingResult,
            Tag(TaskType.IMAGE_CLASSIFICATION_TRAINING.value),
        ]
        | Annotated[OmniText2ImageResult, Tag(TaskType.OMNI_TEXT2IMAGE.value)]
        | Annotated[OmniText2SpeechResult, Tag(TaskType.OMNI_TEXT2SPEECH.value)]
        | Annotated[OmniText2AudioResult, Tag(TaskType.OMNI_TEXT2AUDIO.value)]
        | Annotated[OmniText2GeneralResult, Tag(TaskType.OMNI_TEXT2GENERAL.value)]
        | Annotated[DataProfilingResult, Tag(TaskType.DATA_PROFILING.value)]
        | Annotated[DataRetrievalResult, Tag(TaskType.DATA_RETRIEVAL.value)]
        | Annotated[AgentResult, Tag(TaskType.AGENT.value)]
        | Annotated[RAGResult, Tag(TaskType.RAG.value)]
        | Annotated[EchoResult, Tag(TaskType.ECHO.value)]
        | Annotated[APIResult, Tag(TaskType.API.value)]
        | Annotated[SSHResult, Tag(TaskType.SSH.value)]
        | Annotated[BaseExecutorResult, Tag(_BASE_TAG)]
    ),
    Discriminator(_result_discriminator),
]


class ResultEnvelope(BaseModel):
    """Canonical on-disk shape of ``results.json`` (mirrors the server)."""

    task_id: str
    result: SerializeAsAny[AnyExecutorResult]
    worker_id: str | None = None
    metadata: dict[str, Any] | None = None
    received_at: str | None = Field(default=None)


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
