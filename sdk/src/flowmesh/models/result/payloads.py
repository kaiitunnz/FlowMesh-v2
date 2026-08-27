"""Typed nested payload models for executor results."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ..artifacts import ArtifactRef


class PathResponse(BaseModel):
    ok: bool
    path: str


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
