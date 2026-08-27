"""Typed nested payload models for executor results.

These models describe the exact shape each executor emits inside its result
fields (items, usage, cost estimates, ...). They are standalone — they do not
depend on ``BaseExecutorResult`` — so ``result.py`` imports them without a
cycle. Fields whose interior is genuinely open (arbitrary dataset columns,
Qdrant documents, opaque provenance) stay declared mappings, typed as narrowly
as the emitter allows.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ..artifact import ArtifactRef


class GenerationUsage(BaseModel):
    """Token/latency accounting for text-generation inference."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_requests: int
    latency_sec: float


class EmbeddingUsage(BaseModel):
    """Token/latency accounting for embedding inference."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    total_tokens: int
    num_requests: int
    latency_sec: float
    embedding_dim: int


class InferenceItem(BaseModel):
    """One prompt's generation output.

    ``output`` is polymorphic: a plain string, a structured JSON value when a
    template schema is applied, or a list of grouped outputs (table / grouped-
    image modes). ``metadata`` is open dataset/user passthrough.
    """

    model_config = ConfigDict(extra="forbid")

    index: int | None = None
    prompt: str | None = None
    output: JsonValue = None
    finish_reason: str | list[str | None] | None = None
    metadata: dict[str, Any] | None = None


class OmniImageItem(BaseModel):
    """One text-to-image generation."""

    model_config = ConfigDict(extra="forbid")

    index: int
    prompt: str
    image: ArtifactRef


class OmniSpeechItem(BaseModel):
    """One text-to-speech generation."""

    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    audio: ArtifactRef


class OmniAudioItem(BaseModel):
    """One text-to-audio (BGM) waveform."""

    model_config = ConfigDict(extra="forbid")

    index: int
    prompt_index: int
    waveform_index: int
    prompt: str
    audio: ArtifactRef


class OmniGeneralItem(BaseModel):
    """One text-to-general (narration) segment."""

    model_config = ConfigDict(extra="forbid")

    index: int
    request_id: str
    prompt: str | None = None
    audio: ArtifactRef
    text: str | None = None


class CostEstimates(BaseModel):
    """Aggregated query cost/row estimates for data profiling."""

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
    """One retrieved row/object.

    Shapes differ across the SQL, S3, and Lumid (sql/agent) connectors, and
    the Lumid fields (``access_chain``, token/step metrics) pass through a
    remote contract verbatim. Declared fields are typed; ``extra="allow"``
    keeps the item robust to connector-specific additions. ``access_chain``
    is an opaque provenance object and stays untyped.
    """

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
    """One agent task's final output."""

    model_config = ConfigDict(extra="forbid")

    index: int
    output: str
    finish_reason: str


class AgentUsage(BaseModel):
    """Agent execution accounting."""

    model_config = ConfigDict(extra="forbid")

    execution_time_sec: float
    num_requests: int
    agent_config: str


class AgentBatchSummary(BaseModel):
    """Per-batch agent completion counts."""

    model_config = ConfigDict(extra="forbid")

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0


class AgentMetadata(BaseModel):
    """Agent run metadata (single, batch, or error variant)."""

    model_config = ConfigDict(extra="forbid")

    task: str | None = None
    tasks_count: int | None = None
    execution_log: list[str] = Field(default_factory=list)
    error: str | None = None
    batch_summary: AgentBatchSummary | None = None


class RagQdrant(BaseModel):
    """Qdrant collection the RAG query ran against."""

    model_config = ConfigDict(extra="forbid")

    collection: str
    url: str


class RagEmbedding(BaseModel):
    """Embedding model used for the RAG query."""

    model_config = ConfigDict(extra="forbid")

    model: str


class RagSearch(BaseModel):
    """RAG search parameters."""

    model_config = ConfigDict(extra="forbid")

    top_k: int


class RagUsage(BaseModel):
    """RAG query accounting."""

    model_config = ConfigDict(extra="forbid")

    latency_sec: float
    num_queries: int
    total_results: int


class RagHit(BaseModel):
    """One Qdrant search hit. ``payload`` is the arbitrary stored document."""

    model_config = ConfigDict(extra="forbid")

    id: int | str | None = None
    score: float | None = None
    payload: dict[str, Any] | None = None


class RagQuery(BaseModel):
    """Hits for one RAG query."""

    model_config = ConfigDict(extra="forbid")

    index: int
    query: str
    items: list[RagHit] = Field(default_factory=list)


class EchoItem(BaseModel):
    """One echoed value."""

    model_config = ConfigDict(extra="forbid")

    output: JsonValue = None
