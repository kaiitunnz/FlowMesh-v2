"""Typed nested payload models for executor results.

These models describe the exact shape each executor emits inside its result
fields (items, usage, cost estimates, ...). They are standalone — they do not
depend on ``BaseExecutorResult`` — so ``result.py`` imports them without a
cycle. Fields whose interior is genuinely open (arbitrary dataset columns,
Qdrant documents, opaque provenance) stay declared mappings, typed as narrowly
as the emitter allows.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue

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
