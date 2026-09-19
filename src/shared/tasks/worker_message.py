from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    model_serializer,
    model_validator,
)

from shared.content import ContentReference
from shared.harness import AgentEpisodeDispatch, ServiceLeafEpisodeDispatch
from shared.inference import (
    CanonicalInferenceContract,
    CanonicalInferenceRequest,
    InputResolutionBinding,
)
from shared.schemas.worker import WorkerStatus
from shared.tasks import (
    TaskEnvelopeStrict,
    TaskSpecStrict,
)
from shared.tasks.components import TaskMetadata
from shared.tasks.merged import MergedChildTaskStrict
from shared.utils.json import dedup_json, restore_json


class WorkerTaskMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(description="Dispatched task identifier.")
    workflow_id: str = Field(description="Workflow identifier owning the task.")
    owner_id: str = Field(description="Owner principal identifier.")
    content_scope: str = Field(
        default="",
        description=(
            "The authorization scope content this task writes is isolated in, assigned "
            "by the control plane from the task's owner."
        ),
    )
    task: TaskEnvelopeStrict = Field(description="Task payload.")
    task_type: str | None = Field(default=None, description="Task type hint.")
    assigned_worker: str = Field(description="Worker ID selected for execution.")
    dispatched_at: str = Field(description="Dispatch timestamp (ISO8601).")
    parent_task_id: str | None = Field(
        default=None, description="Parent task ID (merge/shard)."
    )
    shard_index: int | None = Field(default=None, description="Shard index.")
    shard_total: int | None = Field(default=None, description="Total shard count.")
    merged_children: list[MergedChildTaskStrict] | None = Field(
        default=None, description="Optional merged child task payloads."
    )
    upstream_task_ids: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping from upstream stage name to resolved task ID.",
    )
    agent_episode: AgentEpisodeDispatch | None = Field(
        default=None,
        description="Agent-episode continuation context for a run-to-yield step.",
    )
    service_episode: ServiceLeafEpisodeDispatch | None = Field(
        default=None,
        description="Resident service-leaf episode context for a run-to-yield step.",
    )
    declared_contract: CanonicalInferenceContract | None = Field(
        default=None,
        description=(
            "Canonical inference contract for a leaf the fabric resolves rather than "
            "the executor: the model, sampling, and the source its prompts come from."
        ),
    )
    recorded_resolution: InputResolutionBinding | None = Field(
        default=None,
        description=(
            "The input resolution this task is already committed to, when one was "
            "recorded: a re-drive that resolves to anything else fails instead of "
            "running against substituted input."
        ),
    )
    input_preparation: bool = Field(
        default=False,
        description=(
            "Whether this dispatch only resolves the task's declared_contract and "
            "reports the request it materialized, running no model and no executor."
        ),
    )
    recorded_input: ContentReference | None = Field(
        default=None,
        description=(
            "The prepared request this task runs, when a preparation committed one: "
            "the worker hydrates and verifies it rather than resolving the source "
            "again."
        ),
    )
    resolved_contract: CanonicalInferenceRequest | None = Field(
        default=None,
        description=(
            "The request a worker materialized from declared_contract, set on the "
            "origin worker before either embodiment reaches its model: the request to "
            "issue and the result shape to report."
        ),
    )
    traceparent: str | None = Field(
        default=None,
        description=(
            "W3C traceparent naming the episode span this task's worker-side run "
            "parents on, when telemetry is enabled."
        ),
    )

    @property
    def spec(self) -> TaskSpecStrict:
        return self.task.spec

    @property
    def metadata(self) -> TaskMetadata | None:
        return self.task.metadata

    @model_validator(mode="before")
    @classmethod
    def _restore_deduped(cls, data: Any) -> Any:
        if isinstance(data, dict) and set(data) == {"content", "data"}:
            return restore_json(data)
        return data

    @model_serializer(mode="wrap")
    def _dedup(self, serializer: Any, info: SerializationInfo) -> Any:
        plain = serializer(self)
        if info.mode != "json":
            return plain
        return dedup_json(plain)


class CPUInfo(BaseModel):
    logical_cores: int = Field(description="Number of logical CPU cores.")
    model: str = Field(description="CPU model name.")


class MemoryInfo(BaseModel):
    total_bytes: int | None = Field(description="Total memory in bytes.")


class GpuInfo(BaseModel):
    index: int = Field(description="GPU index.")
    name: str = Field(description="GPU name.")
    uuid: str = Field(description="GPU UUID.")
    memory_total_bytes: int | None = Field(description="Total GPU memory in bytes.")


class GpuPlatformInfo(BaseModel):
    driver_version: str | None = Field(description="GPU driver version.")
    cuda_version: str | None = Field(description="CUDA version.")
    devices: list[GpuInfo] = Field(description="List of GPU devices.")
    memory_is_unified: bool = Field(
        default=False,
        description="Whether GPU memory is a unified/shared system memory pool.",
    )
    shared_memory_total_bytes: int | None = Field(
        default=None,
        description="Total shared GPU/system memory pool in bytes when unified.",
    )


class NetworkInfo(BaseModel):
    ip: str | None = Field(description="Network IP address.")
    bandwidth_bytes_per_sec: float | None = Field(
        description="Network bandwidth in bytes per second."
    )


class WorkerHardware(BaseModel):
    cpu: CPUInfo = Field(description="CPU information.")
    memory: MemoryInfo = Field(description="Memory information.")
    gpu: GpuPlatformInfo = Field(description="GPU information.")
    network: NetworkInfo = Field(description="Network information.")


class HardwareUsage(BaseModel):
    gpu: GpuPlatformInfo = Field(description="GPU information.")

    @classmethod
    def from_hw(cls, hw: WorkerHardware) -> "HardwareUsage":
        return cls(gpu=hw.gpu)


__all__ = [
    "CPUInfo",
    "GpuInfo",
    "GpuPlatformInfo",
    "HardwareUsage",
    "MemoryInfo",
    "NetworkInfo",
    "WorkerHardware",
    "WorkerStatus",
    "WorkerTaskMessage",
]
