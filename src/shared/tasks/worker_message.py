from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
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
from shared.tasks.result_binding import ResultBinding, ResultElementRef
from shared.tasks.specs.common import TaskSpecBase
from shared.utils.json import dedup_json, restore_json


def dispatch_uses_gpu(spec: TaskSpecBase, relays_only: bool) -> bool:
    """Whether a dispatch of ``spec`` allocates GPU memory on its worker.

    The dispatcher and the worker both decide through this, so a task the dispatcher
    places on a held device's worker is never one that worker refuses. A dispatch that
    relays only loads no model. Otherwise a declared request counts unless it asks for
    no devices, and a spec that loads onto a GPU counts whatever it declares.
    """
    if relays_only:
        return False
    declared = spec.gpu_requirements()
    return (declared is not None and declared.count != 0) or spec.uses_gpu()


class WorkerTaskMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(description="Dispatched task identifier.")
    workflow_id: str = Field(description="Workflow identifier owning the task.")
    owner_id: str = Field(description="Owner principal identifier.")
    content_scope: str = Field(
        default="",
        description="Authorization scope for this task's content, assigned by control.",
    )
    task: TaskEnvelopeStrict = Field(description="Task payload.")
    task_type: str | None = Field(default=None, description="Task type hint.")
    assigned_worker: str = Field(description="Worker ID selected for execution.")
    dispatch_id: str | None = Field(default=None, description="Dispatch ID.")
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
        description="Upstream task IDs by stage name.",
    )
    upstream_results: dict[str, ResultBinding] | None = Field(
        default=None,
        description="Upstream stage results by stage name.",
    )
    input_element: ResultElementRef | None = Field(
        default=None,
        description="Collection element a fan-out child runs on.",
    )
    credential_pointers: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Pointers to the restored credentials in each dispatched task's spec, by "
            "task id."
        ),
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

    # What the worker hydrated from the references above: each upstream stage's stored
    # envelope bytes, and the fan-out element as a one-item tuple.
    _upstream_envelopes: dict[str, bytes] = PrivateAttr(default_factory=dict)
    _element: tuple[Any] | None = PrivateAttr(default=None)

    @property
    def spec(self) -> TaskSpecStrict:
        return self.task.spec

    def record_hydration(
        self, envelopes: dict[str, bytes], element: tuple[Any] | None
    ) -> None:
        """Keep what the worker hydrated for the consumers that read it raw."""
        self._upstream_envelopes = dict(envelopes)
        self._element = element

    def upstream_envelope(self, stage: str) -> bytes | None:
        """The stored envelope bytes of an upstream stage, once hydrated."""
        return self._upstream_envelopes.get(stage)

    def hydrated_element(self) -> tuple[Any] | None:
        """The fan-out element this task runs on, once hydrated, as a one-item tuple."""
        return self._element

    @property
    def metadata(self) -> TaskMetadata | None:
        return self.task.metadata

    @property
    def relays_only(self) -> bool:
        """Whether this dispatch runs no local model: an input preparation, or a
        resident service episode that carries its invocation to a replica."""
        return self.input_preparation or self.service_episode is not None

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
    # Informational only, never a placement input: a reading taken while the worker's
    # own executor is warm cannot tell its memory from another tenant's.
    memory_free_bytes: int | None = Field(
        default=None, description="Free GPU memory in bytes at the last reading."
    )
    # The only field that gates placement. None means the worker reported no
    # observation, and the device schedules as if it had never been read.
    gpu_available: bool | None = Field(
        default=None,
        description="Whether no process outside FlowMesh holds this device.",
    )

    @property
    def is_available(self) -> bool:
        """Whether this device may be scheduled on; only an explicit ``False``
        withholds it, so a device nobody has read stays schedulable."""
        return self.gpu_available is not False


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
