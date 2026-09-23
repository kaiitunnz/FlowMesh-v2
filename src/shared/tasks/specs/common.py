import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from ...schemas.result import BaseExecutorResult
from .._base import StrictBaseModel, TemplateBaseModel
from ..components import (
    AdapterConfig,
    AdapterConfigTemplate,
    ModelConfig,
    ModelConfigTemplate,
    OutputSpec,
    OutputSpecTemplate,
    PostprocessSpec,
    PostprocessSpecTemplate,
    ResourcesSpec,
    ShardSpec,
    ShardSpecTemplate,
)
from ..placeholders import TemplateBool, TemplateInt


def validate_adapters_loadable(
    adapters: "list[AdapterConfig] | list[AdapterConfigTemplate] | None",
    *,
    resident: bool,
) -> None:
    """Reject a declared adapter with no loadable source.

    A resident service binding serves a single adapter (the request selects one model),
    so more than one is rejected; every adapter must name a ``path``, ``url``, or
    ``task_id`` the runner or replica can load.
    """
    if not adapters:
        return
    if resident and len(adapters) > 1:
        raise ValueError(
            "a resident service binding supports a single adapter; declare at most one."
        )
    for adapter in adapters:
        if not adapter.path and not adapter.url and not adapter.task_id:
            raise ValueError(
                f"adapter {adapter.name or adapter.type!r} specifies no path, url, or "
                "task_id and cannot be loaded."
            )


def validate_resident_only_binding(
    binding: "ServiceBindingSpec | None", *, leaf: str
) -> None:
    """Reject a local-eligible binding on a leaf that admits only resident serving.

    A contract-equivalent local embodiment is proven for pinned chat inference; another
    leaf kind keeps the single embodiment its binding names. Only an explicitly declared
    mode is rejected: an undeclared one resolves to the leaf's own default.
    """
    if binding is not None and binding.mode is ServiceBindingMode.LOCAL_ELIGIBLE:
        raise ValueError(
            f"a {leaf} leaf supports only a resident service binding; local_eligible "
            "is available for inference leaves."
        )


class InferenceEmbodimentKind(StrEnum):
    """How one pinned inference contract is physically realized."""

    RESIDENT_SERVED = "resident_served"
    SELF_CONTAINED = "self_contained"


class ServiceBindingMode(StrEnum):
    """What the binding says about where the leaf's invocation may run."""

    RESIDENT = "resident"
    LOCAL_ELIGIBLE = "local_eligible"


class ServiceBindingSpec(BaseModel):
    """Binds a service-backed leaf to resident-served capacity.

    A ``resident`` binding admits the leaf's invocation to a compatible model-serving
    replica the fabric materializes and reuses. ``service_model_ref`` names the served
    model, defaulting to the task's own model source; ``isolation`` names a co-batch and
    cache isolation domain that is never shared across domains even for the same model.

    A ``local_eligible`` binding declares that the same pinned model contract may also
    be realized by a self-contained local executor. The leaf then carries the resident
    constraints below *and* the local model, executor, and resource constraints of an
    unbound leaf, and ``primary`` optionally names the embodiment the fabric prefers
    when both are placeable.

    An undeclared ``mode`` resolves to the leaf's own default: an inference leaf admits
    both embodiments, and a leaf kind with one proven embodiment keeps it.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ServiceBindingMode | None = None
    service_model_ref: str | None = None
    isolation: str | None = None
    primary: InferenceEmbodimentKind | None = None

    @model_validator(mode="after")
    def _validate_primary(self) -> "ServiceBindingSpec":
        if self.mode is ServiceBindingMode.RESIDENT and self.primary is not None:
            raise ValueError(
                "primary names which of several embodiments to prefer; a resident "
                "binding admits one."
            )
        return self


class ParallelSpec(StrictBaseModel):
    enabled: bool | None = None
    max_shards: int | None = None


class ParallelSpecTemplate(TemplateBaseModel):
    enabled: TemplateBool | None = None
    max_shards: TemplateInt | None = None


class ConditionSpec(StrictBaseModel):
    """Condition that must be met for this task to be dispatched.

    When the condition is not met, the server marks the task as completed
    immediately without dispatching it to a worker.
    """

    node: str = Field(description="Upstream task ID whose result to check.")
    field: str = Field(
        description="Dot-separated path into the upstream result payload "
        "(e.g. ``items.0.output``)."
    )
    equals: str = Field(
        description="Expected value. Task only dispatches if ``actual == equals``."
    )


def _validate_condition_depends_on[T: "TaskSpecStrictBase | TaskSpecTemplateBase"](
    spec: T,
) -> T:
    condition = spec.condition
    if condition is None:
        return spec
    depends_on = spec.dependsOn
    if not depends_on:
        return spec
    dependency_names = {
        dep_stripped for dep in depends_on if (dep_stripped := dep.strip())
    }
    node = condition.node.strip()
    if node not in dependency_names:
        raise ValueError(f"condition.node '{condition.node}' must appear in dependsOn.")
    return spec


def _merge_key_in(key: str | None, context: dict[str, Any]) -> str | None:
    if key is None:
        return None
    return json.dumps([key, context], ensure_ascii=False, sort_keys=True)


class TaskSpecStrictBase(StrictBaseModel):
    resources: ResourcesSpec | None = None
    output: OutputSpec | None = None
    dependsOn: list[str] | None = None
    condition: ConditionSpec | None = None
    shard: ShardSpec | None = None

    # Server-injected stage context (reserve the user-facing key `_upstreamResults`)
    upstreamResults: dict[str, SerializeAsAny[BaseExecutorResult]] | None = Field(
        default=None, alias="_upstreamResults"
    )

    @model_validator(mode="after")
    def _check_condition_depends_on(self) -> "TaskSpecStrictBase":
        _validate_condition_depends_on(self)
        return self

    def get_artifacts(self) -> list[str]:
        output = self.output
        if output is None:
            return []
        artifacts = output.artifacts
        if artifacts is None:
            return []
        return artifacts.copy()

    def validate_dispatchable(self) -> None:
        """Validate spec-internal invariants for a runnable task.

        Called at submit and again before dispatch. Overrides must raise ``ValueError``
        for misconfigurations.
        """
        return None

    def merge_key(self, **context: Any) -> str | None:
        """The key a task merges with its siblings under within ``context``, or None if
        it never merges."""
        return _merge_key_in(self._merge_key(), context)

    def _merge_key(self) -> str | None:
        """What the spec merges on, or None if it never merges."""
        return None


class TaskSpecTemplateBase(TemplateBaseModel):
    resources: ResourcesSpec | None = None
    output: OutputSpecTemplate | None = None
    dependsOn: list[str] | None = None
    condition: ConditionSpec | None = None
    shard: ShardSpecTemplate | None = None

    upstreamResults: dict[str, SerializeAsAny[BaseExecutorResult]] | None = Field(
        default=None, alias="_upstreamResults"
    )

    @model_validator(mode="after")
    def _check_condition_depends_on(self) -> "TaskSpecTemplateBase":
        _validate_condition_depends_on(self)
        return self

    def get_artifacts(self) -> list[str]:
        output = self.output
        if output is None:
            return []
        artifacts = output.artifacts
        if artifacts is None:
            return []
        return artifacts.copy()

    def validate_dispatchable(self) -> None:
        """Validate spec-internal invariants for a runnable task.

        Called at submit and again before dispatch. Overrides must defer
        placeholder-dependent checks and raise ``ValueError`` for genuine
        misconfigurations.
        """
        return None

    def merge_key(self, **context: Any) -> str | None:
        """The key a task merges with its siblings under within ``context``, or None if
        it never merges."""
        return _merge_key_in(self._merge_key(), context)

    def _merge_key(self) -> str | None:
        """What the spec merges on, or None if it never merges."""
        return None


type TaskSpecBase = TaskSpecStrictBase | TaskSpecTemplateBase


class ModelSpecStrict(TaskSpecStrictBase):
    model: ModelConfig | None = None

    @property
    def model_name(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.identifier  # type: ignore

    @property
    def model_revision(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.revision  # type: ignore

    @property
    def model_trust_remote_code(self) -> bool:
        model = self.model
        source = model and model.source
        return bool(source and source.trust_remote_code)

    @property
    def adapters(self) -> list[AdapterConfig] | None:
        model = self.model
        return None if model is None else model.adapters


class ModelSpecTemplate(TaskSpecTemplateBase):
    model: ModelConfigTemplate | None = None

    @property
    def model_name(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.identifier  # type: ignore

    @property
    def model_revision(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.revision  # type: ignore

    @property
    def model_trust_remote_code(self) -> bool:
        model = self.model
        source = model and model.source
        return bool(source and source.trust_remote_code)

    @property
    def adapters(self) -> list[AdapterConfigTemplate] | None:
        model = self.model
        return None if model is None else model.adapters


class ModelInferSpecStrict(ModelSpecStrict):
    data: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    postprocess: PostprocessSpec | None = None


class ModelInferSpecTemplate(ModelSpecTemplate):
    data: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    postprocess: PostprocessSpecTemplate | None = None
