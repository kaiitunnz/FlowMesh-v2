"""A task's durable per-task snapshot, and loading one under its workflow's stored
sources."""

from collections.abc import Mapping
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_serializer,
    model_serializer,
    model_validator,
)

from shared.tasks import PERSISTED_LOAD_CONTEXT

from ..task.models import TERMINAL_TASK_STATUSES, TaskRecord

# The validation-context key holding a workflow's stored sources by digest.
_SOURCES = "workflow_sources"
_SETTLED_EXCLUDES = {"dispatched_artifacts", "dispatched_producers"}


class PersistedTask(BaseModel):
    """A durable per-task snapshot sufficient to rebuild scheduler state.

    The record's workflow source is stored once per workflow and named here by its
    digest; loading it needs that workflow's stored sources.
    """

    model_config = ConfigDict(frozen=True)

    record: TaskRecord
    depends_on: set[str] = Field(default_factory=set)
    epoch_index: int | None = None

    @model_validator(mode="wrap")
    @classmethod
    def _attach_source(
        cls, data: Any, handler: ModelWrapValidatorHandler[Self], info: ValidationInfo
    ) -> Self:
        if not isinstance(data, dict) or (digest := data.get("source_digest")) is None:
            return handler(data)
        sources = (info.context or {}).get(_SOURCES, {})
        if (source := sources.get(digest)) is None:
            raise ValueError(f"workflow source {digest} is not stored")
        task = handler(
            {
                **{key: value for key, value in data.items() if key != "source_digest"},
                "record": {**data["record"], "raw_yaml": source},
            }
        )
        task.record.take_source_digest(digest)
        return task

    @field_serializer("depends_on")
    def _serialize_depends_on(self, value: set[str]) -> list[str]:
        return sorted(value)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        del data["record"]["raw_yaml"]
        data["source_digest"] = self.record.source_digest()
        # A settled task has no dispatch left to name artifacts or producers for.
        settled = self.record.status in TERMINAL_TASK_STATUSES
        data["record"]["internal"] = self.record.internal.model_dump(
            mode="json", exclude=_SETTLED_EXCLUDES if settled else None
        )
        return data


def load_context(sources: Mapping[str, str]) -> dict[str, Any]:
    """Return the validation context a stored task state loads under, given its
    workflow's stored ``sources``."""
    return {**PERSISTED_LOAD_CONTEXT, _SOURCES: sources}


def load_task_state(blob: str | bytes, sources: Mapping[str, str]) -> PersistedTask:
    return PersistedTask.model_validate_json(blob, context=load_context(sources))
