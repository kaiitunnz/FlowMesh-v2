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

from ..task.models import TaskRecord

# The validation-context key holding a workflow's stored sources by digest.
_SOURCES = "workflow_sources"


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
        # ``failed_workers`` is excluded from TaskRecord's dump but routes retries, so
        # it must survive a restart.
        data["record"]["failed_workers"] = self.record.failed_workers.copy()
        # Likewise the result binding and skip detail: internal to the task's result
        # read, and what that read resolves from after a restart.
        reference = self.record.result_reference
        data["record"]["result_reference"] = (
            reference.model_dump(mode="json") if reference is not None else None
        )
        data["record"]["result_skip"] = self.record.result_skip
        # And whether its artifact catalog has anything a restart must load.
        data["record"]["has_artifacts"] = self.record.has_artifacts
        # And which worker a merged dispatch went to, which tells a later report of
        # that worker's failure apart from the task's own.
        data["record"]["merged_dispatch_worker"] = self.record.merged_dispatch_worker
        # And the dispatch holding the task, which fences its worker's events.
        data["record"]["dispatch_id"] = self.record.dispatch_id
        # And the dispatch that reported its latest update, which fences an endpoint
        # the update carries to the dispatch that served it.
        data["record"][
            "latest_update_dispatch_id"
        ] = self.record.latest_update_dispatch_id
        # And where its vaulted credentials go back into its spec at dispatch.
        data["record"]["credential_refs"] = self.record.credential_refs
        # And when its first run started, which a serve task's TTL counts from.
        data["record"]["first_started_ts"] = self.record.first_started_ts
        return data


def load_context(sources: Mapping[str, str]) -> dict[str, Any]:
    """Return the validation context a stored task state loads under, given its
    workflow's stored ``sources``."""
    return {**PERSISTED_LOAD_CONTEXT, _SOURCES: sources}


def load_task_state(blob: str | bytes, sources: Mapping[str, str]) -> PersistedTask:
    return PersistedTask.model_validate_json(blob, context=load_context(sources))
