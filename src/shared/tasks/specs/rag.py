from typing import Any, ClassVar, Literal

from ..task_type import TaskType
from .common import TaskSpecStrictBase, TaskSpecTemplateBase


class RagSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.RAG]

    qdrant: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None
    search: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    query: str | None = None


class RagSpecTemplate(TaskSpecTemplateBase):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *TaskSpecTemplateBase.credential_fields,
        "qdrant",
        "embedding",
        "search",
        "data",
    )
    taskType: Literal[TaskType.RAG]

    qdrant: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None
    search: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    query: str | None = None
