from typing import Annotated, Literal

from pydantic import Field

from ..task_type import TaskType
from .common import ModelSpecStrict, ModelSpecTemplate


class DevModelSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.DEV_MODEL]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward", "proxy"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None


class DevModelSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.DEV_MODEL]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward", "proxy"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
