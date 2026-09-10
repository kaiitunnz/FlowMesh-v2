from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .._base import StrictBaseModel, TemplateBaseModel
from ..placeholders import TemplateFloat
from ..task_type import TaskType
from .common import TaskSpecStrictBase, TaskSpecTemplateBase


class SandboxHostSpec(StrictBaseModel):
    """Binds a sandbox session to the resident sandbox-host family that admits it.

    ``profile`` names the host image and runtime shape the session needs; ``isolation``
    names a host-sharing domain that is never shared across domains even for the same
    profile.
    """

    profile: str = "posix-default"
    isolation: str | None = None


class SandboxCommandSpec(StrictBaseModel):
    """One command the session runs against its own private filesystem."""

    argv: list[str] = Field(min_length=1)
    stdin: str | None = None
    timeoutSeconds: float | None = None


class SandboxCommandSpecTemplate(TemplateBaseModel):
    argv: list[str] = Field(min_length=1)
    stdin: str | None = None
    timeoutSeconds: TemplateFloat | None = None


class SandboxSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.SANDBOX]
    sandbox: SandboxHostSpec = Field(default_factory=SandboxHostSpec)
    commands: list[SandboxCommandSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _commands_name_a_program(self) -> Self:
        for index, command in enumerate(self.commands):
            if not command.argv[0].strip():
                raise ValueError(f"sandbox command {index} names no program")
        return self


class SandboxSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.SANDBOX]
    sandbox: SandboxHostSpec = Field(default_factory=SandboxHostSpec)
    commands: list[SandboxCommandSpecTemplate] = Field(min_length=1)


class SandboxHostSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.SANDBOX_HOST]
    profile: str = "posix-default"
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None


class SandboxHostSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.SANDBOX_HOST]
    profile: str = "posix-default"
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
