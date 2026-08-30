from enum import StrEnum
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..task_type import TaskType
from .common import (
    ModelSpecStrict,
    ModelSpecTemplate,
    TaskSpecStrictBase,
    TaskSpecTemplateBase,
)

# Substrings that mark a harness/model-binding key as credential-bearing. Credentials
# reach an upstream only through a server-side secret_ref, never through opaque config.
_CREDENTIAL_KEY_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
)


def _has_url_credentials(url: str | None) -> bool:
    if not url:
        return False
    parts = urlsplit(url)
    return bool(parts.username or parts.password)


class ModelBindingMode(StrEnum):
    """The finite ways an agent's managed model boundary is satisfied.

    ``canned``/``echo`` are deterministic and credential-free; ``openai`` names an
    external OpenAI-compatible upstream; ``resident`` names a FlowMesh-served model
    dependency by reference, with no url or credential.
    """

    CANNED = "canned"
    ECHO = "echo"
    OPENAI = "openai"
    RESIDENT = "resident"


class AgentModelBindingSpec(BaseModel):
    """The source model binding an agent declares beside its harness.

    A workflow chooses a model dependency, never a raw provider connection or a
    credential: ``secret_ref`` names an authorized server-side secret, and an extra
    field (e.g. a raw ``api_key``) is rejected outright.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ModelBindingMode | None = None
    url: str | None = None
    model: str | None = None
    secret_ref: str | None = None
    service_model_ref: str | None = None

    @model_validator(mode="after")
    def _check_coherent(self) -> Self:
        if _has_url_credentials(self.url):
            raise ValueError("model_binding.url must not embed credentials")
        if self.service_model_ref and self.url:
            raise ValueError("model_binding cannot set both service_model_ref and url")
        if self.mode is ModelBindingMode.RESIDENT and (self.url or self.secret_ref):
            raise ValueError("a resident model_binding carries no url or secret_ref")
        if self.mode is ModelBindingMode.OPENAI and self.service_model_ref:
            raise ValueError("an openai model_binding carries no service_model_ref")
        if self.mode in (ModelBindingMode.CANNED, ModelBindingMode.ECHO) and (
            self.url or self.model or self.secret_ref or self.service_model_ref
        ):
            raise ValueError(f"a {self.mode} model_binding carries no url/model/ref")
        return self


class AgentHarnessSpec(BaseModel):
    """Declares the harness backend that drives an agent as a run-to-yield episode.

    ``backend``/``version`` select and pin the adapter binding; ``params`` are its
    non-secret backend configuration. An agent without this runs the legacy UTU path.
    """

    backend: str
    version: str = "v1"
    params: dict[str, Any] = {}

    @field_validator("params")
    @classmethod
    def _reject_credential_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        for key in params:
            lowered = key.lower()
            if any(marker in lowered for marker in _CREDENTIAL_KEY_MARKERS):
                raise ValueError(
                    f"harness param {key!r} looks credential-bearing; use a secret_ref"
                )
        return params


class ApiSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.API]
    api: dict[str, Any] | None = None


class ApiSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.API]
    api: dict[str, Any] | None = None


class EchoSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.ECHO]
    data: dict[str, Any] | None = None


class EchoSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.ECHO]
    data: dict[str, Any] | None = None


class AgentSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.AGENT]

    configName: str | None = None
    task: str | None = None
    agent: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    harness: AgentHarnessSpec | None = None
    model_binding: AgentModelBindingSpec | None = None


class AgentSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.AGENT]

    configName: str | None = None
    task: str | None = None
    agent: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    harness: AgentHarnessSpec | None = None
    model_binding: AgentModelBindingSpec | None = None


class DataProfilingSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.DATA_PROFILING]
    data: dict[str, Any] | None = None


class DataProfilingSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_PROFILING]
    data: dict[str, Any] | None = None


class DataRetrievalSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None


class DataRetrievalSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None


class EmbeddingSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None


class EmbeddingSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None
