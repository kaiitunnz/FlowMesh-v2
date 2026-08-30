import re
from enum import StrEnum
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator

from ..task_type import TaskType
from .common import (
    ModelSpecStrict,
    ModelSpecTemplate,
    TaskSpecStrictBase,
    TaskSpecTemplateBase,
)

# A harness param key is credential-bearing when it contains one of these substrings
# or is a segment-boundary credential word (so ``max_tokens`` is allowed but
# ``auth_token`` is not). A model credential belongs in the binding's inline ``api_key``
# (vaulted at submission), never in an opaque harness param.
_CREDENTIAL_SUBSTRINGS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "credential",
    "authorization",
    "access_key",
    "auth_token",
    "access_token",
    "bearer_token",
    "session_token",
    "refresh_token",
)
_CREDENTIAL_SEGMENTS = frozenset({"auth"})


def _looks_credential(key: str) -> bool:
    lowered = key.lower()
    if any(sub in lowered for sub in _CREDENTIAL_SUBSTRINGS):
        return True
    return bool(_CREDENTIAL_SEGMENTS & set(re.split(r"[_\-\s]+", lowered)))


def _find_credential_key(value: Any) -> str | None:
    """The first credential-looking key anywhere in a nested params structure."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if _looks_credential(str(key)):
                return str(key)
            if (found := _find_credential_key(nested)) is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            if (found := _find_credential_key(item)) is not None:
                return found
    return None


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

    An external ``openai`` upstream carries an inline ``api_key`` (the user's own
    credential, held as a masked ``SecretStr``); a ``resident`` binding names a
    FlowMesh-served model by reference with no url or credential. The credential is
    vaulted server-side at submission and never persists on the compiled binding.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ModelBindingMode | None = None
    url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None
    service_model_ref: str | None = None

    @model_validator(mode="after")
    def _check_coherent(self) -> Self:
        if _has_url_credentials(self.url):
            raise ValueError("model_binding.url must not embed credentials")
        if self.service_model_ref and self.url:
            raise ValueError("model_binding cannot set both service_model_ref and url")
        if self.mode is ModelBindingMode.RESIDENT and (self.url or self.api_key):
            raise ValueError("a resident model_binding carries no url or api_key")
        if self.mode is ModelBindingMode.OPENAI and self.service_model_ref:
            raise ValueError("an openai model_binding carries no service_model_ref")
        if self.mode in (ModelBindingMode.CANNED, ModelBindingMode.ECHO) and (
            self.url or self.model or self.api_key or self.service_model_ref
        ):
            raise ValueError(f"a {self.mode} model_binding carries no url/model/key")
        if self.api_key is not None and self.service_model_ref:
            raise ValueError("a resident model_binding carries no api_key")
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
        if (key := _find_credential_key(params)) is not None:
            raise ValueError(
                f"harness param {key!r} looks credential-bearing; put a model "
                "credential in model_binding.api_key"
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
