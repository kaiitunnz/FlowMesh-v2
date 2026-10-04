import re
from typing import Annotated, Any, Literal

from pydantic import Field

from ..task_type import TaskType
from .common import ModelSpecStrict, ModelSpecTemplate


class ServeSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.SERVE]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["proxy", "forward"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
    forwardPort: Annotated[int, Field(ge=1, le=65535)] | None = None

    def validate_dispatchable(self) -> None:
        _validate_serve_dispatchable(self)

    def uses_gpu(self) -> bool:
        return True


class ServeSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.SERVE]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["proxy", "forward"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
    forwardPort: Annotated[int, Field(ge=1, le=65535)] | None = None

    def validate_dispatchable(self) -> None:
        _validate_serve_dispatchable(self)

    def uses_gpu(self) -> bool:
        return True


# The engine options the serve executor sets itself: the model it serves, where it
# listens, and the key its sidecar presents. vLLM's parser also reads a config file and
# accepts any unambiguous prefix of an option, so a spec key must be a plain option name
# that neither names a config file nor abbreviates an owned option.
_EXECUTOR_OWNED_ENGINE_OPTIONS = ("api_key", "host", "model", "port", "uds")
_ENGINE_CONFIG_OPTION = "config"
_PLAIN_ENGINE_OPTION = re.compile(r"[a-z0-9_]+")


def _refused_engine_option(key: str) -> bool:
    option = key.replace("-", "_")
    return (
        not _PLAIN_ENGINE_OPTION.fullmatch(option)
        or option == _ENGINE_CONFIG_OPTION
        or any(owned.startswith(option) for owned in _EXECUTOR_OWNED_ENGINE_OPTIONS)
    )


def engine_env_vars(value: Any) -> dict[str, str]:
    """The environment a serve spec's ``model.vllm.env_vars`` sets for its engine."""
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("model.vllm.env_vars must map variable names to strings")
    return value


def _validate_serve_dispatchable(spec: ServeSpecStrict | ServeSpecTemplate) -> None:
    """Refuse a serve spec setting an engine option its executor owns, or without a GPU.

    A serve task launches a persistent vLLM GPU server its executor configures; without
    a GPU, GPU scheduling and accounting are bypassed.
    """
    vllm = spec.model.vllm if spec.model is not None else None
    if refused := sorted(key for key in vllm or {} if _refused_engine_option(key)):
        raise ValueError(f"model.vllm.{refused[0]} is not supported for a serve task")
    engine_env_vars((vllm or {}).get("env_vars"))
    gpu = spec.gpu_requirements()
    if gpu and gpu.count is not None and gpu.count >= 1:
        return
    raise ValueError(
        "serve task launches a persistent vLLM server but requests no GPU. "
        "Set spec.resources.hardware.gpu.count >= 1."
    )
