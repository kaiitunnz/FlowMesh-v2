from typing import Annotated, Literal

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


# The engine settings the serve executor sets itself: the model it serves and the name
# its requests address, where it listens, and the key its sidecar presents.
_FABRIC_OWNED_ENGINE_KEYS = frozenset(
    {"api_key", "host", "model", "port", "revision", "served_model_name"}
)


def _validate_serve_dispatchable(spec: "ServeSpecStrict | ServeSpecTemplate") -> None:
    """A serve task launches a persistent vLLM GPU server its executor configures.

    Its spec sets none of the engine settings the executor owns, and requests at least
    one GPU, or GPU scheduling and accounting are bypassed.
    """
    vllm = spec.model.vllm if spec.model is not None else None
    if owned := sorted(
        key for key in vllm or {} if key.replace("-", "_") in _FABRIC_OWNED_ENGINE_KEYS
    ):
        raise ValueError(f"model.vllm.{owned[0]} is not supported for a serve task")
    gpu = spec.gpu_requirements()
    if gpu and gpu.count is not None and gpu.count >= 1:
        return
    raise ValueError(
        "serve task launches a persistent vLLM server but requests no GPU. "
        "Set spec.resources.hardware.gpu.count >= 1."
    )
