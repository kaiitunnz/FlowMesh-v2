from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from shared.tasks.components.resources import HardwareRequirements
from shared.tasks.placeholders import contains_placeholder
from shared.utils.hardware import normalize_gpu_type, parse_gpu_memory_bytes
from shared.utils.parsing import parse_mem_to_bytes

_ANY_GPU_TYPE = "any"
_BINARY_UNITS = (("Ti", 1024**4), ("Gi", 1024**3), ("Mi", 1024**2), ("Ki", 1024))


class ServingSize(BaseModel):
    """The canonical hardware a resident replica runs on, read from its leaf.

    Equal requirements read as one size however they are spelled. The engine shards over
    ``tensor_parallel_size`` of its ``gpu_count`` devices, all on one worker.
    """

    model_config = ConfigDict(frozen=True)

    cpu: int = 2
    memory_bytes: int = 4 * 1024**3
    gpu_type: str = _ANY_GPU_TYPE
    gpu_count: int = 1
    # The memory each device needs at least; None takes any device.
    gpu_memory_bytes: int | None = None
    tensor_parallel_size: int = 1

    @property
    def is_default(self) -> bool:
        return self == DEFAULT_SERVING_SIZE

    @property
    def memory(self) -> str:
        """The memory as a quantity a hardware requirement reads."""
        return _quantity(self.memory_bytes)

    def key(self) -> str:
        """Return the size's canonical text, the one identity equal sizes share."""
        gpu_memory = (
            f"gpumem{_quantity(self.gpu_memory_bytes)},"
            if self.gpu_memory_bytes is not None
            else ""
        )
        # The GPU type is free text, so it ends the key.
        return (
            f"cpu{self.cpu},mem{self.memory},tp{self.tensor_parallel_size},"
            f"{gpu_memory}gpu{self.gpu_count}x{self.gpu_type}"
        )

    def hardware(self, gpu: bool = True) -> dict[str, Any]:
        """Render the size as a task's ``resources.hardware``, optionally GPU-free.

        A GPU-free rendering asks for no device of any kind.
        """
        devices: dict[str, Any] = {"type": _ANY_GPU_TYPE, "count": 0}
        if gpu:
            devices = {"type": self.gpu_type, "count": self.gpu_count}
            if self.gpu_memory_bytes is not None:
                devices["memory"] = _quantity(self.gpu_memory_bytes)
        return {"cpu": self.cpu, "memory": self.memory, "gpu": devices}

    @classmethod
    def of(
        cls, hardware: HardwareRequirements | None, tensor_parallel_size: Any = None
    ) -> Self:
        """Read a leaf's hardware and tensor-parallel size as a serving size.

        An omitted field takes the default size's value. With neither a GPU count nor a
        tensor-parallel size, both are one; either alone sets the other. A
        tensor-parallel size above the count is capped to it, as a local vLLM engine
        caps it. A value that is not a quantity or is below one is undeclared, as
        placement reads it. A value that renders from upstream is undeclared too, since
        the replica is sized before any upstream value exists.
        """
        default = DEFAULT_SERVING_SIZE
        gpu = hardware.gpu if hardware is not None else None
        count = _positive(_declared(gpu.count if gpu is not None else None))
        tp = _positive(_declared(tensor_parallel_size))
        if count is None:
            count = tp or default.gpu_count
        tp = min(tp, count) if tp is not None else count
        memory = _declared(hardware.memory if hardware is not None else None)
        gpu_type = _declared(gpu.type if gpu is not None else None)
        return cls(
            cpu=_positive(_declared(hardware.cpu if hardware is not None else None))
            or default.cpu,
            memory_bytes=_positive(parse_mem_to_bytes(str(memory)) if memory else None)
            or default.memory_bytes,
            gpu_type=normalize_gpu_type(gpu_type) or _ANY_GPU_TYPE,
            gpu_count=count,
            gpu_memory_bytes=_gpu_memory_bytes(
                _declared(gpu.memory if gpu is not None else None)
            ),
            tensor_parallel_size=tp,
        )


DEFAULT_SERVING_SIZE = ServingSize()


def _declared(value: Any) -> Any:
    """The value as declared, or None when it renders from upstream."""
    return None if value is None or contains_placeholder(value) else value


def _positive(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _gpu_memory_bytes(value: str | int | float | None) -> int | None:
    try:
        return _positive(parse_gpu_memory_bytes(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _quantity(size: int) -> str:
    for unit, scale in _BINARY_UNITS:
        if size % scale == 0:
            return f"{size // scale}{unit}"
    return str(size)
