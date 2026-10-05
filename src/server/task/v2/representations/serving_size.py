"""The hardware and tensor-parallel size a resident replica of one leaf serves at.

A leaf declares its hardware under ``resources.hardware`` and its tensor-parallel size
under ``model.vllm``. Both read here into one canonical size, so two spellings of the
same requirement share a replica and two different requirements never do.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from shared.tasks.components.resources import HardwareRequirements
from shared.tasks.placeholders import contains_placeholder
from shared.utils.hardware import normalize_gpu_type, parse_gpu_memory_bytes
from shared.utils.parsing import parse_mem_to_bytes

ANY_GPU_TYPE = "any"
_BINARY_UNITS = (("Ti", 1024**4), ("Gi", 1024**3), ("Mi", 1024**2), ("Ki", 1024))


class ServingSize(BaseModel):
    """The canonical hardware one resident replica runs on.

    The engine shards over ``tensor_parallel_size`` of its ``gpu_count`` devices, all on
    one worker.
    """

    model_config = ConfigDict(frozen=True)

    cpu: int = 2
    memory_bytes: int = 4 * 1024**3
    gpu_type: str = ANY_GPU_TYPE
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

        A GPU-free rendering asks for no device of any kind, so any worker places it.
        """
        devices: dict[str, Any] = {"type": ANY_GPU_TYPE, "count": 0}
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
        caps it. A value that renders from upstream, is not a quantity, or is below one
        is undeclared, as placement reads it.
        """
        default = DEFAULT_SERVING_SIZE
        gpu = hardware.gpu if hardware is not None else None
        count = _positive(gpu.count if gpu is not None else None)
        tp = _positive(tensor_parallel_size)
        if count is None:
            count = tp or default.gpu_count
        tp = min(tp, count) if tp is not None else count
        cpu = _positive(hardware.cpu if hardware is not None else None)
        memory = hardware.memory if hardware is not None else None
        return cls(
            cpu=cpu or default.cpu,
            memory_bytes=_positive(parse_mem_to_bytes(str(memory)))
            or default.memory_bytes,
            gpu_type=normalize_gpu_type(gpu.type if gpu is not None else None)
            or ANY_GPU_TYPE,
            gpu_count=count,
            gpu_memory_bytes=_gpu_memory_bytes(gpu.memory if gpu is not None else None),
            tensor_parallel_size=tp,
        )


DEFAULT_SERVING_SIZE = ServingSize()


def _positive(value: Any) -> int | None:
    if value is None or isinstance(value, bool) or contains_placeholder(value):
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
