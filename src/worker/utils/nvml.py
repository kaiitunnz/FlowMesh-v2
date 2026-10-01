"""Per-device NVML reads shared by the worker's periodic GPU readers.

Callers own ``nvmlInit()`` and their own error handling: NVML's init is refcounted and
the worker never shuts it down (see ``gpu_sampler``), and each reader decides for itself
what an unreadable device means.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pynvml


@dataclass(frozen=True)
class DeviceMemory:
    """One device's memory as NVML reports it."""

    used_bytes: int
    free_bytes: int


def decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def device_handles() -> Iterator[tuple[int, Any]]:
    """Each device this process can see, with its NVML index."""
    for index in range(pynvml.nvmlDeviceGetCount()):
        yield index, pynvml.nvmlDeviceGetHandleByIndex(index)


def device_uuid(handle: Any) -> str:
    return decode(pynvml.nvmlDeviceGetUUID(handle))


def device_name(handle: Any) -> str:
    return decode(pynvml.nvmlDeviceGetName(handle))


def device_memory(handle: Any) -> DeviceMemory:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return DeviceMemory(used_bytes=int(info.used), free_bytes=int(info.free))
