"""Per-device NVML reads shared by the worker's periodic GPU readers.

Callers own ``nvmlInit()`` and their own error handling: NVML's init is refcounted and
the worker never shuts it down (see ``gpu_sampler``), and each reader decides for itself
what an unreadable device means.
"""

from dataclasses import dataclass
from typing import Any

import pynvml


@dataclass(frozen=True)
class DeviceMemory:
    """One device's memory as NVML reports it."""

    used_bytes: int
    free_bytes: int


def device_memory(handle: Any) -> DeviceMemory:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return DeviceMemory(used_bytes=int(info.used), free_bytes=int(info.free))
