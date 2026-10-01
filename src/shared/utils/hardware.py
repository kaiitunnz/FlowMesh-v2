"""Helpers for parsing GPU requirement specs and matching them against devices."""

import re

from shared.tasks.components.resources import GPURequirements
from shared.tasks.specs import SSHSpecStrict, SSHSpecTemplate
from shared.tasks.specs.common import TaskSpecBase
from shared.tasks.worker_message import GpuInfo, WorkerHardware, dispatch_uses_gpu
from shared.utils.parsing import parse_mem_to_bytes

_GPU_TYPE_WILDCARDS = frozenset({"", "any", "auto", "*"})


def normalize_gpu_type(value: str | None) -> str | None:
    """Lowercase the type; ``None`` for wildcards (``''``/``any``/``auto``/``*``)."""
    if value is None:
        return None
    normalized = value.strip().lower()
    return None if normalized in _GPU_TYPE_WILDCARDS else normalized


def gpu_type_pattern(value: str | None) -> re.Pattern[str] | None:
    """Case-insensitive substring matcher, or ``None`` for wildcard."""
    normalized = normalize_gpu_type(value)
    if normalized is None:
        return None
    return re.compile(re.escape(normalized), re.IGNORECASE)


def parse_gpu_memory_bytes(value: str | int | float | None) -> int | None:
    """Parse ``GPURequirements.memory`` (str / int / float / None) to bytes.

    Returns ``None`` for ``None`` and for unparsable strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return parse_mem_to_bytes(value)
    return int(value)


def gpu_device_matches(
    device: GpuInfo,
    *,
    type_pattern: re.Pattern[str] | None = None,
    min_memory_bytes: int | None = None,
) -> bool:
    """Per-device predicate; ``None`` arg means 'no constraint'."""
    if type_pattern is not None and not type_pattern.search(device.name or ""):
        return False
    return (
        min_memory_bytes is None or (device.memory_total_bytes or 0) >= min_memory_bytes
    )


def available_devices(devices: list[GpuInfo]) -> list[GpuInfo]:
    """Devices not known to be held by a process outside FlowMesh."""
    return [device for device in devices if device.is_available]


def unified_gpu_memory_satisfies(
    hw: WorkerHardware, required_memory_bytes: int, required_count: int
) -> bool:
    """Pessimistic per-slot share of a unified GPU/system memory pool.

    Returns ``True`` only when ``hw.gpu.memory_is_unified`` and
    ``shared_memory_total_bytes / required_count >= required_memory_bytes``.
    """
    if not hw.gpu.memory_is_unified:
        return False
    shared_total = hw.gpu.shared_memory_total_bytes or 0
    if shared_total <= 0:
        return False
    per_gpu_share = shared_total / max(required_count, 1)
    return per_gpu_share >= required_memory_bytes


def select_matching_gpu_indices(
    devices: list[GpuInfo],
    gpu_req: GPURequirements,
    *,
    limit: int | None = None,
) -> list[int]:
    """Indices of devices that individually pass ``gpu_req``'s type + memory.

    Stops after ``limit`` matches when set.
    """
    if limit is not None and limit <= 0:
        return []
    type_pattern = gpu_type_pattern(gpu_req.type)
    min_memory_bytes = (
        parse_gpu_memory_bytes(gpu_req.memory) if gpu_req.memory else None
    )
    result: list[int] = []
    for idx, device in enumerate(devices):
        if not gpu_device_matches(
            device,
            type_pattern=type_pattern,
            min_memory_bytes=min_memory_bytes,
        ):
            continue
        result.append(idx)
        if limit is not None and len(result) >= limit:
            break
    return result


def gpu_meets_requirements(hw: WorkerHardware, gpu_req: GPURequirements) -> bool:
    """Whether ``hw``'s devices satisfy ``gpu_req``'s count, type and memory."""
    required_count = gpu_req.count
    if required_count is not None:
        try:
            required_count = int(required_count)
        except Exception:
            required_count = None
    required_type = normalize_gpu_type(gpu_req.type)
    required_memory_bytes = parse_gpu_memory_bytes(gpu_req.memory)
    needed = required_count or 1

    entries = hw.gpu.devices
    if entries:
        if required_count is not None and len(entries) < required_count:
            return False
        if len(select_matching_gpu_indices(entries, gpu_req)) >= needed:
            return True
        # Unified-memory fallback: when memory is the binding constraint and
        # the worker exposes a unified GPU/system pool large enough to cover
        # the request, still admit it.
        if required_memory_bytes is None:
            return False
        type_only_req = GPURequirements(
            count=gpu_req.count, type=gpu_req.type, memory=None
        )
        if len(select_matching_gpu_indices(entries, type_only_req)) < needed:
            return False
        return unified_gpu_memory_satisfies(hw, required_memory_bytes, needed)

    # Fallback when workers report aggregate GPU data instead of per-device entries.
    count = 0 if hw is None else len(hw.gpu.devices)
    if required_count is not None and count < required_count:
        return False

    first_gpu = hw.gpu.devices[0] if hw and hw.gpu.devices else None
    type_value = first_gpu.name if first_gpu else None
    if required_type and not str(type_value or "").strip().lower().startswith(
        required_type
    ):
        return False

    if required_memory_bytes:
        total_mem = 0 if first_gpu is None else (first_gpu.memory_total_bytes or 0)
        if total_mem <= 0 and unified_gpu_memory_satisfies(
            hw, required_memory_bytes, needed
        ):
            return True
        if total_mem <= 0:
            return False
        per_gpu = total_mem / max(needed, 1)
        if per_gpu < required_memory_bytes:
            return False

    return True


def gpus_fit_dispatch(
    hw: WorkerHardware, spec: TaskSpecBase, relays_only: bool
) -> bool:
    """Whether ``hw``'s devices, with their reported availability, fit a dispatch.

    An SSH session that selects devices is handed only those, so it needs enough free
    ones. Any other GPU dispatch runs on every device its worker sees, so one held
    device makes the worker unavailable to it.
    """
    devices = hw.gpu.devices
    if all(device.is_available for device in devices):
        return True
    if not dispatch_uses_gpu(spec, relays_only):
        return True
    if not isinstance(spec, SSHSpecStrict | SSHSpecTemplate):
        return False
    if (selection := spec.gpu_selection()) is None:
        return False
    if not (free := available_devices(devices)):
        return False
    free_hw = hw.model_copy(update={"gpu": hw.gpu.model_copy(update={"devices": free})})
    return gpu_meets_requirements(free_hw, selection)
