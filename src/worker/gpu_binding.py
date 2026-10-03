"""Choosing the GPUs a binding executor runs a task on."""

from shared.tasks.components.resources import GPURequirements
from shared.tasks.worker_message import GpuInfo, WorkerHardware
from shared.utils.hardware import (
    parse_gpu_memory_bytes,
    select_matching_gpu_indices,
    unified_gpu_memory_satisfies,
)

from .executors.base_executor import ExecutionError
from .executors.ssh_session.config import FreeGpus
from .gpu_availability import DeviceAvailability


def free_uuids(
    reported: dict[str, DeviceAvailability], devices: list[GpuInfo]
) -> frozenset[str]:
    """The devices a reading does not mark held; one it did not cover is no opinion
    rather than held, so it counts as free."""
    return frozenset(
        device.uuid
        for device in devices
        if (seen := reported.get(device.uuid)) is None or seen.available
    )


def pick_devices(
    hardware: WorkerHardware,
    requirement: GPURequirements | None,
    free: FreeGpus,
    warm: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """The UUIDs, in the worker's device order, of the free devices a task runs on.

    ``requirement``'s count is how many, and with no positive count every free matching
    device. Devices match as placement matches them, a unified-memory pool included. A
    ``warm`` binding that is still free and still the pick's equal is kept, so a warm
    executor is not restarted only because the free set changed. The latched free set,
    the one placement used, is tried first; only a reading just taken refuses the task.
    """
    requirement = requirement or GPURequirements()
    count = int(requirement.count) if requirement.count is not None else 0
    limit = count if count > 0 else None

    def pick(usable: frozenset[str]) -> tuple[str, ...] | None:
        candidates = [d for d in hardware.gpu.devices if d.uuid in usable]
        matched = _matching(hardware, candidates, requirement, limit)
        if matched is None:
            return None
        if warm and set(warm) <= usable:
            if limit is None and set(warm) == {d.uuid for d in matched}:
                return warm
            chosen = [d for d in candidates if d.uuid in warm]
            if limit is not None and len(chosen) == limit:
                if _matching(hardware, chosen, requirement, limit) is not None:
                    return warm
        return tuple(device.uuid for device in matched)

    if (picked := pick(free.latched)) is not None:
        return picked
    if (picked := pick(free.fresh)) is not None:
        return picked
    raise ExecutionError(
        "too few of this worker's GPUs are free of processes outside FlowMesh for "
        "this task",
        retryable=True,
    )


def _matching(
    hardware: WorkerHardware,
    candidates: list[GpuInfo],
    requirement: GPURequirements,
    limit: int | None,
) -> list[GpuInfo] | None:
    """The ``candidates`` a task takes, ``limit`` of them or every match, or None when
    too few match; a unified pool large enough covers a device's unreported memory."""
    needed = limit or 1
    indices = select_matching_gpu_indices(candidates, requirement, limit=limit)
    if len(indices) < needed:
        memory = parse_gpu_memory_bytes(requirement.memory)
        if memory is None or not unified_gpu_memory_satisfies(hardware, memory, needed):
            return None
        type_only = GPURequirements(count=requirement.count, type=requirement.type)
        indices = select_matching_gpu_indices(candidates, type_only, limit=limit)
        if len(indices) < needed:
            return None
    return [candidates[index] for index in indices]
