"""Choosing the GPUs a binding executor runs a task on."""

from shared.tasks.components.resources import GPURequirements
from shared.tasks.worker_message import GpuInfo
from shared.utils.hardware import select_matching_gpu_indices

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
    devices: list[GpuInfo],
    requirement: GPURequirements | None,
    free: FreeGpus,
    warm: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """The UUIDs, in the worker's device order, of the free devices a task runs on.

    ``requirement``'s count is how many, and with no count every free matching device.
    A ``warm`` binding that is still free and still fits is kept, so a warm executor
    is not restarted only because the free set changed. The latched free set, the one
    placement used, is tried first; only a reading just taken refuses the task.
    """
    requirement = requirement or GPURequirements()
    if (count := requirement.count) is not None and int(count) <= 0:
        return ()

    def pick(usable: frozenset[str]) -> tuple[str, ...] | None:
        candidates = [device for device in devices if device.uuid in usable]
        if warm and set(warm) <= usable and _fits(warm, candidates, requirement):
            return warm
        limit = int(count) if count is not None else None
        matched = select_matching_gpu_indices(candidates, requirement, limit=limit)
        if len(matched) < (limit or 1):
            return None
        return tuple(candidates[index].uuid for index in matched)

    if (picked := pick(free.latched)) is not None:
        return picked
    if (picked := pick(free.fresh)) is not None:
        return picked
    raise ExecutionError(
        "too few of this worker's GPUs are free of processes outside FlowMesh for "
        "this task",
        retryable=True,
    )


def _fits(
    bound: tuple[str, ...], candidates: list[GpuInfo], requirement: GPURequirements
) -> bool:
    chosen = [device for device in candidates if device.uuid in bound]
    if requirement.count is not None and len(chosen) != int(requirement.count):
        return False
    return len(select_matching_gpu_indices(chosen, requirement)) == len(chosen)
