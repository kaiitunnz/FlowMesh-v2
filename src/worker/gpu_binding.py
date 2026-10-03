"""Choosing the GPUs a task runs on among those free of processes outside FlowMesh."""

from dataclasses import dataclass
from typing import Protocol, Self

from shared.tasks.components.resources import GPURequirements
from shared.tasks.worker_message import GpuInfo, WorkerHardware
from shared.utils.hardware import fitting_gpu_indices

from .executors.base_executor import ExecutionError
from .gpu_availability import DeviceAvailability


class GpuReadings(Protocol):
    def gpu_availability(self) -> dict[str, DeviceAvailability]: ...

    def live_gpu_availability(self) -> dict[str, DeviceAvailability]: ...


@dataclass(frozen=True, slots=True)
class FreeGpus:
    """The UUIDs of the devices no process outside FlowMesh is known to hold.

    ``latched`` follows the reading the worker last reported, the one placement used;
    ``fresh`` follows only the reading just taken, so a device it did not measure
    counts as free.
    """

    latched: frozenset[str]
    fresh: frozenset[str]

    @classmethod
    def read(cls, readings: GpuReadings, devices: list[GpuInfo]) -> Self:
        """Read both free sets of ``devices`` from a worker's GPU readings."""
        return cls(
            latched=_free_uuids(readings.gpu_availability(), devices),
            fresh=_free_uuids(readings.live_gpu_availability(), devices),
        )


def _free_uuids(
    reported: dict[str, DeviceAvailability], devices: list[GpuInfo]
) -> frozenset[str]:
    # A device the reading did not cover is no opinion rather than held.
    return frozenset(
        device.uuid
        for device in devices
        if (seen := reported.get(device.uuid)) is None or seen.available
    )


def matching_positions(
    hardware: WorkerHardware,
    requirement: GPURequirements,
    limit: int | None,
    usable: frozenset[str] | None = None,
) -> list[int] | None:
    """Return the positions in the worker's devices of those a task takes.

    Takes ``limit`` of the ``usable`` devices (every device when None) that fit
    ``requirement``, or every fitting one without a limit; None when too few fit.
    """
    devices = hardware.gpu.devices
    positions = [
        i for i, device in enumerate(devices) if usable is None or device.uuid in usable
    ]
    fitting = fitting_gpu_indices(
        hardware, [devices[i] for i in positions], requirement, limit=limit
    )
    return None if fitting is None else [positions[i] for i in fitting]


def free_matching(
    hardware: WorkerHardware,
    requirement: GPURequirements,
    limit: int | None,
    free: FreeGpus,
) -> tuple[frozenset[str], list[int]] | None:
    """Return the free set a task's devices come from and their positions.

    Tries the latched set first, the one placement used, so only a reading just taken
    refuses a task; None when neither set has enough fitting devices.
    """
    for usable in (free.latched, free.fresh):
        if (
            positions := matching_positions(hardware, requirement, limit, usable)
        ) is not None:
            return usable, positions
    return None


def pick_devices(
    hardware: WorkerHardware,
    requirement: GPURequirements | None,
    free: FreeGpus,
    warm: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Pick the UUIDs, in the worker's device order, of the free devices a task runs on.

    Takes ``requirement``'s positive count of them, else every free fitting device, so
    a task without a count binds a different set once more devices free up, and its
    warm executor restarts on that set. With a count, a still-free ``warm`` binding that
    fits is kept, so the warm executor is reused.
    """
    requirement = requirement or GPURequirements()
    count = int(requirement.count) if requirement.count is not None else 0
    limit = count if count > 0 else None
    devices = hardware.gpu.devices
    if (found := free_matching(hardware, requirement, limit, free)) is None:
        raise ExecutionError(
            "too few of this worker's GPUs are free of processes outside FlowMesh for "
            "this task",
            retryable=True,
        )
    usable, positions = found
    picked = tuple(devices[i].uuid for i in positions)
    if warm and set(warm) <= usable:
        if limit is None and set(warm) == set(picked):
            return warm
        if (
            limit is not None
            and len(warm) == limit
            and matching_positions(hardware, requirement, limit, frozenset(warm))
            is not None
        ):
            return warm
    return picked
