import asyncio

from .registries.resident import ResidentRegistry
from .resident.service import ResidentCapacityControl
from .task.runtime import TaskRuntime


async def rehydrate_root_state(
    runtime: TaskRuntime | None,
    resident_control: ResidentCapacityControl | None,
    resident_registry: ResidentRegistry | None,
) -> None:
    """Rebuild durable root state on startup in a credit-safe order.

    Resident-capacity control binds its loop and loads its claim store before the
    runtime rehydrates, because the runtime's rehydrate re-drives every suspended
    mediated boundary through the resident settler; were the control not running with
    its claims loaded first, an in-flight resident invocation would terminalize against
    an empty store and strand (or re-admit a second) credit.
    """
    if resident_control is not None and resident_registry is not None:
        resident_control.bind_loop(asyncio.get_running_loop())
        snapshot = await resident_registry.load_snapshot_async()
        if snapshot is not None:
            resident_control.rehydrate(snapshot)
    if runtime is not None:
        await runtime.rehydrate()
    if resident_control is not None and resident_registry is not None:
        resident_control.start()
