import asyncio
import logging

from .network.rendezvous import RootRendezvousBridge
from .registries.node import NodeRegistry
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


def start_resident_bridge_pump(
    bridge: RootRendezvousBridge,
    nodes: NodeRegistry,
    logger: logging.Logger,
) -> asyncio.Task[None]:
    """Start the root bridge's pump loop.

    Sweeps every attached node on a short non-blocking interval, forwarding each node's
    up stream to its peers' down streams.
    """

    async def _pump() -> None:
        while True:
            try:
                for node in await nodes.list_nodes_async():
                    await bridge.pump_node(node.id)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("resident relay bridge pump failed")
            await asyncio.sleep(0.05)

    return asyncio.create_task(_pump())
