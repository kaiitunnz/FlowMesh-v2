import asyncio
import logging
import time

from .network.rendezvous import RootRendezvousBridge
from .registries.node import NodeRegistry
from .registries.resident import ResidentRegistry
from .resident.service import ResidentCapacityControl
from .serve import GatedServe
from .task.runtime import TaskRuntime


async def rehydrate_root_state(
    runtime: TaskRuntime | None,
    resident_control: ResidentCapacityControl | None,
    resident_registry: ResidentRegistry | None,
    gated_serve: GatedServe | None = None,
) -> None:
    """Rebuild durable root state on startup in a credit-safe order.

    Resident-capacity control binds its loop and loads its claim store before the
    runtime rehydrates, because the runtime's rehydrate re-drives every suspended
    mediated boundary through the resident settler; were the control not running with
    its claims loaded first, an in-flight resident invocation would terminalize against
    an empty store and strand (or re-admit a second) credit. External serve terminals
    replay after the claim store loads so a claim left UNCERTAIN by a crash between its
    terminal fact and its release settles, before the runtime rehydrate re-drives.
    Workflow terminals replay once the runtime has restored each ledger, releasing a
    claim whose credit a crash kept past its ledger terminal. A worker reserved for a
    dispatch the restored runtime does not hold is released.
    """
    if resident_control is not None and resident_registry is not None:
        resident_control.bind_loop(asyncio.get_running_loop())
        snapshot = await resident_registry.load_snapshot_async()
        if snapshot is not None:
            resident_control.rehydrate(snapshot)
        if gated_serve is not None:
            gated_serve.reconcile_terminals()
    if runtime is not None:
        await runtime.rehydrate()
        await asyncio.to_thread(runtime.release_ended_reservations)
    if resident_control is not None and resident_registry is not None:
        if runtime is not None:
            resident_control.reconcile_workflow_terminals(
                runtime.resident_invocation_completed
            )
        resident_control.start()


_NODE_REFRESH_SEC = 1.0
_PUMP_BLOCK_MS = 1000
_PUMP_ERROR_BACKOFF_SEC = 0.5


def start_relay_bridge_pump(
    bridge: RootRendezvousBridge,
    nodes: NodeRegistry,
    logger: logging.Logger,
    extra_node_ids: tuple[str, ...] = (),
) -> asyncio.Task[None]:
    """Start one root bridge's pump loop over its own namespace.

    Waits on every attached node's up stream at once, plus any ``extra_node_ids`` such
    as an ingress edge's stream, and forwards each frame to its peer's down stream as it
    arrives. The node list refreshes about once a second, which bounds how long a newly
    attached node waits.
    """

    async def _pump() -> None:
        ids: list[str] = []
        refresh_at = 0.0
        while True:
            try:
                if time.monotonic() >= refresh_at:
                    ids = [node.id for node in await nodes.list_nodes_async()]
                    ids.extend(extra_node_ids)
                    refresh_at = time.monotonic() + _NODE_REFRESH_SEC
                if ids:
                    await bridge.pump_ready(ids, _PUMP_BLOCK_MS)
                else:
                    await asyncio.sleep(_NODE_REFRESH_SEC)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("relay bridge pump failed")
                await asyncio.sleep(_PUMP_ERROR_BACKOFF_SEC)

    return asyncio.create_task(_pump())
