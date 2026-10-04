"""One root's durable task and resident state, restartable through the real startup
order over a real task runtime and the wired resident control."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Self, cast

from lumid_hooks import PrincipalContext

from server.config import OrchestrationConfig, ResidentCapacityConfig
from server.resident import ReplicaState, ServiceFamily
from server.resident.service import ResidentCapacityControl
from server.resident.state import ReplicaIncarnation, ResidentSnapshot
from server.resident.wiring import build_resident_capacity
from server.startup import rehydrate_root_state
from server.task.runtime import TaskRuntime
from shared.utils.ids import new_dispatch_id
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.resident.test_service import _Delivery
from tests.server.result_store import make_result_reader
from tests.server.task.test_resident_origin_loss import (
    _RESIDENT_WF,
    _capture_resident_boundary,
)
from tests.server.task.test_v2_orchestration import FakeRegistry, _register
from tests.server.task.test_worker_originated_boundary import _WorkerStub
from tests.support.waiting import pop_ready, until

TS = "2026-06-01T00:00:00Z"
SYSTEM = PrincipalContext(
    principal_id="operator",
    org_id="acme",
    external_id="op",
    principal_type="admin",
    scopes=["*"],
)
FAMILY = ServiceFamily(family="m|chat", engine_batch_key="m|chat", model_ref="m")
ENGINE_KEY = "engine-key"


class _SnapshotRegistry:
    """Round-trips the resident snapshot through JSON, as the Redis registry does."""

    def __init__(self) -> None:
        self.blob: str | None = None

    def save_snapshot(self, snapshot: ResidentSnapshot) -> None:
        self.blob = snapshot.model_dump_json()

    async def load_snapshot_async(self) -> ResidentSnapshot | None:
        return ResidentSnapshot.model_validate_json(self.blob) if self.blob else None


class _YieldingRegistry(FakeRegistry):
    """Suspends on every async read and write, as a Redis round trip does, so work
    scheduled on the loop interleaves with the runtime's rehydrate."""

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if not (name.endswith("_async") and callable(attr)):
            return attr
        call = cast(Callable[..., Awaitable[Any]], attr)

        async def suspended(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return await call(*args, **kwargs)

        return suspended


class _Workers(_WorkerStub):
    def release_worker(self, *args: Any) -> bool:
        return False

    def reservations(self) -> list[Any]:
        return []


class Node:
    """One root's durable state, shared across its restarts."""

    def __init__(
        self,
        cold_start_deadline_sec: float = 60.0,
        substrate: str = "dev_model",
        forward_api_key: str | None = None,
    ) -> None:
        self.cold_start_deadline_sec = cold_start_deadline_sec
        self.substrate = substrate
        self.forward_api_key = forward_api_key
        self.tasks = _YieldingRegistry()
        self.resident = _SnapshotRegistry()
        self.runtime, self.control = self._boot()

    def _boot(self) -> tuple[TaskRuntime, ResidentCapacityControl]:
        runtime = TaskRuntime(
            cast(Any, self.tasks),
            cast(Any, _Workers()),
            OrchestrationConfig(),
            make_result_reader(),
            logging.getLogger("restart-test"),
            credential_vault=InMemoryCredentialVault(),
        )
        control = build_resident_capacity(
            runtime=runtime,
            orchestration=OrchestrationConfig(
                resident=ResidentCapacityConfig(
                    enabled=True,
                    substrate=self.substrate,
                    poll_interval_sec=0.01,
                    cold_start_deadline_sec=self.cold_start_deadline_sec,
                    forward_api_key=self.forward_api_key,
                )
            ),
            system_principal=lambda: SYSTEM,
            registry=cast(Any, self.resident),
            logger=logging.getLogger("restart-test"),
        )
        self.delivery = _Delivery()
        control.set_worker_delivery(self.delivery.build())
        return runtime, control

    def restart(self) -> None:
        asyncio.run(self.restart_async())

    async def restart_async(
        self, on_boot: Callable[[Self], None] | None = None
    ) -> None:
        """Restart the root; ``on_boot`` sees the new runtime and control first."""
        self.control.shutdown()
        self.runtime, self.control = self._boot()
        if on_boot is not None:
            on_boot(self)
        await rehydrate_root_state(self.runtime, self.control, cast(Any, self.resident))

    def materialize(self) -> ReplicaIncarnation:
        return asyncio.run(self.materialize_async())

    async def materialize_async(self) -> ReplicaIncarnation:
        """Cold-start one replica of the family; its serve task is registered."""
        self.control.stores.families.register(FAMILY)
        return await self.control._lifecycle.materialize(FAMILY)

    def serve(
        self,
        serve_task_id: str,
        worker_id: str = "wkr-1",
        dispatch_id: str | None = None,
        port: int = 8001,
        reported_key: str | None = None,
    ) -> None:
        """Dispatch the serve task and have it report its engine endpoint.

        ``reported_key`` is an engine key the update carries, as an earlier worker
        build reported one.
        """
        dispatch_id = dispatch_id or new_dispatch_id()
        self.dispatch(serve_task_id, worker_id, dispatch_id)
        self.runtime.mark_started(serve_task_id, worker_id, {}, TS, dispatch_id)
        serve: dict[str, Any] = {
            "_host": "10.0.0.5",
            "_port": port,
            "model": "m",
            "interface": "chat",
        }
        if reported_key is not None:
            serve["_api_key"] = reported_key
        self.runtime.mark_updated(
            serve_task_id, worker_id, {"serve": serve}, dispatch_id
        )

    def dispatch(
        self, task_id: str, worker_id: str = "wkr-1", dispatch_id: str | None = None
    ) -> None:
        """Hand the next ready task, which must be ``task_id``, to a worker."""
        assert pop_ready(self.runtime) == task_id
        record_dispatch(
            self.runtime, task_id, worker_id, dispatch_id or new_dispatch_id()
        )

    async def submit_serve_async(self) -> str:
        """Submit a user's serve task, outside resident capacity; return its id."""
        _workflow_id, entries = await self.runtime.register(
            SYSTEM.principal_id,
            SYSTEM.org_id,
            json.dumps(
                {
                    "apiVersion": "flowmesh/v1",
                    "kind": "Serve",
                    "metadata": {"name": "standing"},
                    "spec": {
                        "taskType": "dev_model",
                        "resources": {"hardware": {"cpu": 1, "memory": "1Gi"}},
                        "model": {"source": {"type": "huggingface", "identifier": "m"}},
                    },
                }
            ),
            format="native",
        )
        return entries[0].task_id

    def adopt_standing(self, serve_task_id: str) -> ReplicaIncarnation:
        """Adopt a serving task as its family's standing replica."""
        endpoint = self.control.probe_serve_endpoint(serve_task_id)
        assert endpoint is not None
        self.control.stores.families.register(FAMILY)
        return self.control._lifecycle.adopt_standing_replica(
            FAMILY, serve_task_id=serve_task_id, binding_generation=1, endpoint=endpoint
        )

    def warm(self) -> ReplicaIncarnation:
        return asyncio.run(self.warm_async())

    async def warm_async(self) -> ReplicaIncarnation:
        replica = await self.materialize_async()
        assert replica.serve_task_id is not None
        self.serve(replica.serve_task_id)
        self.control._promote_ready_replicas(FAMILY.family)
        assert replica.state is ReplicaState.WARM
        return replica

    def persist(self) -> None:
        self.resident.save_snapshot(self.control.stores.to_snapshot())

    def replica(self, replica_id: str) -> ReplicaIncarnation:
        replica = self.control.stores.directory.get(replica_id)
        assert replica is not None
        return replica

    def status(self, task_id: str) -> str:
        record = self.runtime.get_record(task_id)
        assert record is not None
        return record.status


async def admitted_boundary(node: Node) -> ReplicaIncarnation:
    """Admit a resident agent boundary onto a replica cold-started for it."""
    node.control.bind_loop(asyncio.get_running_loop())
    _, ids = await _register(node.runtime, _RESIDENT_WF)
    _capture_resident_boundary(node.runtime, ids["writer"])
    directory = node.control.stores.directory
    await until(lambda: any(r.serve_task_id for r in directory.all()))
    (replica,) = directory.all()
    assert replica.serve_task_id is not None
    node.serve(replica.serve_task_id, "wkr-2")
    await until(lambda: "resident_handoff" in node.delivery.kinds())
    return replica


def handoff_replicas(node: Node) -> list[tuple[str, int]]:
    return [
        (p["handoff"]["replica_id"], p["handoff"]["incarnation"])
        for _w, kind, p in node.delivery.relays
        if kind == "resident_handoff"
    ]
