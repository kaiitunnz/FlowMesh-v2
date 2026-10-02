"""A root restart re-attaches each live resident replica and reaps what backs nothing.

Each scenario runs the real startup order over a real task runtime and the wired
resident control, restarting both from the durable state the first run persisted.
"""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any, cast

from lumid_hooks import PrincipalContext

from server.config import OrchestrationConfig, ResidentCapacityConfig
from server.resident import ReplicaState, ServiceFamily
from server.resident.service import ResidentCapacityControl
from server.resident.state import (
    AdmissionProfile,
    ReplicaIncarnation,
    ResidentSnapshot,
)
from server.resident.wiring import build_resident_capacity
from server.startup import rehydrate_root_state
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
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

_TS = "2026-06-01T00:00:00Z"
_SYSTEM = PrincipalContext(
    principal_id="operator",
    org_id="acme",
    external_id="op",
    principal_type="admin",
    scopes=["*"],
)
_FAMILY = ServiceFamily(family="m|chat", engine_batch_key="m|chat", model_ref="m")
_ENGINE_KEY = "engine-key"


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


class _Node:
    """One root's durable state, shared across its restarts."""

    def __init__(self) -> None:
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
                    enabled=True, substrate="dev_model", poll_interval_sec=0.01
                )
            ),
            system_principal=lambda: _SYSTEM,
            registry=cast(Any, self.resident),
            logger=logging.getLogger("restart-test"),
        )
        self.delivery = _Delivery()
        control.set_worker_delivery(self.delivery.build())
        runtime.set_resident_handlers(
            originate=control.originate,
            on_ack=control.on_bootstrap_ack,
            on_outcome=control.on_outcome,
            on_route_observation=control.on_route_observation,
        )
        return runtime, control

    def restart(self) -> None:
        asyncio.run(self.restart_async())

    async def restart_async(self) -> None:
        self.control.shutdown()
        self.runtime, self.control = self._boot()
        await rehydrate_root_state(self.runtime, self.control, cast(Any, self.resident))

    def materialize(self) -> ReplicaIncarnation:
        """Cold-start one replica of the family; its serve task is registered."""
        self.control.stores.families.register(_FAMILY)
        lifecycle = self.control._lifecycle
        return asyncio.run(lifecycle.materialize(_FAMILY))

    def serve(self, serve_task_id: str, worker_id: str = "wkr-1") -> None:
        """Dispatch the serve task and have it report its engine endpoint."""
        assert self.runtime.next_ready(threading.Event(), timeout=0.01) == serve_task_id
        record_dispatch(self.runtime, serve_task_id, worker_id)
        self.runtime.mark_started(serve_task_id, worker_id, {}, _TS)
        self.runtime.mark_updated(
            serve_task_id,
            worker_id,
            {
                "serve": {
                    "_host": "10.0.0.5",
                    "_port": 8001,
                    "_api_key": _ENGINE_KEY,
                    "model": "m",
                    "interface": "chat",
                }
            },
        )

    def warm(self) -> ReplicaIncarnation:
        replica = self.materialize()
        assert replica.serve_task_id is not None
        self.serve(replica.serve_task_id)
        self.control._promote_ready_replicas(_FAMILY.family)
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


def test_a_restart_reattaches_a_warm_replica_and_keeps_its_serve_task() -> None:
    node = _Node()
    warm = node.warm()
    assert warm.serve_task_id is not None

    node.restart()

    replica = node.replica(warm.replica_id)
    assert replica.state is ReplicaState.WARM
    assert replica.incarnation == warm.incarnation
    assert replica.endpoint is not None and replica.endpoint.api_key == _ENGINE_KEY
    assert node.status(warm.serve_task_id) == TaskStatus.DISPATCHED
    assert node.control.stores.pools.feasible_candidates(
        _FAMILY.family, AdmissionProfile(engine_batch_key=_FAMILY.engine_batch_key)
    )


def test_a_restart_invalidates_a_replica_whose_serve_task_was_requeued() -> None:
    node = _Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    # A drain gave the serve task up before the restart; its record keeps the endpoint
    # the gone dispatch reported.
    node.runtime.mark_cancelled(warm.serve_task_id, "wkr-1", {}, _TS)
    assert node.status(warm.serve_task_id) == TaskStatus.PENDING

    node.restart()

    assert node.replica(warm.replica_id).state is ReplicaState.PREEMPTED
    assert node.status(warm.serve_task_id) == TaskStatus.CANCELLED


def test_a_restart_preempts_a_replica_whose_serve_task_settled() -> None:
    node = _Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    node.runtime.mark_failed(
        warm.serve_task_id, "wkr-1", {}, _TS, error="engine exited"
    )

    node.restart()

    assert node.replica(warm.replica_id).state is ReplicaState.PREEMPTED


def test_a_restart_reaps_a_serve_task_no_replica_backs() -> None:
    node = _Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    # A replica invalidated while its serve task's record was absent left the task
    # running with nothing behind it.
    node.replica(warm.replica_id).state = ReplicaState.PREEMPTED
    node.persist()

    node.restart()

    assert node.status(warm.serve_task_id) == TaskStatus.CANCELLING


def test_a_restart_invalidates_a_cold_start_whose_serve_task_never_recorded() -> None:
    node = _Node()
    cold = node.materialize()
    serve_task_id = cold.serve_task_id
    assert serve_task_id is not None
    # A crash between registering the serve task and recording it on the replica.
    cold.serve_task_id = None
    node.persist()

    node.restart()

    assert node.replica(cold.replica_id).state is ReplicaState.PREEMPTED
    assert node.status(serve_task_id) == TaskStatus.CANCELLED
    assert node.control._lifecycle.plan_capacity(_FAMILY.family, "m").action == (
        "materialize"
    )


def test_a_restart_keeps_a_cold_start_whose_serve_task_waits_for_a_worker() -> None:
    node = _Node()
    cold = node.materialize()
    assert cold.serve_task_id is not None

    node.restart()

    assert node.replica(cold.replica_id).state is ReplicaState.MATERIALIZING
    assert node.status(cold.serve_task_id) == TaskStatus.PENDING


def test_reaping_a_serve_task_twice_cancels_it_once() -> None:
    node = _Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    cancels: list[str] = []
    cancel = node.runtime.cancel_workflow

    def counting(workflow_id: str, reason: str = "cancelled") -> list[str]:
        cancels.append(workflow_id)
        return cancel(workflow_id, reason)

    node.runtime.cancel_workflow = counting  # type: ignore[method-assign]
    node.control._lifecycle.on_preempt(warm.replica_id)
    node.control._lifecycle.reconcile_serve_tasks(frozenset({warm.serve_task_id}))

    assert len(cancels) == 1
    assert node.status(warm.serve_task_id) == TaskStatus.CANCELLING


def test_admission_waits_for_the_restored_replicas_to_reattach() -> None:
    node = _Node()
    warm = node.warm()
    node.persist()
    assert node.resident.blob is not None
    control = node._boot()[1]
    control.rehydrate(ResidentSnapshot.model_validate_json(node.resident.blob))
    (restored,) = control.stores.directory.all()
    # The restored endpoint carries no engine key until the replica is re-attached.
    assert restored.endpoint is not None and restored.endpoint.api_key is None

    async def run() -> None:
        gate = asyncio.create_task(control._replicas_attached.wait())
        await asyncio.sleep(0)
        assert not gate.done()
        control.reattach_replicas()
        await asyncio.wait_for(gate, timeout=1.0)

    asyncio.run(run())
    assert restored.replica_id == warm.replica_id


async def _until(predicate: Any, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


async def _admitted_boundary(node: _Node) -> ReplicaIncarnation:
    """Admit a resident agent boundary onto a replica cold-started for it."""
    node.control.bind_loop(asyncio.get_running_loop())
    _, ids = await _register(node.runtime, _RESIDENT_WF)
    _capture_resident_boundary(node.runtime, ids["writer"])
    directory = node.control.stores.directory
    await _until(lambda: any(r.serve_task_id for r in directory.all()))
    (replica,) = directory.all()
    assert replica.serve_task_id is not None
    node.serve(replica.serve_task_id, "wkr-2")
    await _until(lambda: "resident_handoff" in node.delivery.kinds())
    return replica


def _handoff_replicas(node: _Node) -> list[tuple[str, int]]:
    return [
        (p["handoff"]["replica_id"], p["handoff"]["incarnation"])
        for _w, kind, p in node.delivery.relays
        if kind == "resident_handoff"
    ]


def test_a_boundary_redriven_by_the_restart_resumes_on_the_reattached_replica() -> None:
    async def run() -> None:
        node = _Node()
        replica = await _admitted_boundary(node)

        await node.restart_async()
        await _until(lambda: bool(_handoff_replicas(node)))

        assert _handoff_replicas(node) == [(replica.replica_id, replica.incarnation)]
        assert len(node.control.stores.directory.all()) == 1
        assert node.replica(replica.replica_id).state is ReplicaState.WARM

    asyncio.run(run())


def test_a_boundary_redriven_by_the_restart_never_admits_on_a_dead_replica() -> None:
    async def run() -> None:
        node = _Node()
        replica = await _admitted_boundary(node)
        assert replica.serve_task_id is not None
        node.runtime.mark_failed(
            replica.serve_task_id, "wkr-2", {}, _TS, error="engine exited"
        )

        await node.restart_async()
        await asyncio.sleep(0.2)

        assert _handoff_replicas(node) == []
        assert node.replica(replica.replica_id).state is ReplicaState.PREEMPTED

    asyncio.run(run())
