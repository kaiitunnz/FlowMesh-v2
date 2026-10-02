"""A demand replica retires when its serve task stops serving.

Each scenario drives the real task runtime from another thread, as the event monitor
and watchdog do, against the resident control ``build_resident_capacity`` wires to it.
"""

import asyncio
import threading
from typing import Any

from server.resident import ReplicaState
from server.resident.state import ClaimState, ReplicaIncarnation
from server.task.models import TERMINAL_TASK_STATUSES, TaskStatus
from tests.server.dispatch_helpers import record_dispatch
from tests.server.resident.test_restart_reattach import (
    _FAMILY,
    _TS,
    _admitted_boundary,
    _Node,
    _until,
)


def _loop_bound(node: _Node) -> None:
    node.control.bind_loop(asyncio.get_running_loop())


async def _serving(node: _Node) -> ReplicaIncarnation:
    replica = await node.warm_async()
    assert replica.serve_task_id is not None
    return replica


async def _off_loop(fn: Any, *args: Any) -> None:
    await asyncio.to_thread(fn, *args)


def _lose_worker(node: _Node, worker_id: str) -> None:
    """Recover a departed worker's tasks and fail each it held, as the watchdog does."""
    recovery = node.runtime.recover_tasks_for_worker(worker_id, spend_attempt=True)
    for task_id in recovery.lost:
        node.runtime.mark_failed(
            task_id,
            worker_id,
            {"reason": "worker_heartbeat_expired", "synthetic": True},
            _TS,
            error="worker_heartbeat_expired",
        )


def test_a_lost_worker_retires_its_replica_on_the_control_loop() -> None:
    async def run() -> None:
        node = _Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        loop_thread = threading.get_ident()
        preempted_on: list[int] = []
        preempt = node.control._lifecycle.on_preempt

        def recording(replica_id: str) -> None:
            preempted_on.append(threading.get_ident())
            preempt(replica_id)

        node.control._lifecycle.on_preempt = recording  # type: ignore[method-assign]

        await _off_loop(_lose_worker, node, "wkr-1")
        await _until(lambda: replica.state is ReplicaState.PREEMPTED)

        assert preempted_on == [loop_thread]
        assert node.status(replica.serve_task_id) in TERMINAL_TASK_STATUSES
        assert node.runtime.ready_queue_length() == 0

    asyncio.run(run())


def test_a_drained_worker_retires_its_replica_and_nothing_reruns_it() -> None:
    async def run() -> None:
        node = _Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None

        await _off_loop(
            node.runtime.mark_cancelled, replica.serve_task_id, "wkr-1", {}, _TS
        )
        await _until(lambda: replica.state is ReplicaState.PREEMPTED)

        assert node.status(replica.serve_task_id) == TaskStatus.CANCELLED
        assert node.runtime.ready_queue_length() == 0

    asyncio.run(run())


def test_a_serve_task_end_reported_twice_invalidates_its_replica_once() -> None:
    async def run() -> None:
        node = _Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        incarnation = replica.incarnation

        # The requeue reports the end; the reap that follows settles the task, which
        # reports it again.
        await _off_loop(
            node.runtime.mark_cancelled, replica.serve_task_id, "wkr-1", {}, _TS
        )
        await _until(lambda: node.status(str(replica.serve_task_id)) == "CANCELLED")
        node.control.on_serve_task_end(replica.serve_task_id)
        await asyncio.sleep(0.05)

        assert replica.state is ReplicaState.PREEMPTED
        assert replica.incarnation == incarnation + 1

    asyncio.run(run())


def test_an_idle_retire_ends_stopped_rather_than_preempted() -> None:
    async def run() -> None:
        node = _Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        lifecycle = node.control._lifecycle
        lifecycle._idle_retain_sec = 1.0
        replica.last_active_at = "2000-01-01T00:00:00+00:00"

        lifecycle.sweep_idle()
        assert replica.state is ReplicaState.DRAINING
        lifecycle.sweep_idle()
        await asyncio.sleep(0.05)

        assert replica.state is ReplicaState.STOPPED
        assert node.status(replica.serve_task_id) == TaskStatus.CANCELLING

    asyncio.run(run())


def test_a_draining_replica_whose_serve_task_is_given_up_keeps_retiring() -> None:
    async def run() -> None:
        node = _Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        node.control._lifecycle.drain(replica.replica_id)

        await _off_loop(
            node.runtime.mark_cancelled, replica.serve_task_id, "wkr-1", {}, _TS
        )
        await _until(lambda: node.status(str(replica.serve_task_id)) == "CANCELLED")

        assert replica.state is ReplicaState.DRAINING
        assert node.runtime.ready_queue_length() == 0

    asyncio.run(run())


def test_a_cold_start_whose_serve_task_fails_before_reporting_is_invalidated() -> None:
    async def run() -> None:
        node = _Node()
        _loop_bound(node)
        cold = await node.materialize_async()
        serve_task_id = cold.serve_task_id
        assert serve_task_id is not None
        assert node.runtime.next_ready(threading.Event(), timeout=0.01) == serve_task_id
        record_dispatch(node.runtime, serve_task_id, "wkr-1")

        await _off_loop(
            lambda: node.runtime.mark_failed(
                serve_task_id, "wkr-1", {}, _TS, error="engine exited"
            )
        )
        await _until(lambda: cold.state is ReplicaState.PREEMPTED)

        plan = node.control._lifecycle.plan_capacity(_FAMILY.family, "m")
        assert plan.action == "materialize"

    asyncio.run(run())


def test_retiring_a_replica_leaves_its_admitted_credit_held() -> None:
    async def run() -> None:
        node = _Node()
        replica = await _admitted_boundary(node)
        assert replica.serve_task_id is not None
        (claim,) = node.control.stores.claims.all()
        assert claim.holds_credit

        await _off_loop(_lose_worker, node, "wkr-2")
        await _until(lambda: replica.state is ReplicaState.PREEMPTED)

        assert claim.state is not ClaimState.TERMINAL
        assert claim.holds_credit
        assert node.control.stores.credit_ledger.held(replica.replica_id) == 1

    asyncio.run(run())
