"""A demand replica retires when its serve task stops serving.

Each scenario drives the real task runtime from another thread, as the event monitor
and watchdog do, against the resident control ``build_resident_capacity`` wires to it.
"""

import asyncio
import contextlib
import threading
from typing import Any

import pytest

from server.resident import ReplicaState, materializer
from server.resident.state import (
    AdmissionProfile,
    ClaimState,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ReplicaIncarnation,
    ServiceClaim,
)
from server.task.models import TERMINAL_TASK_STATUSES, TaskStatus
from tests.server.resident.node_harness import FAMILY, TS, Node, admitted_boundary
from tests.server.task.test_resident_origin_loss import (
    _RESIDENT_WF,
    _capture_resident_boundary,
)
from tests.server.task.test_v2_orchestration import _register
from tests.support.waiting import until

_PROFILE = AdmissionProfile(engine_batch_key=FAMILY.engine_batch_key)


def _loop_bound(node: Node) -> None:
    node.control.bind_loop(asyncio.get_running_loop())


async def _serving(node: Node) -> ReplicaIncarnation:
    replica = await node.warm_async()
    assert replica.serve_task_id is not None
    return replica


async def _off_loop(fn: Any, *args: Any) -> None:
    await asyncio.to_thread(fn, *args)


async def _ask_yield(node: Node, serve_task_id: str) -> None:
    """Ask resident capacity to free the worker the serve task's dispatch occupies."""
    record = node.runtime.get_record(serve_task_id)
    assert record is not None and record.dispatch_id is not None
    await _off_loop(
        node.runtime.request_resident_yield, serve_task_id, record.dispatch_id
    )
    await asyncio.sleep(0)


def _lose_worker(node: Node, worker_id: str) -> None:
    """Recover a departed worker's tasks and fail each it held, as the watchdog does."""
    recovery = node.runtime.recover_tasks_for_worker(worker_id, spend_attempt=True)
    for task_id in recovery.lost:
        node.runtime.mark_failed(
            task_id,
            worker_id,
            {"reason": "worker_heartbeat_expired", "synthetic": True},
            TS,
            error="worker_heartbeat_expired",
        )


def test_a_lost_worker_retires_its_replica_on_the_control_loop() -> None:
    async def run() -> None:
        node = Node()
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
        await until(lambda: replica.state is ReplicaState.PREEMPTED)

        assert preempted_on == [loop_thread]
        assert node.status(replica.serve_task_id) in TERMINAL_TASK_STATUSES
        assert node.runtime.ready_queue_length() == 0

    asyncio.run(run())


def test_a_drained_worker_retires_its_replica_and_nothing_reruns_it() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None

        await _off_loop(
            node.runtime.mark_cancelled, replica.serve_task_id, "wkr-1", {}, TS
        )
        await until(lambda: replica.state is ReplicaState.PREEMPTED)

        assert node.status(replica.serve_task_id) == TaskStatus.CANCELLED
        assert node.runtime.ready_queue_length() == 0

    asyncio.run(run())


def test_a_serve_task_end_reported_twice_invalidates_its_replica_once() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        incarnation = replica.incarnation

        # The requeue reports the end; the reap that follows settles the task, which
        # reports it again.
        await _off_loop(
            node.runtime.mark_cancelled, replica.serve_task_id, "wkr-1", {}, TS
        )
        await until(lambda: node.status(str(replica.serve_task_id)) == "CANCELLED")
        node.control.on_serve_task_end(replica.serve_task_id)
        await asyncio.sleep(0)

        assert replica.state is ReplicaState.PREEMPTED
        assert replica.incarnation == incarnation + 1

    asyncio.run(run())


def test_an_idle_retire_ends_stopped_rather_than_preempted() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        lifecycle = node.control._lifecycle
        lifecycle._idle_retain_sec = 1.0
        replica.last_active_at = "2000-01-01T00:00:00+00:00"

        lifecycle.sweep_idle()
        assert replica.state is ReplicaState.DRAINING
        lifecycle.sweep_idle()
        await asyncio.sleep(0)

        assert replica.state is ReplicaState.STOPPED
        assert node.status(replica.serve_task_id) == TaskStatus.CANCELLING

    asyncio.run(run())


def test_a_draining_replica_whose_serve_task_is_given_up_keeps_retiring() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        node.control._lifecycle.drain(replica.replica_id)

        await _off_loop(
            node.runtime.mark_cancelled, replica.serve_task_id, "wkr-1", {}, TS
        )
        await until(lambda: node.status(str(replica.serve_task_id)) == "CANCELLED")

        assert replica.state is ReplicaState.DRAINING
        assert node.runtime.ready_queue_length() == 0

    asyncio.run(run())


def test_a_cold_start_whose_serve_task_fails_before_reporting_is_invalidated() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        cold = await node.materialize_async()
        serve_task_id = cold.serve_task_id
        assert serve_task_id is not None
        node.dispatch(serve_task_id, "wkr-1")

        await _off_loop(
            lambda: node.runtime.mark_failed(
                serve_task_id, "wkr-1", {}, TS, error="engine exited"
            )
        )
        await until(lambda: cold.state is ReplicaState.PREEMPTED)

        plan = node.control._lifecycle.plan_capacity(FAMILY.family, "m")
        assert plan.action == "materialize"

    asyncio.run(run())


def test_retiring_a_replica_leaves_its_admitted_credit_held() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        assert replica.serve_task_id is not None
        (claim,) = node.control.stores.claims.all()
        assert claim.holds_credit

        await _off_loop(_lose_worker, node, "wkr-2")
        await until(lambda: replica.state is ReplicaState.PREEMPTED)

        assert claim.state is not ClaimState.TERMINAL
        assert claim.holds_credit
        assert node.control.stores.credit_ledger.held(replica.replica_id) == 1

    asyncio.run(run())


def test_a_redispatch_behind_a_held_write_still_retires_the_replica() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        replica = await _serving(node)
        serve_task_id = replica.serve_task_id
        assert serve_task_id is not None
        commit = node.tasks.commit_transition
        failures = [RuntimeError("control store unavailable")]

        def flaky(*args: Any, **kwargs: Any) -> Any:
            if failures:
                raise failures.pop()
            return commit(*args, **kwargs)

        node.tasks.commit_transition = flaky  # type: ignore[method-assign]

        def give_up() -> None:
            # The give-up's commit fails and is held for the report's redelivery, so
            # the requeue is in memory only.
            with contextlib.suppress(RuntimeError):
                node.runtime.mark_cancelled(serve_task_id, "wkr-1", {}, TS)

        await _off_loop(give_up)
        assert node.status(serve_task_id) == TaskStatus.PENDING
        await asyncio.sleep(0)
        assert replica.state is ReplicaState.WARM

        def redispatch() -> None:
            node.dispatch(serve_task_id, "wkr-2", "dsp-next")

        await _off_loop(redispatch)
        await until(lambda: replica.state is ReplicaState.PREEMPTED)

    asyncio.run(run())


def _retryable_failure(node: Node, task_id: str, worker_id: str, dispatch_id: str):
    return node.runtime.fail_dispatch(
        task_id,
        worker_id,
        {},
        TS,
        dispatch_id,
        error="not enough GPU memory to load the model",
        retryable=True,
    )


def test_a_retryable_cold_start_failure_retries_the_same_task_elsewhere() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        cold = await node.materialize_async()
        serve_task_id = cold.serve_task_id
        assert serve_task_id is not None
        node.dispatch(serve_task_id, "wkr-1", "dsp-a")

        await _off_loop(_retryable_failure, node, serve_task_id, "wkr-1", "dsp-a")
        await asyncio.sleep(0)

        assert cold.state is ReplicaState.MATERIALIZING
        assert node.status(serve_task_id) == TaskStatus.PENDING
        record = node.runtime.get_record(serve_task_id)
        assert record is not None and record.failed_workers == ["wkr-1"]

        node.serve(serve_task_id, "wkr-2", "dsp-b", port=8002)
        node.control._promote_ready_replicas(FAMILY.family)

        assert cold.state is ReplicaState.WARM
        assert cold.endpoint is not None
        assert record.latest_update_dispatch_id == "dsp-b"

    asyncio.run(run())


def test_an_earlier_dispatch_endpoint_never_promotes_a_cold_start() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        cold = await node.materialize_async()
        serve_task_id = cold.serve_task_id
        assert serve_task_id is not None
        # The first dispatch reports its endpoint, then fails before promotion.
        node.serve(serve_task_id, "wkr-1", "dsp-a")
        await _off_loop(_retryable_failure, node, serve_task_id, "wkr-1", "dsp-a")
        node.dispatch(serve_task_id, "wkr-2", "dsp-b")

        node.control._promote_ready_replicas(FAMILY.family)

        assert cold.state is ReplicaState.MATERIALIZING
        assert node.control.probe_serve_endpoint(serve_task_id) is None

    asyncio.run(run())


def test_a_cancelled_cold_start_is_invalidated() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        cold = await node.materialize_async()
        serve_task_id = cold.serve_task_id
        assert serve_task_id is not None
        node.dispatch(serve_task_id, "wkr-1")
        record = node.runtime.get_record(serve_task_id)
        assert record is not None

        await _off_loop(node.runtime.cancel_workflow, record.workflow_id)
        await until(lambda: cold.state is ReplicaState.PREEMPTED)

    asyncio.run(run())


def test_a_cold_start_whose_end_went_unreported_frees_the_family() -> None:
    async def run() -> None:
        node = Node()
        cold = await node.materialize_async()
        serve_task_id = cold.serve_task_id
        assert serve_task_id is not None
        node.dispatch(serve_task_id, "wkr-1")
        # The task ends before the control loop is bound, so no report reaches it.
        node.runtime.mark_failed(serve_task_id, "wkr-1", {}, TS, error="engine exited")
        assert cold.state is ReplicaState.MATERIALIZING

        node.control._promote_ready_replicas(FAMILY.family)

        assert cold.state is ReplicaState.PREEMPTED
        assert not node.control._has_materializing(FAMILY.family)

    asyncio.run(run())


@pytest.mark.parametrize("draining", [False, True])
def test_a_standing_replica_ignores_its_serve_task_end(draining: bool) -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        serve_task_id = await node.submit_serve_async()
        node.serve(serve_task_id)
        standing = node.adopt_standing(serve_task_id)
        if draining:
            node.control._lifecycle.drain(standing.replica_id)
        state = standing.state

        node.control.on_serve_task_end(serve_task_id)
        await asyncio.sleep(0)

        assert standing.state is state
        assert node.status(serve_task_id) == TaskStatus.DISPATCHED

    asyncio.run(run())


def test_a_requeued_cold_start_that_never_redispatches_ends_at_its_deadline() -> None:
    async def run() -> None:
        node = Node(cold_start_deadline_sec=0.3)
        node.control.bind_loop(asyncio.get_running_loop())
        _, ids = await _register(node.runtime, _RESIDENT_WF)
        _capture_resident_boundary(node.runtime, ids["writer"])
        directory = node.control.stores.directory
        await until(lambda: any(r.serve_task_id for r in directory.all()))
        (cold,) = directory.all()
        serve_task_id = cold.serve_task_id
        assert serve_task_id is not None
        node.dispatch(serve_task_id, "wkr-1", "dsp-a")
        await _off_loop(_retryable_failure, node, serve_task_id, "wkr-1", "dsp-a")

        (claim,) = node.control.stores.claims.all()
        await until(lambda: claim.state is ClaimState.TERMINAL, timeout=5.0)

        assert not claim.holds_credit
        assert "resident_handoff" not in node.delivery.kinds()

    asyncio.run(run())


@pytest.mark.parametrize(
    "error", [RuntimeError("resource registry unavailable"), asyncio.CancelledError()]
)
def test_a_cold_start_whose_registrar_raises_leaves_no_serve_task(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    async def failing_registrar(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(materializer, "register_resource", failing_registrar)

    async def run() -> None:
        node = Node()
        _loop_bound(node)
        with pytest.raises(type(error)):
            await node.materialize_async()

        (replica,) = node.control.stores.directory.all()
        assert replica.state is ReplicaState.PREEMPTED
        assert node.runtime.live_resident_task_ids() == set()
        assert not node.control._has_materializing(FAMILY.family)

    asyncio.run(run())


def test_a_yield_request_retires_an_idle_demand_replica() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None

        await _ask_yield(node, replica.serve_task_id)

        assert replica.state is ReplicaState.STOPPED
        assert node.status(replica.serve_task_id) == TaskStatus.CANCELLING

    asyncio.run(run())


def test_a_yield_request_waits_for_the_replica_credit_to_release() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        assert replica.serve_task_id is not None
        (claim,) = node.control.stores.claims.all()

        await _ask_yield(node, replica.serve_task_id)
        assert replica.state is ReplicaState.WARM
        assert claim.holds_credit

        node.control.on_invocation_terminal(claim.invocation_id)
        await asyncio.sleep(0)
        await _ask_yield(node, replica.serve_task_id)

        assert replica.state is ReplicaState.STOPPED

    asyncio.run(run())


def test_a_yield_request_leaves_a_standing_replica() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        serve_task_id = await node.submit_serve_async()
        node.serve(serve_task_id)
        standing = node.adopt_standing(serve_task_id)

        node.control.on_yield_requested(serve_task_id)
        await asyncio.sleep(0)

        assert standing.state is ReplicaState.WARM
        assert node.status(serve_task_id) == TaskStatus.DISPATCHED

    asyncio.run(run())


def _queue_claim(node: Node) -> ServiceClaim:
    """A claim of the family raised and not yet polled onto a replica."""
    return node.control._admission.raise_claim(
        invocation_id="inv-queued",
        subject=InvocationSubject(
            kind=InvocationSubjectKind.WORKFLOW, id="wfl-queued", tenant="org"
        ),
        family=FAMILY.family,
        profile=_PROFILE,
    )


@pytest.mark.parametrize("awaited", [False, True])
def test_a_yield_request_retires_a_cold_start_only_once_no_claim_awaits_it(
    awaited: bool,
) -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        cold = await node.materialize_async()
        assert cold.serve_task_id is not None
        node.dispatch(cold.serve_task_id, "wkr-holder")
        if awaited:
            _queue_claim(node)

        await _ask_yield(node, cold.serve_task_id)

        if awaited:
            assert cold.state is ReplicaState.MATERIALIZING
            assert node.status(cold.serve_task_id) == TaskStatus.DISPATCHED
        else:
            assert cold.state is ReplicaState.PREEMPTED
            assert node.status(cold.serve_task_id) == TaskStatus.CANCELLING

    asyncio.run(run())


def test_a_yield_request_leaves_a_replica_a_queued_claim_will_join() -> None:
    async def run() -> None:
        node = Node()
        _loop_bound(node)
        replica = await _serving(node)
        assert replica.serve_task_id is not None
        claim = _queue_claim(node)

        await _ask_yield(node, replica.serve_task_id)
        assert replica.state is ReplicaState.WARM

        admission = node.control._admission
        assert admission.admit(claim, _PROFILE, idempotency_key=None) is not None
        admission.settle_invocation_terminal(
            claim.invocation_id, ClaimTerminalReason.COMPLETED
        )
        await _ask_yield(node, replica.serve_task_id)

        assert replica.state is ReplicaState.STOPPED

    asyncio.run(run())


def test_a_released_claim_restarts_its_replica_retain_window() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        (claim,) = node.control.stores.claims.all()
        lifecycle = node.control._lifecycle
        lifecycle._idle_retain_sec = 60.0
        # The call outlasted the retain window before its credit released.
        replica.last_active_at = "2000-01-01T00:00:00+00:00"
        lifecycle.sweep_idle()
        assert replica.state is ReplicaState.WARM

        node.control.on_invocation_terminal(claim.invocation_id)
        await asyncio.sleep(0)
        lifecycle.sweep_idle()

        assert not claim.holds_credit
        assert replica.state is ReplicaState.WARM

    asyncio.run(run())


def test_a_root_rewrite_keeps_an_earlier_dispatch_endpoint_fenced() -> None:
    async def run() -> None:
        node = Node()
        serve_task_id = await node.submit_serve_async()
        node.serve(serve_task_id, "wkr-1", "dsp-a")
        node.runtime.mark_cancelled(serve_task_id, "wkr-1", {}, TS, "dsp-a")
        node.dispatch(serve_task_id, "wkr-2", "dsp-b")
        record = node.runtime.get_record(serve_task_id)
        assert record is not None and record.latest_update is not None

        assert node.runtime.rewrite_update(
            serve_task_id, record.latest_update, dict(record.latest_update)
        )

        assert node.control.probe_serve_endpoint(serve_task_id) is None

    asyncio.run(run())


def test_a_root_rewrite_never_relabels_a_newer_worker_update() -> None:
    async def run() -> None:
        node = Node()
        serve_task_id = await node.submit_serve_async()
        node.serve(serve_task_id, "wkr-1", "dsp-a")
        record = node.runtime.get_record(serve_task_id)
        assert record is not None and record.latest_update is not None
        read = record.latest_update
        newer = {"serve": {**read["serve"], "_socket": "/run/newer.sock"}}
        node.runtime.mark_updated(serve_task_id, "wkr-1", newer, "dsp-a")

        assert not node.runtime.rewrite_update(serve_task_id, read, dict(read))

        assert record.latest_update is newer
        assert record.latest_update_dispatch_id == "dsp-a"

    asyncio.run(run())
