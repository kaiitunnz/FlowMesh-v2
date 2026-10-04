"""A root restart re-attaches each live resident replica and reaps what backs nothing.

Each scenario runs the real startup order over a real task runtime and the wired
resident control, restarting both from the durable state the first run persisted.
"""

import asyncio
from collections.abc import Set
from typing import Any

from server.resident import ReplicaState
from server.resident.state import (
    AdmissionProfile,
    ClaimState,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
)
from server.task.models import TaskStatus
from tests.server.resident.node_harness import (
    ENGINE_KEY,
    FAMILY,
    TS,
    Node,
    admitted_boundary,
    handoff_replicas,
)
from tests.server.task.test_resident_origin_loss import (
    _RESIDENT_WF,
    _capture_resident_boundary,
)
from tests.server.task.test_v2_orchestration import _register
from tests.support.waiting import until


def test_a_restart_reattaches_a_warm_replica_and_keeps_its_serve_task() -> None:
    node = Node()
    warm = node.warm()
    assert warm.serve_task_id is not None

    node.restart()

    replica = node.replica(warm.replica_id)
    assert replica.state is ReplicaState.WARM
    assert replica.incarnation == warm.incarnation
    assert replica.endpoint is not None and replica.endpoint.api_key is None
    assert node.status(warm.serve_task_id) == TaskStatus.DISPATCHED
    record = node.runtime.get_record(warm.serve_task_id)
    assert record is not None and record.dispatch_id is not None
    assert record.latest_update_dispatch_id == record.dispatch_id
    assert node.control.stores.pools.feasible_candidates(
        FAMILY.family, AdmissionProfile(engine_batch_key=FAMILY.engine_batch_key)
    )


def test_a_restart_invalidates_a_replica_whose_serve_task_was_requeued() -> None:
    node = Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    # A drain gave the serve task up before the restart; its record keeps the endpoint
    # the gone dispatch reported.
    node.runtime.mark_cancelled(warm.serve_task_id, "wkr-1", {}, TS)
    assert node.status(warm.serve_task_id) == TaskStatus.PENDING

    node.restart()

    assert node.replica(warm.replica_id).state is ReplicaState.PREEMPTED
    assert node.status(warm.serve_task_id) == TaskStatus.CANCELLED


def test_a_restart_preempts_a_replica_whose_serve_task_settled() -> None:
    node = Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    node.runtime.mark_failed(warm.serve_task_id, "wkr-1", {}, TS, error="engine exited")

    node.restart()

    assert node.replica(warm.replica_id).state is ReplicaState.PREEMPTED


def test_a_restart_reaps_a_serve_task_no_replica_backs() -> None:
    node = Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    # A replica invalidated while its serve task's record was absent left the task
    # running with nothing behind it.
    node.replica(warm.replica_id).state = ReplicaState.PREEMPTED
    node.persist()

    node.restart()

    assert node.status(warm.serve_task_id) == TaskStatus.CANCELLING


def test_a_restart_invalidates_a_cold_start_whose_serve_task_never_recorded() -> None:
    node = Node()
    cold = node.materialize()
    serve_task_id = cold.serve_task_id
    assert serve_task_id is not None
    # A crash between registering the serve task and recording it on the replica.
    cold.serve_task_id = None
    node.persist()

    node.restart()

    assert node.replica(cold.replica_id).state is ReplicaState.PREEMPTED
    assert node.status(serve_task_id) == TaskStatus.CANCELLED
    assert node.control._lifecycle.plan_capacity(FAMILY.family, "m").action == (
        "materialize"
    )


def test_a_restart_keeps_a_cold_start_whose_serve_task_waits_for_a_worker() -> None:
    node = Node()
    cold = node.materialize()
    assert cold.serve_task_id is not None

    node.restart()

    assert node.replica(cold.replica_id).state is ReplicaState.MATERIALIZING
    assert node.status(cold.serve_task_id) == TaskStatus.PENDING


def test_reaping_a_serve_task_twice_cancels_it_once() -> None:
    node = Node()
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


def test_a_restart_keeps_a_draining_replica_serve_task_while_it_holds_credit() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        assert replica.serve_task_id is not None
        node.control._lifecycle.drain(replica.replica_id)
        node.persist()

        await node.restart_async()

        assert node.replica(replica.replica_id).state is ReplicaState.DRAINING
        assert node.status(replica.serve_task_id) == TaskStatus.DISPATCHED

    asyncio.run(run())


def test_a_restart_stops_a_draining_replica_holding_no_credit() -> None:
    node = Node()
    warm = node.warm()
    assert warm.serve_task_id is not None
    node.control._lifecycle.drain(warm.replica_id)
    node.persist()

    node.restart()

    assert node.replica(warm.replica_id).state is ReplicaState.STOPPED
    assert node.status(warm.serve_task_id) == TaskStatus.CANCELLING


def test_a_restart_reattaches_a_standing_replica() -> None:
    node = Node()
    serve_task_id = asyncio.run(node.submit_serve_async())
    node.serve(serve_task_id)
    standing = node.adopt_standing(serve_task_id)
    node.persist()

    node.restart()

    replica = node.replica(standing.replica_id)
    assert replica.state is ReplicaState.WARM
    assert replica.endpoint is not None and replica.endpoint.api_key is None


def test_control_never_reads_an_engine_key_an_earlier_worker_reported() -> None:
    node = Node()
    serve_task_id = asyncio.run(node.submit_serve_async())
    node.serve(serve_task_id, reported_key=ENGINE_KEY)
    standing = node.adopt_standing(serve_task_id)
    assert standing.endpoint is not None and standing.endpoint.api_key is None
    node.persist()

    node.restart()

    replica = node.replica(standing.replica_id)
    assert replica.endpoint is not None and replica.endpoint.api_key is None


def test_a_restart_never_reattaches_an_endpoint_an_earlier_dispatch_reported() -> None:
    node = Node()
    warm = node.warm()
    serve_task_id = warm.serve_task_id
    assert serve_task_id is not None
    # The serve task is given up and dispatched again; the new dispatch has not
    # reported its endpoint when the root restarts.
    node.runtime.mark_cancelled(serve_task_id, "wkr-1", {}, TS)
    node.dispatch(serve_task_id, "wkr-2", "dsp-next")
    node.persist()

    node.restart()

    assert node.replica(warm.replica_id).state is ReplicaState.PREEMPTED


def test_a_restart_without_a_snapshot_keeps_a_cold_start_the_redrive_began() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        # No resident state survived; the runtime still re-drives the boundary, which
        # cold-starts a replica afresh.
        node.resident.blob = None
        relays_at_reattach: list[list[Any]] = []

        def on_boot(booted: Node) -> None:
            _observe_reattach(booted, relays_at_reattach)
            # The cold start's serve task is registered only after the re-attach, so a
            # re-attach that ran mid cold start finds it with no serve task yet.
            lifecycle = booted.control._lifecycle
            materialize = lifecycle._materialize_fn
            assert materialize is not None
            reattach = booted.control.reattach_replicas
            reattached = asyncio.Event()

            async def after_reattach(*args: Any) -> str:
                await reattached.wait()
                return await materialize(*args)

            def reattach_then_release(live_serve_tasks: Set[str]) -> None:
                reattach(live_serve_tasks)
                reattached.set()

            lifecycle._materialize_fn = after_reattach
            booted.control.reattach_replicas = reattach_then_release  # type: ignore[method-assign]

        await node.restart_async(on_boot)
        directory = node.control.stores.directory
        await until(lambda: any(r.serve_task_id for r in directory.all()))

        (cold,) = directory.all()
        assert cold.replica_id != replica.replica_id
        assert cold.state is ReplicaState.MATERIALIZING
        assert relays_at_reattach == [[]]

    asyncio.run(run())


class _ObservedEvent(asyncio.Event):
    """Records that a waiter parked on it while it was clear."""

    parked = False

    async def wait(self) -> Any:
        if not self.is_set():
            self.parked = True
        return await super().wait()


def _observe_reattach(node: Node, relays_at_reattach: list[list[Any]]) -> None:
    """Hold the startup's re-attach until the re-driven boundary has parked on the
    admission gate or acted past it, and record what it relayed by then."""
    gate = _ObservedEvent()
    gate.set()
    node.control._replicas_attached = gate
    rehydrate = node.runtime.rehydrate
    reattach = node.control.reattach_replicas

    async def rehydrated() -> int:
        directory = node.control.stores.directory
        restored = {replica.replica_id for replica in directory.all()}
        rehydrated_count = await rehydrate()
        await until(
            lambda: gate.parked
            or bool(node.delivery.relays)
            or any(r.replica_id not in restored for r in directory.all())
        )
        return rehydrated_count

    def reattached(live_serve_tasks: Set[str]) -> None:
        relays_at_reattach.append(list(node.delivery.relays))
        reattach(live_serve_tasks)

    node.runtime.rehydrate = rehydrated  # type: ignore[method-assign]
    node.control.reattach_replicas = reattached  # type: ignore[method-assign]


def test_a_boundary_redriven_by_the_restart_resumes_on_the_reattached_replica() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        relays_at_reattach: list[list[Any]] = []

        await node.restart_async(
            lambda booted: _observe_reattach(booted, relays_at_reattach)
        )
        await until(lambda: bool(handoff_replicas(node)))

        assert relays_at_reattach == [[]]
        assert handoff_replicas(node) == [(replica.replica_id, replica.incarnation)]
        binds = [p for _w, k, p in node.delivery.relays if k == "resident_sidecar_bind"]
        assert binds and all(b["engine"]["api_key"] is None for b in binds)
        assert len(node.control.stores.directory.all()) == 1

    asyncio.run(run())


def test_a_boundary_redriven_by_the_restart_never_admits_on_a_dead_replica() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        assert replica.serve_task_id is not None
        node.runtime.mark_failed(
            replica.serve_task_id, "wkr-2", {}, TS, error="engine exited"
        )
        relays_at_reattach: list[list[Any]] = []

        await node.restart_async(
            lambda booted: _observe_reattach(booted, relays_at_reattach)
        )
        await until(
            lambda: node.replica(replica.replica_id).state is ReplicaState.PREEMPTED
        )

        assert relays_at_reattach == [[]]
        assert replica.replica_id not in {r for r, _ in handoff_replicas(node)}

    asyncio.run(run())


def test_a_restart_never_reattaches_an_endpoint_no_dispatch_reported() -> None:
    node = Node()
    warm = node.warm()
    serve_task_id = warm.serve_task_id
    assert serve_task_id is not None
    record = node.runtime.get_record(serve_task_id)
    assert record is not None
    record.latest_update_dispatch_id = None
    with node.runtime._lock:
        node.runtime._persist_locked(serve_task_id)

    node.restart()

    assert node.replica(warm.replica_id).state is ReplicaState.PREEMPTED


def test_a_claim_pending_at_a_restart_expires_and_frees_its_family_to_yield() -> None:
    async def run() -> None:
        node = Node()
        warm = await node.warm_async()
        assert warm.serve_task_id is not None
        # A claim whose workflow was cancelled while it waited, before it settled.
        pending = node.control._admission.raise_claim(
            invocation_id="inv-pending",
            subject=InvocationSubject(
                kind=InvocationSubjectKind.WORKFLOW, id="wfl-gone", tenant="org"
            ),
            family=FAMILY.family,
            profile=AdmissionProfile(engine_batch_key=FAMILY.engine_batch_key),
        )
        node.persist()

        await node.restart_async()

        restored = node.control.stores.claims.get(pending.claim_id)
        assert restored is not None
        assert restored.terminal_reason is ClaimTerminalReason.EXPIRED
        node.control._lifecycle.yield_serve_task(warm.serve_task_id)
        assert node.replica(warm.replica_id).state is ReplicaState.STOPPED

    asyncio.run(run())


def test_a_boundary_pending_at_a_restart_admits_on_a_successor_claim() -> None:
    async def run() -> None:
        node = Node()
        node.control.bind_loop(asyncio.get_running_loop())
        _, ids = await _register(node.runtime, _RESIDENT_WF)
        _capture_resident_boundary(node.runtime, ids["writer"])
        directory = node.control.stores.directory
        await until(lambda: any(r.serve_task_id for r in directory.all()))
        (cold,) = directory.all()
        assert cold.serve_task_id is not None
        (first,) = node.control.stores.claims.all()
        assert first.state is ClaimState.PENDING

        await node.restart_async()
        claims = node.control.stores.claims
        await until(lambda: len(claims.all()) == 2)
        node.serve(cold.serve_task_id, "wkr-2")
        await until(lambda: "resident_handoff" in node.delivery.kinds())

        restored = claims.get(first.claim_id)
        assert restored is not None
        assert restored.terminal_reason is ClaimTerminalReason.EXPIRED
        (successor,) = [c for c in claims.all() if c.claim_id != first.claim_id]
        assert successor.invocation_id == first.invocation_id
        assert successor.admission_epoch > first.admission_epoch
        assert successor.holds_credit
        assert handoff_replicas(node) == [(cold.replica_id, cold.incarnation)]

    asyncio.run(run())
