"""Cold starts promote when their serve task reports, and a cordon drains the demand
replicas on its worker.

Each scenario drives the real task runtime against the resident control
``build_resident_capacity`` wires to it.
"""

import asyncio

from server.resident import ReplicaState
from server.resident.state import AdmissionProfile, ClaimState
from server.task.models import TaskStatus
from tests.server.resident.node_harness import (
    FAMILY,
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

# Every replica's serve task runs on this worker in the harness delivery.
_SERVE_WORKER = "wkr-replica"


def test_a_cold_start_whose_claim_ended_promotes_and_serves_the_next_claim() -> None:
    async def run() -> None:
        node = Node()
        node.control.bind_loop(asyncio.get_running_loop())
        _, ids = await _register(node.runtime, _RESIDENT_WF)
        _capture_resident_boundary(node.runtime, ids["writer"])
        directory = node.control.stores.directory
        await until(lambda: any(r.serve_task_id for r in directory.all()))
        (cold,) = directory.all()
        assert cold.serve_task_id is not None
        writer = node.runtime.get_record(ids["writer"])
        assert writer is not None
        node.runtime.cancel_workflow(writer.workflow_id)
        await until(
            lambda: all(
                c.state is ClaimState.TERMINAL for c in node.control.stores.claims.all()
            )
        )

        node.serve(cold.serve_task_id, "wkr-2")
        await until(lambda: cold.state is ReplicaState.WARM)

        _, again = await _register(node.runtime, _RESIDENT_WF)
        _capture_resident_boundary(node.runtime, again["writer"])
        await until(lambda: "resident_handoff" in node.delivery.kinds())
        handoff = node.delivery.frame("resident_handoff")["handoff"]
        assert handoff["replica_id"] == cold.replica_id
        assert directory.all() == [cold]

    asyncio.run(run())


def test_a_cordon_drains_a_warm_demand_replica_after_its_in_flight_claim() -> None:
    async def run() -> None:
        node = Node()
        replica = await admitted_boundary(node)
        (claim,) = node.control.stores.claims.all()

        node.delivery.cordoned.add(_SERVE_WORKER)
        node.control.on_workers_cordoned([_SERVE_WORKER])
        await asyncio.sleep(0)

        assert replica.state is ReplicaState.DRAINING
        assert not node.control.stores.pools.feasible_candidates(
            FAMILY.family, AdmissionProfile(engine_batch_key=FAMILY.engine_batch_key)
        )
        node.control.on_invocation_terminal(claim.invocation_id)
        await asyncio.sleep(0)

        assert replica.state is ReplicaState.STOPPED
        assert replica.serve_task_id is not None
        assert node.status(replica.serve_task_id) == TaskStatus.CANCELLING
        plan = node.control._lifecycle.plan_capacity(FAMILY.family, "m")
        assert plan.action == "materialize"

    asyncio.run(run())


def test_a_new_claim_cold_starts_elsewhere_while_the_family_replica_drains() -> None:
    async def run() -> None:
        node = Node()
        draining = await admitted_boundary(node)
        (first,) = node.control.stores.claims.all()
        node.delivery.cordoned.add(_SERVE_WORKER)
        node.control.on_workers_cordoned([_SERVE_WORKER])
        await asyncio.sleep(0)
        assert draining.state is ReplicaState.DRAINING

        _, ids = await _register(node.runtime, _RESIDENT_WF)
        _capture_resident_boundary(node.runtime, ids["writer"])
        directory = node.control.stores.directory
        await until(lambda: len(directory.all()) == 2)
        (fresh,) = [r for r in directory.all() if r.replica_id != draining.replica_id]
        assert fresh.serve_task_id is not None
        node.delivery.serve_workers[fresh.serve_task_id] = "wkr-3"
        node.serve(fresh.serve_task_id, "wkr-3")
        await until(
            lambda: any(rid == fresh.replica_id for rid, _ in handoff_replicas(node))
        )
        (second,) = [
            c
            for c in node.control.stores.claims.all()
            if c.invocation_id != first.invocation_id
        ]
        assert second.replica_id == fresh.replica_id
        assert draining.state is ReplicaState.DRAINING

        node.control.on_invocation_terminal(first.invocation_id)
        node.control.on_invocation_terminal(second.invocation_id)
        await asyncio.sleep(0)

        assert draining.state is ReplicaState.STOPPED
        assert [
            r.replica_id for r in directory.all() if r.state is ReplicaState.WARM
        ] == [fresh.replica_id]

    asyncio.run(run())


def test_a_cordon_invalidates_a_cold_start_on_its_worker() -> None:
    async def run() -> None:
        node = Node()
        node.control.bind_loop(asyncio.get_running_loop())
        cold = await node.materialize_async()

        node.control.on_workers_cordoned([_SERVE_WORKER])
        await asyncio.sleep(0)

        assert cold.state is ReplicaState.PREEMPTED

    asyncio.run(run())


def test_a_cold_start_reporting_ready_on_a_cordoned_worker_is_retired() -> None:
    async def run() -> None:
        node = Node()
        node.control.bind_loop(asyncio.get_running_loop())
        cold = await node.materialize_async()
        assert cold.serve_task_id is not None
        node.delivery.cordoned.add(_SERVE_WORKER)

        node.serve(cold.serve_task_id)
        await asyncio.sleep(0)

        assert cold.state is ReplicaState.PREEMPTED

    asyncio.run(run())


def test_a_restart_drains_a_warm_demand_replica_on_a_cordoned_worker() -> None:
    node = Node()
    warm = node.warm()

    asyncio.run(
        node.restart_async(lambda booted: booted.delivery.cordoned.add(_SERVE_WORKER))
    )

    assert node.replica(warm.replica_id).state is ReplicaState.STOPPED


def test_a_cordon_leaves_a_standing_replica_serving() -> None:
    async def run() -> None:
        node = Node()
        node.control.bind_loop(asyncio.get_running_loop())
        serve_task_id = await node.submit_serve_async()
        node.serve(serve_task_id)
        standing = node.adopt_standing(serve_task_id)

        node.delivery.cordoned.add(_SERVE_WORKER)
        node.control.on_workers_cordoned([_SERVE_WORKER])
        await asyncio.sleep(0)

        assert standing.state is ReplicaState.WARM
        assert node.status(serve_task_id) == TaskStatus.DISPATCHED

    asyncio.run(run())
