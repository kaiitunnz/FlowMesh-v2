"""A node's unregister of a worker id reaches the root's record only when that node
wrote it, so an id another node holds keeps its live worker, and recovers only what
the unregistering node's worker held."""

from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis

from server.clients.redis import WORKERS_SET_KEY, worker_hb_key, worker_key
from server.registries.worker import WorkerRegistry
from server.services.monitoring import EventMonitor
from shared.schemas.event import WorkerEvent
from tests.server.redis_helpers import fake_redis_client
from tests.server.task.test_task_merge import _monitor

_WORKER = "wkr-1"


def _setup(holder: str | None) -> tuple[EventMonitor, MagicMock, fakeredis.FakeRedis]:
    server = fakeredis.FakeServer()
    rds = fakeredis.FakeRedis(server=server, decode_responses=True)
    if holder is not None:
        rds.sadd(WORKERS_SET_KEY, _WORKER)
        rds.hset(
            worker_key(_WORKER), mapping={"node_alias": holder, "node_id": "nde-b"}
        )
        rds.setex(worker_hb_key(_WORKER), 120, "ts")
    runtime = MagicMock()
    monitor = _monitor(cast(Any, runtime))
    monitor._worker_registry = WorkerRegistry(fake_redis_client(server))
    return monitor, runtime, rds


def _unregister(node_alias: str) -> WorkerEvent:
    return WorkerEvent(
        type="UNREGISTER", worker_id=_WORKER, payload={"node_alias": node_alias}
    )


def test_an_unregister_of_an_id_another_node_holds_leaves_its_worker() -> None:
    monitor, runtime, rds = _setup(holder="box-b")

    monitor._handle_worker_event(_unregister("box-a"))

    assert rds.hget(worker_key(_WORKER), "node_alias") == "box-b"
    assert rds.sismember(WORKERS_SET_KEY, _WORKER)
    runtime.recover_tasks_for_worker.assert_called_once_with(
        _WORKER, spend_attempt=True, node_id=None, node_alias="box-a"
    )


def test_a_nodes_unregister_of_its_own_worker_ends_it() -> None:
    monitor, runtime, rds = _setup(holder="box-a")

    monitor._handle_worker_event(_unregister("box-a"))

    assert not rds.exists(worker_key(_WORKER))
    assert not rds.sismember(WORKERS_SET_KEY, _WORKER)
    runtime.recover_tasks_for_worker.assert_called_once()


def test_an_unregister_of_an_id_with_no_record_still_recovers_its_tasks() -> None:
    monitor, runtime, _ = _setup(holder=None)

    monitor._handle_worker_event(_unregister("box-a"))

    runtime.recover_tasks_for_worker.assert_called_once()
