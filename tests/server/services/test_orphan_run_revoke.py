"""A worker reporting itself busy on a dispatch control no longer holds has that
dispatch revoked, once."""

import asyncio
import json
import threading
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import WORKERS_SET_KEY, worker_hb_key, worker_key
from server.registries.worker import WorkerRegistry
from server.services import monitoring as monitoring_module
from server.services.monitoring import EventMonitor
from server.task.runtime import TaskRuntime
from shared.schemas.event import WorkerEvent
from shared.schemas.worker import WorkerStatus
from tests.server.dispatch_helpers import record_dispatch
from tests.server.registries.test_worker_status_fence import _Sync
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import LINEAR, FakeRegistry
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import _runtime as _runtime_v2

_WORKER = "wkr-1"


class _RecordingSync(_Sync):
    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.published: list[dict[str, Any]] = []

    def publish_control(self, *args: Any) -> int:
        _channel, message = args
        self.published.append(json.loads(message))
        return 1


class _Rds:
    def __init__(self, sync: _RecordingSync) -> None:
        self.sync = sync


@pytest.fixture
def sync() -> _RecordingSync:
    client = fakeredis.FakeRedis(decode_responses=True)
    client.sadd(WORKERS_SET_KEY, _WORKER)
    client.hset(worker_key(_WORKER), mapping={"status": "IDLE", "node_id": "nod-1"})
    client.setex(worker_hb_key(_WORKER), 120, "ts")
    return _RecordingSync(client)


def _setup(sync: _RecordingSync) -> tuple[TaskRuntime, EventMonitor, str]:
    runtime = _runtime_v2(FakeRegistry())
    _, ids = asyncio.run(_register_v2(runtime, LINEAR))
    task_id = ids["a"]
    assert runtime.next_ready(threading.Event(), timeout=0.01) == task_id
    monitor = _monitor(runtime)
    registry = WorkerRegistry(cast(Any, _Rds(sync)))
    runtime._worker_registry = registry
    monitor._worker_registry = registry
    return runtime, monitor, task_id


def _busy(kind: str, task_id: str, dispatch_id: str) -> WorkerEvent:
    return WorkerEvent(
        type=kind,
        worker_id=_WORKER,
        status=WorkerStatus.BUSY,
        dispatch_id=dispatch_id,
        payload={"task_id": task_id, "ttl_sec": 120},
    )


def _revokes(sync: _RecordingSync) -> list[tuple[str, str]]:
    return [
        (m["task_id"], m["dispatch_id"])
        for m in sync.published
        if m.get("kind") == "revoke"
    ]


def test_a_run_of_a_resolved_dispatch_is_revoked_once(sync: _RecordingSync) -> None:
    runtime, monitor, task_id = _setup(sync)
    record_dispatch(runtime, task_id, _WORKER, "dsp-1")
    assert runtime.resolve_disowned_dispatch(task_id, "dsp-1", _WORKER, 0)
    assert _revokes(sync) == [(task_id, "dsp-1")]
    sync.published.clear()
    assert runtime.next_ready(threading.Event(), timeout=0.01) == task_id
    record_dispatch(runtime, task_id, _WORKER, "dsp-2")

    # The late dsp-1 started on the worker, which reports it and then repeats it.
    monitor._handle_worker_event(_busy("STATUS", task_id, "dsp-1"))
    monitor._handle_worker_event(_busy("HEARTBEAT", task_id, "dsp-1"))
    monitor._handle_worker_event(_busy("HEARTBEAT", task_id, "dsp-1"))

    assert _revokes(sync) == [(task_id, "dsp-1")]


def test_a_run_of_the_held_dispatch_is_left_alone(sync: _RecordingSync) -> None:
    runtime, monitor, task_id = _setup(sync)
    record_dispatch(runtime, task_id, _WORKER, "dsp-1")

    monitor._handle_worker_event(_busy("STATUS", task_id, "dsp-1"))
    monitor._handle_worker_event(_busy("HEARTBEAT", task_id, "dsp-1"))

    assert _revokes(sync) == []


def test_an_unregister_revokes_what_its_worker_held(sync: _RecordingSync) -> None:
    runtime, monitor, task_id = _setup(sync)
    record_dispatch(runtime, task_id, _WORKER, "dsp-1")

    monitor._handle_worker_event(
        WorkerEvent(type="UNREGISTER", worker_id=_WORKER, payload={})
    )

    assert _revokes(sync) == [(task_id, "dsp-1")]


def test_a_run_still_reported_past_the_resend_interval_is_revoked_again(
    sync: _RecordingSync, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, monitor, task_id = _setup(sync)
    record_dispatch(runtime, task_id, _WORKER, "dsp-1")
    assert runtime.resolve_disowned_dispatch(task_id, "dsp-1", _WORKER, 0)
    sync.published.clear()
    monkeypatch.setattr(monitoring_module, "_REVOKE_RESEND_SEC", 0.0)

    monitor._handle_worker_event(_busy("HEARTBEAT", task_id, "dsp-1"))
    monitor._handle_worker_event(_busy("HEARTBEAT", task_id, "dsp-1"))

    assert _revokes(sync) == [(task_id, "dsp-1"), (task_id, "dsp-1")]
