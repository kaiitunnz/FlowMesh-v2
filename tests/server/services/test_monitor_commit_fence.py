"""What the event monitor does after a runtime transition waits for its commit.

A task's log stream closes, and what its dispatch exposed is torn down, only once the
transition that ended the dispatch is durable; a report handled again never repeats
that teardown on a later dispatch. A caller that handles a report internally gets its
outcome while the writes are held, and a report acknowledged by the stream stays
unacknowledged while any workflow its handling touched owes writes.
"""

import asyncio
import logging
import threading
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.dispatcher.base import Dispatcher
from server.registries.workflow import PersistedTask
from server.services import monitoring
from server.services.monitoring import EventMonitor
from server.task.models import PublishGate, TaskStatus
from server.task.runtime import TaskRuntime, TransitionNotDurable
from shared.schemas.event import TaskEvent, WorkerEvent
from tests.server.dispatch_helpers import record_dispatch
from tests.server.services.test_task_event_fence import _ECHO_V2, _event
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_task_merge import (
    _WORKER,
    _merged_success,
    _siblings,
)
from tests.server.task.test_v2_orchestration import (
    _TS,
    LINEAR,
    FakeRegistry,
    _register,
    _worker,
)
from tests.support.waiting import pop_ready

_ENTRY = "1-0"

TWO_V1 = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: two}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""


class _Store(FakeRegistry):
    """Refuses every task-record commit of the workflows in ``down``, or of every
    workflow while ``all_down``."""

    def __init__(self) -> None:
        super().__init__()
        self.down: set[str] = set()
        self.all_down = False

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        if self.all_down or workflow_id in self.down:
            raise ConnectionError("records refused")
        super().commit_transition(workflow_id, **kwargs)

    def status(self, task_id: str) -> str:
        return self.task_record(task_id).status

    def task_record(self, task_id: str) -> Any:
        return PersistedTask.model_validate_json(self.task_blobs[task_id]).record


class _Telemetry:
    """The stream cursor and the telemetry writes the monitor makes."""

    def __init__(self) -> None:
        self.acknowledged: list[str] = []
        self.sealed: list[str] = []

    def set_value(self, key: str, value: str) -> None:
        if key.endswith(":closed"):
            self.sealed.append(key)
        else:
            self.acknowledged.append(value)

    def xadd_telemetry(self, *args: Any, **kwargs: Any) -> None:
        pass

    def expire_telemetry(self, *args: Any) -> None:
        pass

    def expire(self, *args: Any) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitoring, "_NOT_DURABLE_BACKOFF_SEC", 0.0)


def _monitor(runtime: TaskRuntime) -> tuple[EventMonitor, _Telemetry, MagicMock]:
    watchdog = MagicMock()
    watchdog.enabled = False
    telemetry = _Telemetry()
    forward = MagicMock()
    monitor = EventMonitor(
        redis_client=cast(Any, telemetry),
        logger=logging.getLogger("commit-fence"),
        runtime=runtime,
        dispatcher=Dispatcher(runtime, MagicMock(), logging.getLogger("fence-d")),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=watchdog,
        port_forward=forward,
    )
    return monitor, telemetry, forward


def _consume(monitor: EventMonitor, event: TaskEvent) -> str:
    setattr(monitor, "_parse_stream_event", lambda _fields: event)
    return monitor._consume_stream_batch([(_ENTRY, {})], "0-0")


def _dispatched(template: str) -> tuple[_Store, TaskRuntime, str, str]:
    store = _Store()
    runtime = _runtime(store)
    workflow_id, ids = asyncio.run(_register(runtime, template))
    task_id = next(iter(ids.values()))
    assert pop_ready(runtime, 0.05) == task_id
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    return store, runtime, workflow_id, task_id


def test_a_held_settlement_closes_no_log_and_tears_nothing_down_until_durable() -> None:
    store, runtime, workflow_id, task_id = _dispatched(_ECHO_V2)
    monitor, telemetry, forward = _monitor(runtime)
    success = _event("TASK_SUCCEEDED", runtime, task_id, "wkr-1", "dsp-1")
    store.all_down = True

    assert _consume(monitor, success) == "0-0"

    assert telemetry.acknowledged == []
    assert runtime.get_record(task_id).status == TaskStatus.DONE  # type: ignore[union-attr]
    assert telemetry.sealed == []
    forward.unregister_task.assert_not_called()

    store.all_down = False
    runtime._retry_durability(workflow_id)
    assert store.status(task_id) == TaskStatus.DONE
    assert len(telemetry.sealed) == 1
    forward.unregister_task.assert_called_once_with(task_id)

    assert _consume(monitor, success) == _ENTRY
    assert telemetry.acknowledged == [_ENTRY]
    assert len(telemetry.sealed) == 1
    forward.unregister_task.assert_called_once_with(task_id)


def test_a_redelivered_failure_does_not_tear_down_the_next_dispatch() -> None:
    store, runtime, workflow_id, task_id = _dispatched(TWO_V1)
    monitor, telemetry, forward = _monitor(runtime)
    failure = _event("TASK_FAILED", runtime, task_id, "wkr-1", "dsp-1")
    store.all_down = True
    assert _consume(monitor, failure) == "0-0"
    forward.unregister_task.assert_not_called()

    store.all_down = False
    runtime._retry_durability(workflow_id)
    forward.unregister_task.assert_called_once_with(task_id)
    assert pop_ready(runtime, 0.05) is not None
    record_dispatch(runtime, task_id, "wkr-2", "dsp-2")
    forward.reset_mock()

    assert _consume(monitor, failure) == _ENTRY

    record = runtime.get_record(task_id)
    assert record is not None
    assert (record.status, record.dispatch_id) == (TaskStatus.DISPATCHED, "dsp-2")
    forward.unregister_task.assert_not_called()


def test_a_task_does_not_publish_again_while_its_teardown_is_owed() -> None:
    store, runtime, workflow_id, task_id = _dispatched(TWO_V1)
    monitor, _, forward = _monitor(runtime)
    failure = _event("TASK_FAILED", runtime, task_id, "wkr-1", "dsp-1")
    store.all_down = True
    _consume(monitor, failure)
    store.all_down = False
    assert pop_ready(runtime, 0.05) == task_id

    # Publishing makes the failure durable and delivers its teardown, not the
    # dispatch.
    gate = runtime.begin_publish(task_id, _worker("wkr-2"), "dsp-2")
    assert gate is PublishGate.NOT_DURABLE
    forward.unregister_task.assert_called_once_with(task_id)
    assert runtime.begin_publish(task_id, _worker("wkr-2"), "dsp-2") is (
        PublishGate.PUBLISH
    )


def test_a_teardown_runs_once_and_never_on_a_later_dispatch() -> None:
    store, runtime, workflow_id, task_id = _dispatched(TWO_V1)
    torn: list[str] = []
    store.all_down = True
    runtime.mark_started(task_id, "wkr-1", {}, _TS, "dsp-1")
    runtime.file_cleanup(task_id, "dsp-0", "release", lambda: torn.append("dsp-0"))
    runtime.file_cleanup(task_id, "dsp-1", "release", lambda: torn.append("dsp-1"))
    runtime.file_cleanup(task_id, "dsp-1", "release", lambda: torn.append("again"))
    assert torn == []

    store.all_down = False
    runtime._retry_durability(workflow_id)
    assert torn == ["dsp-1"]


def test_a_failed_teardown_is_retried_with_its_workflow() -> None:
    store, runtime, workflow_id, task_id = _dispatched(TWO_V1)
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("telemetry down")

    store.all_down = True
    runtime.mark_succeeded(task_id, "wkr-1", {}, _TS, "dsp-1")
    runtime.file_cleanup(task_id, "dsp-1", "close-log", flaky)
    assert attempts == []

    store.all_down = False
    runtime._retry_durability(workflow_id)
    assert attempts == [1]
    assert runtime._durability.pending(workflow_id)
    runtime._retry_durability(workflow_id)
    assert attempts == [1, 1]
    assert not runtime._actions.cleanup_owed(task_id)


def test_a_replay_waits_for_every_workflow_its_first_handling_held() -> None:
    store = _Store()
    runtime = _runtime(store)
    w1, ids1 = asyncio.run(_register(runtime, _siblings(names=("a",))))
    w2, ids2 = asyncio.run(_register(runtime, _siblings(names=("b",))))
    a, b = ids1["a"], ids2["b"]
    assert pop_ready(runtime, 0.05) == a
    assert runtime.plan_merge(a, 8, _WORKER.id) == [b]
    record_dispatch(runtime, a, _WORKER)
    payload = _merged_success(runtime, a)  # no result for b: b returns to the queue

    store.down = {w2}
    with pytest.raises(TransitionNotDurable) as first, runtime.acknowledging():
        runtime.mark_succeeded(a, "wkr-1", payload, _TS)
    assert set(first.value.held) == {w2}
    # The stream redelivers while the other workflow's store is still down.
    with pytest.raises(TransitionNotDurable), runtime.acknowledging():
        runtime.mark_succeeded(a, "wkr-1", payload, _TS)

    store.down = set()
    with runtime.acknowledging():
        runtime.mark_succeeded(a, "wkr-1", payload, _TS)
    assert w2 not in runtime._committer.debt
    assert store.status(b) == TaskStatus.PENDING
    assert a not in runtime._committer.unacknowledged


def test_an_unregister_returns_every_task_while_a_write_is_held() -> None:
    store = _Store()
    runtime = _runtime(store)
    workflow_id, ids = asyncio.run(_register(runtime, TWO_V1))
    for _ in ids:
        task_id = pop_ready(runtime, 0.05)
        assert task_id is not None
        record_dispatch(runtime, task_id, "wkr-1", f"dsp-{task_id}")
    first, other = ids["a"], ids["b"]
    with runtime._lock:
        runtime._tasks[first].attempts = runtime._tasks[first].max_attempts - 1
    monitor, _, _ = _monitor(runtime)
    store.all_down = True

    monitor._handle_worker_event(
        WorkerEvent(type="UNREGISTER", worker_id="wkr-1", graceful=False)
    )

    assert runtime.get_record(first).status == TaskStatus.FAILED  # type: ignore[union-attr]
    assert runtime.get_record(other).status == TaskStatus.PENDING  # type: ignore[union-attr]
    assert pop_ready(runtime, 0.05) == other
    store.all_down = False
    runtime._retry_durability(workflow_id)
    assert store.status(first) == TaskStatus.FAILED
    assert store.status(other) == TaskStatus.PENDING


def test_failing_a_task_under_a_held_write_reports_it_and_its_dependents() -> None:
    store = _Store()
    runtime = _runtime(store)
    workflow_id, ids = asyncio.run(_register(runtime, LINEAR))
    metrics = MagicMock()
    dispatcher = Dispatcher(
        runtime, MagicMock(), logging.getLogger("fail"), metrics_recorder=metrics
    )
    store.all_down = True

    dispatcher.fail_task(ids["a"], "bad spec")

    failed = {
        call.args[0].task_id
        for call in metrics.record_task_event.call_args_list
        if call.args[0].type == "TASK_FAILED"
    }
    assert failed == {ids["a"], ids["b"], ids["c"]}
    assert workflow_id in runtime._committer.debt


def test_an_unregister_revokes_after_its_requeues_whatever_thread_delivers() -> None:
    store = _Store()
    runtime = _runtime(store)
    asyncio.run(_register(runtime, TWO_V1))
    for _ in range(2):
        task_id = pop_ready(runtime, 0.05)
        assert task_id is not None
        record_dispatch(runtime, task_id, "wkr-1", f"dsp-{task_id}")
    monitor, _, _ = _monitor(runtime)
    order: list[str] = []
    workers = runtime._worker_registry
    publish, requeue = workers.publish_revoke, monitor._dispatcher.requeue_task

    def revoke(*args: Any) -> int:
        order.append("revoke")
        return publish(*args)

    def requeue_while_another_transition_ends(task_id: str, **kwargs: Any) -> Any:
        if not order:
            other = threading.Thread(target=runtime.release_merge, args=("tsk-x",))
            other.start()
            other.join()
        order.append("requeue")
        return requeue(task_id, **kwargs)

    setattr(workers, "publish_revoke", revoke)
    setattr(monitor._dispatcher, "requeue_task", requeue_while_another_transition_ends)

    monitor._handle_worker_event(
        WorkerEvent(type="UNREGISTER", worker_id="wkr-1", graceful=False)
    )

    assert order == ["requeue", "requeue", "revoke", "revoke"]
