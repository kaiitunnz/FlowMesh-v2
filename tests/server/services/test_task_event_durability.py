"""A task event whose transition is not durable is never acknowledged.

The consumer holds the event's stream cursor for as long as the transition's writes
fail with a persistence error, however many times it retries, and acknowledges it only
once the transition commits, in the same process or after a restart that replays it.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, cast
from unittest import mock

import pytest
import redis.exceptions

from server.config import OrchestrationConfig
from server.registries.workflow import PersistedTask
from server.services import monitoring
from server.services.monitoring import TASK_EVENT_HANDLER_MAX_ATTEMPTS, EventMonitor
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime, TransitionNotDurable
from server.task.workflow_retry import WorkflowRetryScheduler
from shared.schemas.event import TaskEvent
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader
from tests.server.services.test_task_event_fence import _ECHO_V2, _event
from tests.server.task.test_task_merge import (
    _monitor,
    _next,
    _register,
    _Registry,
    _WorkerRegistryStub,
)

_ENTRY = "1-0"


class _Store(_Registry):
    """Raises ``error`` from every task-record commit while it is set, and counts each
    commit that settles a task DONE."""

    def __init__(self) -> None:
        super().__init__()
        self.error: BaseException | None = None
        self.done_commits: list[str] = []

    def commit_transition(
        self, workflow_id: str, *, done: Sequence[str] = (), **kwargs: Any
    ) -> None:
        if self.error is not None:
            raise self.error
        super().commit_transition(workflow_id, done=done, **kwargs)
        self.done_commits += done

    def record(self, task_id: str) -> Any:
        return PersistedTask.model_validate_json(self.task_blobs[task_id]).record


class _Cursor:
    def __init__(self) -> None:
        self.acknowledged: list[str] = []

    def set_value(self, _key: str, value: str) -> None:
        self.acknowledged.append(value)


def _runtime(store: _Store) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, store),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("event-durability"),
        credential_vault=InMemoryCredentialVault(),
        durability_retry=lambda fire, logger: WorkflowRetryScheduler(
            fire, logger, base_delay_sec=0.0, run_thread=False
        ),
    )


def _consumer(runtime: TaskRuntime, event: TaskEvent) -> tuple[EventMonitor, _Cursor]:
    monitor = _monitor(runtime)
    monitor._metrics = mock.MagicMock()
    cursor = _Cursor()
    monitor._redis_client = cast(Any, cursor)
    monitor._parse_stream_event = lambda _fields: event  # type: ignore[method-assign,assignment]
    return monitor, cursor


def _consume(monitor: EventMonitor) -> str:
    return monitor._consume_stream_batch([(_ENTRY, {})], "0-0")


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitoring, "_NOT_DURABLE_BACKOFF_SEC", 0.0)


def _succeeding() -> tuple[_Store, TaskRuntime, str, TaskEvent]:
    store = _Store()
    runtime = _runtime(store)
    asyncio.run(_register(runtime, _ECHO_V2))
    task_id = _next(runtime)
    assert task_id is not None
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    return (
        store,
        runtime,
        task_id,
        _event("TASK_SUCCEEDED", runtime, task_id, "wkr-1", "dsp-1"),
    )


def _assert_counted_once(monitor: EventMonitor) -> None:
    metrics = cast(mock.MagicMock, monitor._metrics)
    assert [call.args[0].type for call in metrics.record_task_event.call_args_list] == [
        "TASK_SUCCEEDED"
    ]


def _held_past_the_budget(
    store: _Store, runtime: TaskRuntime, event: TaskEvent
) -> tuple[EventMonitor, _Cursor]:
    monitor, cursor = _consumer(runtime, event)
    store.error = redis.exceptions.ReadOnlyError("read-only replica")
    for _ in range(TASK_EVENT_HANDLER_MAX_ATTEMPTS * 2):
        assert _consume(monitor) == "0-0"
    assert cursor.acknowledged == []
    assert store.done_commits == []
    return monitor, cursor


def test_a_held_event_is_acknowledged_only_once_its_transition_commits() -> None:
    store, runtime, task_id, event = _succeeding()
    monitor, cursor = _held_past_the_budget(store, runtime, event)
    store.error = None

    assert _consume(monitor) == _ENTRY

    assert cursor.acknowledged == [_ENTRY]
    assert set(store.done_commits) == {task_id}
    assert store.record(task_id).status == TaskStatus.DONE
    _assert_counted_once(monitor)


def test_a_crash_while_an_event_is_held_replays_it_from_its_cursor() -> None:
    store, runtime, task_id, event = _succeeding()
    _held_past_the_budget(store, runtime, event)
    reference = runtime._tasks[task_id].result_reference
    runtime.shutdown()
    store.error = None

    restored = _runtime(store)
    assert asyncio.run(restored.rehydrate()) == 1
    assert store.record(task_id).status == TaskStatus.DISPATCHED
    monitor, cursor = _consumer(restored, event)

    assert _consume(monitor) == _ENTRY

    assert cursor.acknowledged == [_ENTRY]
    assert set(store.done_commits) == {task_id}
    _assert_counted_once(monitor)
    durable = store.record(task_id)
    assert durable.status == TaskStatus.DONE
    assert durable.dispatch_id == "dsp-1"
    assert durable.result_reference == reference


@pytest.mark.parametrize(
    "error",
    [
        TypeError("unserializable record"),
        redis.exceptions.DataError("Invalid input of type: 'NoneType'."),
        redis.exceptions.ResponseError(
            "Command # 2 (SADD k x) of pipeline caused error: WRONGTYPE Operation "
            "against a key holding the wrong kind of value"
        ),
    ],
    ids=type,
)
def test_a_programming_error_inside_a_write_is_not_held(error: Exception) -> None:
    store, runtime, task_id, event = _succeeding()
    store.error = error

    with pytest.raises(type(error)):
        runtime.mark_succeeded(task_id, "wkr-1", event.payload, event.ts, "dsp-1")

    assert runtime._committer.debt == {}
    assert not runtime._durability.pending(runtime._tasks[task_id].workflow_id)


def test_a_programming_error_inside_a_write_spends_the_handler_budget() -> None:
    store, runtime, _, event = _succeeding()
    monitor, cursor = _consumer(runtime, event)
    store.error = TypeError("unserializable record")

    for _ in range(TASK_EVENT_HANDLER_MAX_ATTEMPTS - 1):
        assert _consume(monitor) == "0-0"
    assert _consume(monitor) == _ENTRY

    assert cursor.acknowledged == [_ENTRY]
    assert monitor._not_durable_tries == {}


def test_a_transition_not_durable_reaches_the_consumer_as_its_own_signal() -> None:
    store, runtime, task_id, event = _succeeding()
    store.error = redis.exceptions.ReadOnlyError("read-only replica")

    with pytest.raises(TransitionNotDurable) as raised:
        runtime.mark_succeeded(task_id, "wkr-1", event.payload, event.ts, "dsp-1")

    assert set(raised.value.held) == {runtime._tasks[task_id].workflow_id}
    assert isinstance(
        next(iter(raised.value.held.values())), redis.exceptions.ReadOnlyError
    )


def test_a_redelivered_event_owes_its_writes_once() -> None:
    store, runtime, task_id, event = _succeeding()
    monitor, _ = _consumer(runtime, event)
    store.error = redis.exceptions.ReadOnlyError("read-only replica")
    _consume(monitor)
    owed = list(runtime._committer.debt[runtime._tasks[task_id].workflow_id])

    for _ in range(TASK_EVENT_HANDLER_MAX_ATTEMPTS):
        _consume(monitor)

    assert runtime._committer.debt[runtime._tasks[task_id].workflow_id] == owed
