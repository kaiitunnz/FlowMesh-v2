"""Control revokes a dispatch it resolves without its worker ending it, so a frame of
it still queued at the worker's supervisor never runs."""

import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.clients.redis import SyncRedisClient
from server.registries.worker import Reservation, Worker
from server.supervisor.services.task_listener import TaskListener
from server.task.runtime import TaskRuntime
from shared.schemas.command import InterruptMessage, TaskMessage
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerTaskMessage
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_v2_orchestration import LINEAR, FakeRegistry
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import _runtime as _runtime_v2
from tests.worker.factories import make_worker_task_message

_LOGGER = logging.getLogger("test.dispatch_revocation")
_WORKER = "wkr-1"


class _Registry:
    """A worker registry that publishes onto one supervisor's task listener."""

    def __init__(self, listener: TaskListener | None = None) -> None:
        self.listener = listener
        self.interrupts: list[InterruptMessage] = []
        self.reserved: list[Reservation] = []

    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(id=worker_id, node_id="nod-1")

    def publish_interrupt(self, worker: Any, payload: InterruptMessage) -> int:
        self.interrupts.append(payload)
        if self.listener is not None:
            self.listener._handle_message(json.loads(payload.model_dump_json()))
        return 1

    def release_worker(self, *args: Any) -> bool:
        return True

    def reservations(self) -> list[Reservation]:
        return self.reserved


def _runtime(registry: _Registry) -> tuple[TaskRuntime, str]:
    runtime = _runtime_v2(FakeRegistry())
    runtime._worker_registry = cast(Any, registry)
    _, ids = asyncio.run(_register_v2(runtime, LINEAR))
    assert runtime.next_ready(threading.Event(), timeout=0.01) == ids["a"]
    return runtime, ids["a"]


def _revocations(registry: _Registry) -> list[tuple[str, str, str | None]]:
    return [(i.task_id, i.worker_id, i.dispatch_id) for i in registry.interrupts]


def _frame(task_id: str, dispatch_id: str) -> dict[str, Any]:
    message = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id=task_id,
        assigned_worker=_WORKER,
        dispatch_id=dispatch_id,
    )
    return json.loads(
        TaskMessage(
            worker_id=_WORKER,
            payload=message.model_dump(mode="json", exclude_none=True),
        ).model_dump_json()
    )


@pytest.mark.asyncio
async def test_a_disowned_dispatch_queued_while_the_stream_was_down_never_runs() -> (
    None
):
    listener = TaskListener(cast(SyncRedisClient, None), "nod-1", _LOGGER)
    listener._loop = asyncio.get_running_loop()
    listener.add_worker(_WORKER)
    registry = _Registry(listener)
    runtime, task_id = await asyncio.to_thread(_runtime, registry)

    # dsp-1 waits at the supervisor while the worker's task stream is down.
    record_dispatch(runtime, task_id, _WORKER, "dsp-1")
    listener._handle_message(_frame(task_id, "dsp-1"))
    outcome = runtime.resolve_disowned_dispatch(task_id, "dsp-1", _WORKER, 0)
    assert outcome is not None
    # The task is bound to the same worker again.
    assert runtime.next_ready(threading.Event(), timeout=0.01) == task_id
    record_dispatch(runtime, task_id, _WORKER, "dsp-2")
    listener._handle_message(_frame(task_id, "dsp-2"))
    await asyncio.sleep(0)

    stream = listener.attach_stream(_WORKER)
    assert stream is not None
    listener.remove_worker(_WORKER)
    frames = []
    while (frame := await stream.next()) is not None:
        frames.append(frame)
    assert [
        WorkerTaskMessage.wire_dispatch_id(f) for f in frames if "kind" not in f
    ] == ["dsp-2"]


def test_recovering_a_departed_workers_dispatch_revokes_it() -> None:
    registry = _Registry()
    runtime, task_id = _runtime(registry)
    record_dispatch(runtime, task_id, _WORKER, "dsp-1")

    recovery = runtime.recover_tasks_for_worker(_WORKER, spend_attempt=True)

    assert recovery.resolved
    assert _revocations(registry) == [(task_id, _WORKER, "dsp-1")]


def test_recovering_an_unrecorded_publish_revokes_it() -> None:
    registry = _Registry()
    runtime, task_id = _runtime(registry)
    worker = cast(Worker, SimpleNamespace(id=_WORKER, node_id="nod-1"))
    runtime.begin_publish(task_id, worker, "dsp-1")

    runtime.recover_tasks_for_worker(_WORKER, spend_attempt=True)

    assert _revocations(registry) == [(task_id, _WORKER, "dsp-1")]


def test_a_restart_revokes_a_reserved_dispatch_it_does_not_hold() -> None:
    registry = _Registry()
    runtime, task_id = _runtime(registry)
    registry.reserved = [Reservation(_WORKER, task_id, "dsp-gone")]

    runtime.release_ended_reservations()

    assert _revocations(registry) == [(task_id, _WORKER, "dsp-gone")]


def test_a_restart_leaves_a_dispatch_in_flight_alone() -> None:
    registry = _Registry()
    runtime, task_id = _runtime(registry)
    record_dispatch(runtime, task_id, _WORKER, "dsp-1")
    registry.reserved = [Reservation(_WORKER, task_id, "dsp-1")]

    runtime.release_ended_reservations()

    assert registry.interrupts == []
