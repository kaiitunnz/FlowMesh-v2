"""Contract P3, the far end: a dispatched task's traceparent reaches the lifecycle
notifications the runner makes around its run.

``WorkerTaskMessage.traceparent`` is set once at dispatch; this drives one task through
the real ``Runner.start()`` loop and asserts every lifecycle call it makes along the way
-- started, the log emitter, and the terminal notification -- carries the same value,
and that a message with none set carries none through either.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.executors.base_executor import Executor
from worker.runner import Runner

_TP = "00-11111111111111111111111111111111-2222222222222222-01"


class _EchoExecutor(Executor):
    name = "echo"

    def __init__(self) -> None:  # noqa: D107
        pass

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        return None


def _run(tmp_path: Path, traceparent: str | None) -> MagicMock:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    executor = _EchoExecutor()
    msg = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        traceparent=traceparent,
    )
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path,
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    )
    runner.start()
    return lifecycle


def test_a_dispatched_traceparent_reaches_every_lifecycle_call(tmp_path: Path) -> None:
    lifecycle = _run(tmp_path, _TP)

    assert lifecycle.notify_task_started.call_args.kwargs["traceparent"] == _TP
    assert lifecycle.set_succeeded.call_args.kwargs["traceparent"] == _TP
    assert (
        lifecycle.client.create_task_log_emitter.call_args.kwargs["traceparent"] == _TP
    )


def test_no_dispatched_traceparent_reaches_no_lifecycle_call(tmp_path: Path) -> None:
    lifecycle = _run(tmp_path, None)

    assert lifecycle.notify_task_started.call_args.kwargs["traceparent"] is None
    assert lifecycle.set_succeeded.call_args.kwargs["traceparent"] is None
    assert (
        lifecycle.client.create_task_log_emitter.call_args.kwargs["traceparent"] is None
    )
