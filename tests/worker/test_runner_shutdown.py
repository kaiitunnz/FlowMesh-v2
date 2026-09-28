"""Runner shutdown and the bookkeeping of cancels and stops it was sent."""

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.executors.base_executor import Executor
from worker.runner import Runner


class _Echo(Executor):
    name = "echo"

    def __init__(self, on_run: Any = None) -> None:
        self.on_run = on_run

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        if self.on_run is not None:
            self.on_run(task.task_id)
        return BaseExecutorResult()


def _runner(tmp_path: Path, executor: Executor, *task_ids: str) -> Runner:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    return Runner(
        lifecycle=lifecycle,
        task_stream=[
            make_worker_task_message(
                {"taskType": "echo"}, task_id=task_id, task_type=TaskType.ECHO
            )
            for task_id in task_ids
        ],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    )


def test_a_stop_from_a_frame_holding_the_executor_lock_returns(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _Echo())
    runner._active_executor = _Echo()
    runner._current_task_id = "tsk-1"
    returned = threading.Event()

    def signal_while_holding_the_lock() -> None:
        # A signal handler runs on the thread it interrupts, which may be inside the
        # task loop's hold of the lock.
        with runner._active_executor_lock:
            runner.stop()
            returned.set()

    threading.Thread(target=signal_while_holding_the_lock, daemon=True).start()

    assert returned.wait(timeout=2.0)
    assert runner._shutdown_thread is not None
    runner._shutdown_thread.join(timeout=2.0)
    runner.lifecycle.stop.assert_called_once_with()  # type: ignore[attr-defined]


def test_cancels_and_stops_for_a_task_are_dropped_once_it_ends(
    tmp_path: Path,
) -> None:
    runner: Runner

    def signalled_while_running(task_id: str) -> None:
        with runner._cancel_lock:
            runner._pending_cancels.add(task_id)
            runner._pending_stops.add(task_id)

    runner = _runner(tmp_path, _Echo(on_run=signalled_while_running), "tsk-1")
    runner.start()

    assert runner._pending_cancels == set()
    assert runner._pending_stops == set()
