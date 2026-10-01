"""A worker that re-registers gives up the dispatch it was running."""

import subprocess  # nosec B404
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    make_worker_hardware,
    make_worker_task_message,
    no_mediated_op,
)
from worker.executors.base_executor import Executor, TaskCancelledError
from worker.runner import Runner


class _ChildProcess(Executor):
    """Runs a child process until cancelled, as a serve engine or a session does."""

    name = "echo"

    def __init__(self) -> None:
        self.child: subprocess.Popen[bytes] | None = None
        self.running = threading.Event()
        self.ran: list[str] = []

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        self.ran.append(task.task_id)
        self.child = subprocess.Popen(  # nosec B603 - fixed argv, no shell
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        self.running.set()
        if self.child.wait(timeout=30) != 0:
            raise TaskCancelledError(f"Task {task.task_id} cancelled")
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        if self.child is not None:
            self.child.terminate()


def _runner(tmp_path: Path, executor: Executor, dispatches: list[str]) -> Runner:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
    lifecycle.held_boundaries.return_value = []
    return Runner(
        lifecycle=lifecycle,
        task_stream=[
            make_worker_task_message(
                {"taskType": "echo"},
                task_id="tsk-1",
                task_type=TaskType.ECHO,
                dispatch_id=dispatch_id,
            )
            for dispatch_id in dispatches
        ],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    )


def test_abandoning_the_running_dispatch_ends_its_child_process(
    tmp_path: Path,
) -> None:
    executor = _ChildProcess()
    runner = _runner(tmp_path, executor, ["dsp-1"])
    loop = threading.Thread(target=runner.start, daemon=True)
    loop.start()
    assert executor.running.wait(timeout=10)

    runner.abandon_running("dsp-1")
    loop.join(timeout=10)

    assert not loop.is_alive()
    assert executor.child is not None and executor.child.poll() is not None
    runner.lifecycle.set_cancelled.assert_called_once()  # type: ignore[attr-defined]
    runner.lifecycle.set_succeeded.assert_not_called()  # type: ignore[attr-defined]


def test_a_dispatch_given_up_before_it_starts_never_runs(tmp_path: Path) -> None:
    executor = _ChildProcess()
    runner = _runner(tmp_path, executor, ["dsp-1"])

    runner.abandon_running("dsp-1")
    runner.start()

    assert executor.ran == []
    runner.lifecycle.set_cancelled.assert_called_once()  # type: ignore[attr-defined]


def test_abandoning_another_dispatch_leaves_the_running_one(tmp_path: Path) -> None:
    executor = _ChildProcess()
    runner = _runner(tmp_path, executor, ["dsp-2"])
    loop = threading.Thread(target=runner.start, daemon=True)
    loop.start()
    assert executor.running.wait(timeout=10)

    runner.abandon_running("dsp-1")

    assert executor.child is not None and executor.child.poll() is None
    executor.child.terminate()
    loop.join(timeout=10)
