"""Runner shutdown and the bookkeeping of cancels and stops it was sent."""

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.executors.base_executor import Executor
from worker.main import run_until_exit
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


def test_a_task_loop_ending_on_an_error_unregisters_ungracefully(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _Echo(), "tsk-1")
    lifecycle = MagicMock()

    with (
        patch.object(runner, "_resolve_output_dir", side_effect=OSError(28, "full")),
        pytest.raises(OSError),
    ):
        run_until_exit(runner, lifecycle, MagicMock())

    lifecycle.shutdown.assert_called_once_with(graceful=False)


def test_a_requested_shutdown_unregisters_gracefully(tmp_path: Path) -> None:
    runner: Runner

    def stop_while_running(_task_id: str) -> None:
        runner.stop()

    runner = _runner(tmp_path, _Echo(on_run=stop_while_running), "tsk-1", "tsk-2")
    lifecycle = MagicMock()

    run_until_exit(runner, lifecycle, MagicMock())

    lifecycle.shutdown.assert_called_once_with(graceful=True)


def test_a_worker_shutting_down_never_reports_itself_idle(tmp_path: Path) -> None:
    runner: Runner

    def stop_while_running(_task_id: str) -> None:
        runner.stop()

    runner = _runner(tmp_path, _Echo(on_run=stop_while_running), "tsk-1")
    runner.start()

    runner.lifecycle.set_busy.assert_called_once_with("tsk-1")  # type: ignore[attr-defined]
    runner.lifecycle.set_idle.assert_not_called()  # type: ignore[attr-defined]


class _Recording(_Echo):
    def __init__(self) -> None:
        super().__init__()
        self.stops: list[str] = []
        self.ran: list[str] = []

    def stop(self, task_id: str) -> None:
        self.stops.append(task_id)

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        self.ran.append(task.task_id)
        return BaseExecutorResult()


def test_a_stop_landing_before_the_executor_binds_reaches_it(tmp_path: Path) -> None:
    executor = _Recording()
    runner = _runner(tmp_path, executor, "tsk-1")
    stops: list[tuple[str, str]] = []
    runner.lifecycle.client.iter_stops.side_effect = lambda: (  # type: ignore[attr-defined]
        [stops.pop()] if stops else []
    )

    def stop_while_hydrating(_msg: Any) -> None:
        stops.append(("tsk-1", "user"))
        # The interrupt monitor polls every half second.
        time.sleep(1.2)

    with patch.object(
        runner._input_hydrator, "hydrate", side_effect=stop_while_hydrating
    ):
        runner.start()

    assert executor.stops == ["tsk-1"]
    assert executor.ran == ["tsk-1"]


def test_a_shutdown_landing_before_the_executor_binds_gives_the_task_up(
    tmp_path: Path,
) -> None:
    executor = _Recording()
    runner = _runner(tmp_path, executor, "tsk-1")

    def shut_down_while_hydrating(_msg: Any) -> None:
        runner.stop()
        assert runner._shutdown_thread is not None
        runner._shutdown_thread.join()

    with patch.object(
        runner._input_hydrator, "hydrate", side_effect=shut_down_while_hydrating
    ):
        runner.start()

    assert executor.ran == []
    runner.lifecycle.set_cancelled.assert_called_once()  # type: ignore[attr-defined]


def test_a_repeated_stop_starts_one_shutdown(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _Echo())
    runner.stop()
    first = runner._shutdown_thread
    runner.stop()

    assert runner._shutdown_thread is first
    assert first is not None
    first.join(timeout=2.0)
    runner.lifecycle.stop.assert_called_once_with()  # type: ignore[attr-defined]
