"""A worker never runs a dispatch control ended or its re-registration gave up."""

import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    make_worker_hardware,
    make_worker_task_message,
    no_mediated_op,
)
from worker.executors.base_executor import Executor, RunSignals, TaskCancelledError
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


def _lifecycle(runner: Runner) -> MagicMock:
    return cast(MagicMock, runner.lifecycle)


def _runner(tmp_path: Path, executor: Executor, dispatches: list[str]) -> Runner:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_revokes.return_value = []
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
    _lifecycle(runner).set_cancelled.assert_called_once()
    _lifecycle(runner).set_succeeded.assert_not_called()


def test_a_dispatch_given_up_before_it_starts_never_runs(tmp_path: Path) -> None:
    executor = _ChildProcess()
    runner = _runner(tmp_path, executor, ["dsp-1"])

    runner.abandon_running("dsp-1")
    runner.start()

    assert executor.ran == []
    _lifecycle(runner).set_cancelled.assert_called_once()


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


class _Recording(Executor):
    name = "echo"

    def __init__(self) -> None:
        self.ran: list[str | None] = []

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        self.ran.append(task.dispatch_id)
        return BaseExecutorResult()


def _deliver_revoke(runner: Runner, revoke: tuple[str, str]) -> None:
    pending = [revoke]
    _lifecycle(runner).client.iter_revokes.side_effect = lambda: (
        [pending.pop()] if pending else []
    )


def _deliver_interrupt(runner: Runner, interrupt: tuple[str, str, str | None]) -> None:
    pending = [interrupt]
    _lifecycle(runner).client.iter_interrupts.side_effect = lambda: (
        [pending.pop()] if pending else []
    )


def _until(condition: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_a_revoke_for_a_received_dispatch_ends_only_that_dispatch(
    tmp_path: Path,
) -> None:
    executor = _Recording()
    runner = _runner(tmp_path, executor, ["dsp-1", "dsp-2"])
    _deliver_revoke(runner, ("tsk-1", "dsp-1"))

    def revoked_while_hydrating(msg: Any) -> None:
        if msg.dispatch_id == "dsp-1":
            _until(lambda: "dsp-1" in runner._revoked_dispatches)

    with patch.object(
        runner._input_hydrator, "hydrate", side_effect=revoked_while_hydrating
    ):
        runner.start()

    assert executor.ran == ["dsp-2"]


def test_a_revoke_for_another_dispatch_leaves_the_running_one(
    tmp_path: Path,
) -> None:
    executor = _ChildProcess()
    runner = _runner(tmp_path, executor, ["dsp-2"])
    _deliver_revoke(runner, ("tsk-1", "dsp-1"))
    loop = threading.Thread(target=runner.start, daemon=True)
    loop.start()
    assert executor.running.wait(timeout=10)

    _until(lambda: "dsp-1" in runner._revoked_dispatches)

    assert executor.child is not None and executor.child.poll() is None
    executor.child.terminate()
    loop.join(timeout=10)


def test_a_dispatch_cancelled_before_it_starts_is_reported_cancelled(
    tmp_path: Path,
) -> None:
    executor = _Recording()
    runner = _runner(tmp_path, executor, ["dsp-1"])
    _deliver_interrupt(runner, ("tsk-1", "cancelled", "dsp-1"))

    def cancelled_while_hydrating(msg: Any) -> None:
        # A second poll of the interrupts follows the handling of the first.
        _until(lambda: _lifecycle(runner).client.iter_interrupts.call_count >= 2)

    with patch.object(
        runner._input_hydrator, "hydrate", side_effect=cancelled_while_hydrating
    ):
        runner.start()

    assert executor.ran == []
    _lifecycle(runner).set_cancelled.assert_called_once()
    cast(MagicMock, runner.logger).info.assert_any_call(
        "Task %s cancelled: %s", "tsk-1", "Task tsk-1 was cancelled before execution"
    )


class _Signalled(Executor):
    """Records whether each run began cancelled, as a signal-aware executor sees it."""

    name = "echo"

    def __init__(self) -> None:
        self._signals = RunSignals()
        self.ran: list[tuple[str | None, bool]] = []

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        with self._signals.running(task.task_id):
            self.ran.append((task.dispatch_id, self._signals.cancelled))
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        self._signals.cancel(task_id)


def test_a_revoke_landing_before_its_run_leaves_the_next_dispatch_uncancelled(
    tmp_path: Path,
) -> None:
    executor = _Signalled()
    runner = _runner(tmp_path, executor, ["dsp-0", "dsp-1", "dsp-2"])
    _deliver_revoke(runner, ("tsk-1", "dsp-1"))

    def revoked_while_hydrating(msg: Any) -> None:
        if msg.dispatch_id == "dsp-1":
            # A second poll of the revokes follows the handling of the first.
            _until(lambda: _lifecycle(runner).client.iter_revokes.call_count >= 2)

    with patch.object(
        runner._input_hydrator, "hydrate", side_effect=revoked_while_hydrating
    ):
        runner.start()

    assert executor.ran == [("dsp-0", False), ("dsp-2", False)]


def test_a_revoke_landing_while_its_result_is_stored_leaves_the_next_dispatch(
    tmp_path: Path,
) -> None:
    executor = _Signalled()
    runner = _runner(tmp_path, executor, ["dsp-1", "dsp-2"])
    pending = [("tsk-1", "dsp-1")]
    landed = threading.Event()

    def revokes() -> list[tuple[str, str]]:
        if landed.is_set() and pending:
            return [pending.pop()]
        return []

    _lifecycle(runner).client.iter_revokes.side_effect = revokes
    write_results = runner._write_results

    def revoked_while_storing(msg: Any, out_dir: Path, out: Any) -> Any:
        if msg.dispatch_id == "dsp-1":
            # The run has ended; the revoke lands while its result is stored.
            landed.set()
            polls = _lifecycle(runner).client.iter_revokes.call_count
            _until(
                lambda: not pending
                and _lifecycle(runner).client.iter_revokes.call_count >= polls + 2
            )
        return write_results(msg, out_dir, out)

    with patch.object(runner, "_write_results", side_effect=revoked_while_storing):
        runner.start()

    assert executor.ran == [("dsp-1", False), ("dsp-2", False)]
