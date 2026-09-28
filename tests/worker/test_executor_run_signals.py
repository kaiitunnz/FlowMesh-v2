"""Cancel and stop requests reach only the task an executor runs, and a graceful stop
ends a serve task successfully whenever it lands."""

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.schemas.result import ServeResult
from shared.schemas.result.catalog import DevModelResult
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import DevModelSpecStrict
from shared.tasks.specs.serve import ServeSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors import dev_model_executor
from worker.executors.base_executor import RunSignals, TaskCancelledError
from worker.executors.dev_model_executor import DevModelExecutor
from worker.executors.vllm_serve_executor import VLLMServeExecutor


def _dev_model() -> DevModelExecutor:
    return DevModelExecutor(
        make_worker_config(enable_dev_model=True), make_worker_hardware()
    )


def _dev_model_task(task_id: str, ttl: float = 60.0) -> Any:
    spec = DevModelSpecStrict(taskType=TaskType.DEV_MODEL, ttlSeconds=ttl)
    return make_worker_task_message(
        spec=spec, task_type=TaskType.DEV_MODEL, task_id=task_id
    )


def _serve_task(task_id: str) -> Any:
    spec = ServeSpecStrict(
        taskType=TaskType.SERVE, model=ModelConfig(source=ModelSource(identifier="m"))
    )
    return make_worker_task_message(
        spec=spec, task_type=TaskType.SERVE, task_id=task_id
    )


def test_a_cancel_after_its_task_ended_does_not_cancel_the_next_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_model_executor, "_POLL_INTERVAL_SEC", 0.01)
    ex = _dev_model()
    with patch.object(ex, "emit_update"):
        ex.run(_dev_model_task("tsk-1", ttl=0.05), tmp_path / "1")
        # An interrupt for the finished task lands while its results are written.
        ex.cancel("tsk-1")
        result = ex.run(_dev_model_task("tsk-2", ttl=0.05), tmp_path / "2")

    assert isinstance(result, DevModelResult)


def test_a_dev_model_stopped_before_launch_succeeds(tmp_path: Path) -> None:
    ex = _dev_model()
    emit = MagicMock()
    ex.stop("tsk-1")
    with patch.object(ex, "emit_update", emit):
        result = ex.run(_dev_model_task("tsk-1"), tmp_path)

    assert isinstance(result, DevModelResult)
    emit.assert_not_called()


def test_a_serve_stopped_before_launch_succeeds_without_starting_vllm(
    tmp_path: Path,
) -> None:
    ex = VLLMServeExecutor(make_worker_config(), make_worker_hardware())
    ex.stop("tsk-1")
    with (
        patch("subprocess.Popen") as popen,
        patch.object(ex, "_poll_health"),
        patch.object(ex, "_wait_for_serve"),
        patch.object(ex, "_terminate_process_group"),
        patch.object(ex, "emit_update"),
    ):
        result = ex.run(_serve_task("tsk-1"), tmp_path)

    assert isinstance(result, ServeResult)
    popen.assert_not_called()


def test_a_serve_stopped_before_it_is_ready_succeeds(tmp_path: Path) -> None:
    ex = VLLMServeExecutor(make_worker_config(), make_worker_hardware())
    proc = MagicMock()
    proc.stdout = io.StringIO("")
    proc.poll.return_value = None
    emit = MagicMock()
    with (
        patch("subprocess.Popen", return_value=proc),
        patch.object(
            ex, "_poll_health", side_effect=lambda *_a, **_k: ex.stop("tsk-1")
        ),
        patch.object(ex, "_terminate_process_group"),
        patch.object(ex, "emit_update", emit),
    ):
        result = ex.run(_serve_task("tsk-1"), tmp_path)

    assert isinstance(result, ServeResult)
    emit.assert_not_called()


def test_a_late_request_for_an_ended_task_keeps_the_running_tasks_request() -> None:
    signals = RunSignals()
    with signals.running("tsk-2"):
        signals.stop("tsk-2")
        signals.stop("tsk-1")
        signals.cancel("tsk-1")

        assert signals.stopped
        assert not signals.cancelled
        assert signals.raise_if_cancelled() is True


def test_a_serve_cancelled_before_launch_never_starts_vllm(tmp_path: Path) -> None:
    ex = VLLMServeExecutor(make_worker_config(), make_worker_hardware())
    ex.cancel("tsk-1")
    with (
        patch("subprocess.Popen") as popen,
        patch.object(ex, "_poll_health"),
        patch.object(ex, "_wait_for_serve"),
        patch.object(ex, "_terminate_process_group"),
        patch.object(ex, "emit_update"),
        pytest.raises(TaskCancelledError),
    ):
        ex.run(_serve_task("tsk-1"), tmp_path)

    popen.assert_not_called()


def test_a_dev_model_cancelled_before_launch_never_serves(tmp_path: Path) -> None:
    ex = _dev_model()
    emit = MagicMock()
    ex.cancel("tsk-1")
    with patch.object(ex, "emit_update", emit), pytest.raises(TaskCancelledError):
        ex.run(_dev_model_task("tsk-1"), tmp_path)

    emit.assert_not_called()
