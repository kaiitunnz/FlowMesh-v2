"""A batch SSH task ends with its container: it fails with the container's exit code,
and a breach of its output limit fails it, whether its output is copied out of the
container or mounted directly."""

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError
from docker.models.containers import Container

from shared.schemas.result import SSHResult
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_live_worker_config
from worker.executors import ssh_executor as ssh_module
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_executor import SSHExecutor

_TASK_ID = "tsk-ssh-exit"
# Far past how long any of these runs, so reaching it fails the test's deadline.
_TTL_SEC = 30
_PROMPT_SEC = 5.0


def _task(max_bytes: int) -> WorkerTaskMessage:
    return WorkerTaskMessage.model_validate(
        {
            "task_id": _TASK_ID,
            "workflow_id": "wfl-1",
            "owner_id": "owner",
            "assigned_worker": "worker-1",
            "dispatched_at": "2026-03-22T00:00:00Z",
            "task": {
                "apiVersion": "mloc/v1",
                "kind": "Task",
                "metadata": {"name": "wf:s"},
                "spec": {
                    "taskType": "ssh",
                    "interactive": False,
                    "ttlSeconds": _TTL_SEC,
                    "image": "python:3.12-slim",
                    "command": ["true"],
                    "sshOutput": {"mountPath": "/out", "maxBytes": max_bytes},
                },
            },
        }
    )


@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SSHExecutor:
    monkeypatch.setenv("SSH_POLL_INTERVAL_SEC", "0.01")
    ex = SSHExecutor(make_live_worker_config(tmp_path), lifecycle=None)
    ex._docker = MagicMock()
    return ex


def _container(exit_code: int, output_bytes: int, polls: int = 3) -> MagicMock:
    """A batch container that exits with ``exit_code`` after ``polls`` reloads.

    Docker refuses an exec into a stopped container, as it does for real.
    """
    container = MagicMock(spec=Container)
    container.name = "ssh-c"
    container.status = "running"

    def reload() -> None:
        if container.reload.call_count >= polls:
            container.status = "exited"

    def exec_run(command: list[str], *_: Any, **__: Any) -> MagicMock:
        if container.status != "running":
            raise APIError("409 Conflict: container is not running")
        if "du -sb" in command[-1]:
            return MagicMock(output=f"{output_bytes}\n".encode(), exit_code=0)
        return MagicMock(exit_code=1)

    container.reload.side_effect = reload
    container.exec_run.side_effect = exec_run
    container.wait.return_value = {"StatusCode": exit_code}
    return container


def _run(ex: SSHExecutor, tmp_path: Path, container: Any, max_bytes: int) -> SSHResult:
    plan = MagicMock()
    plan.copy_output_path = "/out"
    plan.direct_output_path = None
    with (
        patch.object(ex, "prepare"),
        patch.object(ex, "_resolve_noninteractive_command", return_value=["true"]),
        patch.object(ex, "_resolve_inputs", return_value=[]),
        patch.object(ex, "_build_mount_plan", return_value=plan),
        patch.object(ex, "_build_environment", return_value={}),
        patch.object(ex, "_build_run_kwargs", return_value={}),
        patch.object(ex, "_start_container", return_value=(container, None)),
        patch.object(ex, "_stream_container_logs"),
        patch.object(ex, "_save_container_logs"),
        patch.object(ex, "_copy_output_directory"),
        patch.object(ex, "_cleanup_mount_plan"),
        patch.object(ex, "emit_update"),
        patch.object(ssh_module, "maybe_upload_artifacts"),
    ):
        return ex.run(_task(max_bytes), tmp_path / "out")


def test_a_clean_exit_succeeds_promptly(executor: SSHExecutor, tmp_path: Path) -> None:
    started = time.monotonic()

    result = _run(executor, tmp_path, _container(0, output_bytes=10), max_bytes=100)

    assert result.exit_code == 0
    assert time.monotonic() - started < _PROMPT_SEC


def test_a_failed_exit_fails_with_its_code_promptly(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    started = time.monotonic()

    with pytest.raises(ExecutionError, match="exited with code 3"):
        _run(executor, tmp_path, _container(3, output_bytes=10), max_bytes=100)
    assert time.monotonic() - started < _PROMPT_SEC


def test_an_output_limit_breach_fails_promptly(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    container = _container(0, output_bytes=500, polls=1_000_000)
    started = time.monotonic()

    with pytest.raises(ExecutionError, match="exceeded maxBytes"):
        _run(executor, tmp_path, container, max_bytes=100)
    assert time.monotonic() - started < _PROMPT_SEC
    container.stop.assert_called()
