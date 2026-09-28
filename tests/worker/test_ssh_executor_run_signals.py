"""An SSH task a stop reaches succeeds and one a cancel reaches is cancelled, wherever
it lands; a session container lost for no requested reason fails the task."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import NotFound
from docker.models.containers import Container

from shared.schemas.result import SSHResult
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_live_worker_config
from worker.executors import ssh_executor as ssh_module
from worker.executors.base_executor import ExecutionError, TaskCancelledError
from worker.executors.ssh_executor import SSHExecutor

_TASK_ID = "tsk-ssh-1"


def _task(interactive: bool) -> WorkerTaskMessage:
    spec: dict[str, Any] = {"taskType": "ssh", "interactive": interactive}
    if interactive:
        spec["authorizedKeys"] = ["ssh-ed25519 AAAA test"]
    else:
        spec.update({"image": "python:3.12-slim", "command": ["sleep", "600"]})
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
                "spec": spec,
            },
        }
    )


@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SSHExecutor:
    monkeypatch.setenv("SSH_POLL_INTERVAL_SEC", "0.01")
    ex = SSHExecutor(make_live_worker_config(tmp_path), lifecycle=None)
    ex._docker = MagicMock()
    return ex


def _container(on_reload: Callable[[MagicMock], None]) -> MagicMock:
    """A session container whose reload runs ``on_reload`` against it."""
    container = MagicMock(spec=Container)
    container.name = "ssh-c"
    container.status = "running"
    container.ports = {}
    container.exec_run.return_value = MagicMock(exit_code=1)
    container.wait.return_value = {"StatusCode": 143}
    container.reload.side_effect = lambda: on_reload(container)
    return container


def _run(
    ex: SSHExecutor,
    interactive: bool,
    tmp_path: Path,
    container: Any,
    start: MagicMock | None = None,
) -> SSHResult:
    plan = MagicMock()
    plan.copy_output_path = None
    start = start or MagicMock(return_value=(container, None))
    with (
        patch.object(ex, "prepare"),
        patch.object(ex, "_resolve_noninteractive_command", return_value=["sleep"]),
        patch.object(ex, "_resolve_inputs", return_value=[]),
        patch.object(ex, "_build_mount_plan", return_value=plan),
        patch.object(ex, "_build_environment", return_value={}),
        patch.object(ex, "_build_run_kwargs", return_value={}),
        patch.object(ex, "_start_container", start),
        patch.object(ex, "_stream_container_logs"),
        patch.object(ex, "_save_container_logs"),
        patch.object(ex, "_cleanup_mount_plan"),
        patch.object(ex, "emit_update"),
        patch.object(ssh_module, "maybe_upload_artifacts"),
    ):
        return ex.run(_task(interactive), tmp_path / "out")


def _signalled_then(ex: SSHExecutor, kind: str, end: str) -> Callable[[Any], None]:
    """On the first reload, a stop or cancel lands and the container ends as the
    request makes it: stopped, or removed."""

    def on_reload(container: Any) -> None:
        if container.status == "running":
            getattr(ex, kind)(_TASK_ID)
            container.status = "exited"
        if end == "removed":
            raise NotFound("gone")

    return on_reload


@pytest.mark.parametrize("interactive", [False, True], ids=["batch", "interactive"])
@pytest.mark.parametrize("end", ["exited", "removed"])
def test_a_stop_while_the_session_runs_succeeds(
    executor: SSHExecutor, tmp_path: Path, interactive: bool, end: str
) -> None:
    container = _container(_signalled_then(executor, "stop", end))

    result = _run(executor, interactive, tmp_path, container)

    assert result.exit_code == 0


@pytest.mark.parametrize("interactive", [False, True], ids=["batch", "interactive"])
@pytest.mark.parametrize("end", ["exited", "removed"])
def test_a_cancel_while_the_session_runs_is_cancelled(
    executor: SSHExecutor, tmp_path: Path, interactive: bool, end: str
) -> None:
    container = _container(_signalled_then(executor, "cancel", end))

    with pytest.raises(TaskCancelledError):
        _run(executor, interactive, tmp_path, container)


@pytest.mark.parametrize("interactive", [False, True], ids=["batch", "interactive"])
def test_a_session_container_removed_unasked_fails(
    executor: SSHExecutor, tmp_path: Path, interactive: bool
) -> None:
    def removed(_container: Any) -> None:
        raise NotFound("gone")

    with pytest.raises(ExecutionError):
        _run(executor, interactive, tmp_path, _container(removed))


def test_a_batch_container_exiting_unasked_reports_its_exit_code(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    def exits(container: Any) -> None:
        container.status = "exited"

    with pytest.raises(ExecutionError, match="exited with code 143"):
        _run(executor, False, tmp_path, _container(exits))


@pytest.mark.parametrize("interactive", [False, True], ids=["batch", "interactive"])
def test_a_stop_before_the_session_starts_succeeds_without_a_container(
    executor: SSHExecutor, tmp_path: Path, interactive: bool
) -> None:
    executor.stop(_TASK_ID)
    start = MagicMock()

    result = _run(executor, interactive, tmp_path, MagicMock(), start)

    assert result.exit_code == 0
    start.assert_not_called()


def test_a_cancel_before_the_session_starts_is_cancelled_without_a_container(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    executor.cancel(_TASK_ID)
    start = MagicMock()

    with pytest.raises(TaskCancelledError):
        _run(executor, False, tmp_path, MagicMock(), start)

    start.assert_not_called()
