"""A batch SSH task ends with its container: it fails with the container's exit code,
and a breach of its output limit fails it, whether its output is copied out of the
container or mounted directly."""

import io
import tarfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, NotFound
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


def _task(max_bytes: int, ttl_sec: int = _TTL_SEC) -> WorkerTaskMessage:
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
                    "ttlSeconds": ttl_sec,
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


def _run(
    ex: SSHExecutor,
    tmp_path: Path,
    container: Any,
    max_bytes: int,
    ttl_sec: int = _TTL_SEC,
    stream_logs: Any = None,
    copy: bool = False,
    direct_output: Path | None = None,
    save_logs: bool = False,
) -> SSHResult:
    plan = MagicMock()
    plan.copy_output_path = None if direct_output is not None else "/out"
    plan.direct_output_path = direct_output
    with (
        patch.object(ex, "prepare"),
        patch.object(ex, "_resolve_noninteractive_command", return_value=["true"]),
        patch.object(ex, "_resolve_inputs", return_value=[]),
        patch.object(ex, "_build_mount_plan", return_value=plan),
        patch.object(ex, "_build_environment", return_value={}),
        patch.object(ex, "_build_run_kwargs", return_value={}),
        patch.object(ex, "_start_container", return_value=(container, None)),
        patch.object(ex, "_stream_container_logs", side_effect=stream_logs),
        patch.object(
            ex,
            "_save_container_logs",
            wraps=ex._save_container_logs if save_logs else None,
        ),
        patch.object(
            ex,
            "_copy_output_directory",
            wraps=ex._copy_output_directory if copy else None,
        ),
        patch.object(ex, "_cleanup_mount_plan"),
        patch.object(ex, "emit_update"),
        patch.object(ssh_module, "maybe_upload_artifacts"),
    ):
        return ex.run(_task(max_bytes, ttl_sec), tmp_path / "out")


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


def test_a_session_past_its_ttl_stops_before_its_logs_are_joined(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    stopped = threading.Event()
    container = _container(0, output_bytes=10, polls=1_000_000)
    container.stop.side_effect = lambda **_: stopped.set()

    def stream_logs(_stream: Any) -> None:
        # A log stream ends only once its container stops.
        stopped.wait(timeout=60.0)

    started = time.monotonic()
    result = _run(
        executor, tmp_path, container, 100, ttl_sec=1, stream_logs=stream_logs
    )

    assert result.exit_code == 0
    assert time.monotonic() - started < _PROMPT_SEC
    container.remove.assert_called_once_with(force=True)


def _output_archive(size: int) -> list[bytes]:
    """The tar stream Docker returns for an output directory holding one file."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("out/result.bin")
        info.size = size
        archive.addfile(info, io.BytesIO(b"x" * size))
    data = buffer.getvalue()
    return [data[i : i + 1024] for i in range(0, len(data), 1024)]


@pytest.mark.parametrize("size", [500, 5000], ids=["within", "past"])
def test_output_copied_out_at_exit_is_held_to_its_limit(
    executor: SSHExecutor, tmp_path: Path, size: int
) -> None:
    # The file lands after the last size check, as the job exits.
    container = _container(0, output_bytes=0)
    container.get_archive.return_value = (_output_archive(size), {})

    if size > 1000:
        with pytest.raises(ExecutionError, match="exceeded maxBytes"):
            _run(executor, tmp_path, container, 1000, copy=True)
    else:
        _run(executor, tmp_path, container, 1000, copy=True)
        copied = tmp_path / "out" / "artifacts" / "result.bin"
        assert copied.read_bytes() == b"x" * size


def test_output_written_directly_at_exit_is_held_to_its_limit(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    output = tmp_path / "direct"
    output.mkdir()
    container = _container(0, output_bytes=0)

    def exits_after_writing() -> None:
        (output / "late.bin").write_bytes(b"x" * 5000)
        container.status = "exited"

    container.reload.side_effect = exits_after_writing

    with pytest.raises(ExecutionError, match="exceeded maxBytes"):
        _run(executor, tmp_path, container, 1000, direct_output=output)


@pytest.mark.parametrize("stopped", [True, False])
def test_output_that_was_never_created_is_empty_only_for_a_stopped_task(
    executor: SSHExecutor, tmp_path: Path, stopped: bool
) -> None:
    container = _container(0, output_bytes=0)
    container.get_archive.side_effect = NotFound("no such path")
    if stopped:
        # The stop lands while the session is still staging its inputs.
        container.reload.side_effect = lambda: (
            executor.stop(_TASK_ID) if container.reload.call_count == 1 else None
        )

    if stopped:
        assert _run(executor, tmp_path, container, 1000, copy=True).exit_code == 0
    else:
        with pytest.raises(ExecutionError, match="Failed to collect SSH output"):
            _run(executor, tmp_path, container, 1000, copy=True)


def test_a_session_log_never_counts_against_its_output_limit(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    # A worker with no results mount writes a session's output where its logs go.
    output = tmp_path / "out" / "artifacts"
    output.mkdir(parents=True)
    (output / "result.bin").write_bytes(b"x" * 500)
    container = _container(0, output_bytes=0)
    container.logs.return_value = b"y" * 2000

    result = _run(
        executor, tmp_path, container, 1000, direct_output=output, save_logs=True
    )

    assert result.exit_code == 0
    assert (output / "logs" / "container_output.log").stat().st_size == 2000


def test_output_written_during_the_stop_at_the_ttl_is_held_to_its_limit(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    output = tmp_path / "direct"
    output.mkdir()
    container = _container(0, output_bytes=0, polls=1_000_000)
    # The job's SIGTERM handler flushes its output as the container stops.
    container.stop.side_effect = lambda **_: (output / "flush.bin").write_bytes(
        b"x" * 5000
    )

    with pytest.raises(ExecutionError, match="exceeded maxBytes"):
        _run(executor, tmp_path, container, 1000, ttl_sec=1, direct_output=output)
