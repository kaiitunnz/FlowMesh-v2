"""A batch SSH task ends with its container: it fails with the container's exit code,
and a breach of its output limit fails it, whether its output is copied out of the
container or mounted directly."""

import io
import tarfile
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, NotFound
from docker.models.containers import Container

from shared.schemas.result import SSHResult
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_live_worker_config, make_ssh_executor
from worker.executors import ssh_executor as ssh_module
from worker.executors.base_executor import ExecutionError, TaskCancelledError
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session import DockerSessionBackend
from worker.executors.ssh_session.backends.docker import DockerSession, SSHMountPlan

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
    ex = make_ssh_executor(make_live_worker_config(tmp_path), lifecycle=None)
    _backend(ex)._docker = MagicMock()
    return ex


def _backend(ex: SSHExecutor) -> DockerSessionBackend:
    backend = ex.backend
    assert isinstance(backend, DockerSessionBackend)
    return backend


def _copy_session(ex: SSHExecutor, container: Any) -> DockerSession:
    """A session whose output is copied out of ``container`` from ``/out``."""
    plan = SSHMountPlan(
        volumes=[],
        staged_input_specs=[],
        create_dirs=[],
        direct_output_path=None,
        copy_output_path="/out",
        staged_inputs_dir=None,
        staged_inputs_volume=None,
    )
    return DockerSession(MagicMock(), container, plan, None, MagicMock(), ex._signals)


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
    start_failure: Exception | None = None,
) -> SSHResult:
    plan = MagicMock()
    plan.copy_output_path = None if direct_output is not None else "/out"
    plan.direct_output_path = direct_output
    backend = _backend(ex)
    with (
        patch.object(backend, "prepare"),
        patch.object(backend, "_resolve_noninteractive_command", return_value=["true"]),
        patch.object(ssh_module, "resolve_inputs", return_value=[]),
        patch.object(backend, "_build_mount_plan", return_value=plan),
        patch.object(backend, "_build_environment", return_value={}),
        patch.object(backend, "_build_run_kwargs", return_value={}),
        patch.object(
            backend,
            "_start_container",
            return_value=(container, None),
            side_effect=start_failure,
        ),
        patch.object(DockerSession, "_stream_container_logs", side_effect=stream_logs),
        patch.object(
            DockerSession,
            "save_logs",
            autospec=True,
            side_effect=DockerSession.save_logs if save_logs else None,
        ),
        patch.object(
            DockerSession,
            "collect_output",
            autospec=True,
            side_effect=DockerSession.collect_output if copy else None,
        ),
        patch.object(DockerSessionBackend, "_cleanup_mount_plan") as cleanup,
        patch.object(ex, "emit_update"),
        patch.object(ssh_module, "maybe_upload_artifacts"),
    ):
        try:
            return ex.run(_task(max_bytes, ttl_sec), tmp_path / "out")
        finally:
            cleanup.assert_called_once_with(backend._docker, plan)


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


class _Archive:
    """An archive stream as Docker returns it: a generator whose connection is
    released only when its body ends, so closing it before its first read releases
    nothing."""

    def __init__(
        self,
        data: bytes,
        chunk: int = 1024,
        on_read: Callable[[int], Any] = lambda _: None,
    ) -> None:
        self._on_read = on_read
        self._stream = self._generate(data, chunk)
        self.chunks_read = 0
        self.released = False

    def _generate(self, data: bytes, chunk: int) -> Generator[bytes]:
        try:
            for offset in range(0, len(data), chunk):
                self.chunks_read += 1
                self._on_read(self.chunks_read)
                yield data[offset : offset + chunk]
        finally:
            self.released = True

    def __iter__(self) -> "_Archive":
        return self

    def __next__(self) -> bytes:
        return next(self._stream)

    def close(self) -> None:
        self._stream.close()


def _output_archive(size: int) -> _Archive:
    """The tar stream Docker returns for an output directory holding one file."""
    return _Archive(_archive_of({"result.bin": size}))


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


def test_a_session_whose_container_fails_to_start_cleans_its_mounts(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    with pytest.raises(ExecutionError, match="no such network"):
        _run(
            executor,
            tmp_path,
            None,
            1000,
            start_failure=ExecutionError("no such network"),
        )


def _archive_of(files: dict[str, int]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, size in files.items():
            info = tarfile.TarInfo(f"out/{name}")
            info.size = size
            archive.addfile(info, io.BytesIO(b"x" * size))
    return buffer.getvalue()


@pytest.mark.parametrize("kind", ["cancel", "stop"])
def test_a_cancel_during_an_output_copy_ends_it(
    executor: SSHExecutor, tmp_path: Path, kind: str
) -> None:
    data = _archive_of({"first.bin": 4096, "second.bin": 4096, "third.bin": 4096})

    def stream() -> Any:
        yield data[:2048]
        getattr(executor, kind)(_TASK_ID)
        yield data[2048:]

    container = MagicMock(spec=Container)
    container.get_archive.return_value = (stream(), {})
    destination = tmp_path / "copied"

    with executor._signals.running(_TASK_ID):
        if kind == "cancel":
            with pytest.raises(TaskCancelledError):
                _copy_session(executor, container).collect_output(destination, None)
        else:
            _copy_session(executor, container).collect_output(destination, None)

    copied = sorted(path.name for path in destination.iterdir())
    assert copied == (
        ["first.bin"] if kind == "cancel" else ["first.bin", "second.bin", "third.bin"]
    )


def test_an_output_copy_reads_its_archive_in_any_chunking(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    data = _archive_of({"a.bin": 70_000, "b.bin": 3})
    container = MagicMock(spec=Container)
    container.get_archive.return_value = (_Archive(data, chunk=777), {})

    _copy_session(executor, container).collect_output(tmp_path / "copied", None)

    assert (tmp_path / "copied" / "a.bin").read_bytes() == b"x" * 70_000
    assert (tmp_path / "copied" / "b.bin").read_bytes() == b"xxx"


@pytest.mark.parametrize("end", ["complete", "breach", "cancel", "failure"])
def test_an_output_copy_releases_its_archive_before_the_container_stops(
    executor: SSHExecutor, tmp_path: Path, end: str
) -> None:
    def on_read(chunks_read: int) -> None:
        if chunks_read == 2 and end == "cancel":
            executor.cancel(_TASK_ID)
        if chunks_read == 2 and end == "failure":
            raise OSError("connection reset")

    archive = _Archive(
        _archive_of({"result.bin": 500 if end == "complete" else 5000}),
        on_read=on_read,
    )
    container = _container(0, output_bytes=0)
    container.get_archive.return_value = (archive, {})
    released_at_stop: list[bool] = []
    container.stop.side_effect = lambda **_: released_at_stop.append(archive.released)

    if end == "complete":
        _run(executor, tmp_path, container, 1000, copy=True)
    else:
        with pytest.raises((ExecutionError, TaskCancelledError, OSError)):
            _run(executor, tmp_path, container, 1000, copy=True)

    assert released_at_stop == [True]


def test_a_cancel_during_the_archive_request_releases_its_stream(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    archive = _Archive(_archive_of({"result.bin": 100_000}))
    container = MagicMock(spec=Container)

    def get_archive(_path: str) -> tuple[_Archive, dict[str, Any]]:
        executor.cancel(_TASK_ID)
        return archive, {}

    container.get_archive.side_effect = get_archive

    with executor._signals.running(_TASK_ID):
        with pytest.raises(TaskCancelledError):
            _copy_session(executor, container).collect_output(tmp_path / "copied", None)

    assert archive.released


def test_a_cancel_ends_an_output_copy_within_one_large_file(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    archive = _Archive(
        _archive_of({"large.bin": 4 * 1024 * 1024}),
        chunk=64 * 1024,
        on_read=lambda n: executor.cancel(_TASK_ID) if n == 2 else None,
    )
    container = MagicMock(spec=Container)
    container.get_archive.return_value = (archive, {})

    with executor._signals.running(_TASK_ID):
        with pytest.raises(TaskCancelledError):
            _copy_session(executor, container).collect_output(tmp_path / "copied", None)

    assert archive.released
    assert archive.chunks_read <= 3


@pytest.mark.parametrize("output", ["copied", "direct"])
def test_a_cancel_as_the_session_ends_cancels_it_whatever_its_output_path(
    executor: SSHExecutor, tmp_path: Path, output: str
) -> None:
    container = _container(0, output_bytes=0)
    container.get_archive.return_value = (_output_archive(10), {})

    def cancel_as_it_exits() -> dict[str, int]:
        executor.cancel(_TASK_ID)
        return {"StatusCode": 0}

    container.wait.side_effect = cancel_as_it_exits
    direct = tmp_path / "direct" if output == "direct" else None
    if direct is not None:
        direct.mkdir()

    with pytest.raises(TaskCancelledError):
        _run(
            executor,
            tmp_path,
            container,
            1000,
            copy=output == "copied",
            direct_output=direct,
        )


def test_a_direct_output_too_deep_to_walk_fails_the_docker_size_check(
    executor: SSHExecutor, tmp_path: Path
) -> None:
    deepest = tmp_path.joinpath(*(["d"] * 70))
    deepest.mkdir(parents=True)
    plan = SSHMountPlan(
        volumes=[],
        staged_input_specs=[],
        create_dirs=[],
        direct_output_path=tmp_path,
        copy_output_path=None,
        staged_inputs_dir=None,
        staged_inputs_volume=None,
    )
    session = DockerSession(
        MagicMock(),
        MagicMock(spec=Container),
        plan,
        None,
        MagicMock(),
        executor._signals,
    )

    with pytest.raises(ExecutionError, match="deeper"):
        session.output_size_bytes()
