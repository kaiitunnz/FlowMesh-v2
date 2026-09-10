"""Running a sandbox command against a private tree without reaching a network."""

from pathlib import Path

import pytest

from worker.sandbox import (
    MAX_STREAM_CHARS,
    PosixProcessSandbox,
    SandboxCommand,
    SandboxUnavailable,
    build_sandbox_runtime,
)

_PROBE = (
    "import socket; s = socket.socket(); s.settimeout(2); "
    'print(s.connect_ex(("1.1.1.1", 80)))'
)
_ENETUNREACH = 101


def test_command_runs_against_the_session_root(tmp_path: Path) -> None:
    result = PosixProcessSandbox().run(
        tmp_path, SandboxCommand(argv=("sh", "-c", "echo written > note.txt"))
    )
    assert result.exit_code == 0
    assert (tmp_path / "note.txt").read_text() == "written\n"


def test_command_reaches_no_network(tmp_path: Path) -> None:
    result = PosixProcessSandbox().run(
        tmp_path, SandboxCommand(argv=("python3", "-c", _PROBE))
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == str(_ENETUNREACH)


def test_command_inherits_no_worker_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_SECRET", "leaked")
    result = PosixProcessSandbox().run(
        tmp_path,
        SandboxCommand(argv=("sh", "-c", 'printf "%s" "${WORKER_SECRET-unset}"')),
    )
    assert result.stdout == "unset"


def test_failing_command_reports_its_exit_code(tmp_path: Path) -> None:
    result = PosixProcessSandbox().run(
        tmp_path, SandboxCommand(argv=("sh", "-c", "exit 3"))
    )
    assert result.exit_code == 3
    assert not result.timed_out


def test_timed_out_command_is_terminal(tmp_path: Path) -> None:
    result = PosixProcessSandbox().run(
        tmp_path, SandboxCommand(argv=("sleep", "5"), timeout_sec=0.5)
    )
    assert result.timed_out
    assert result.exit_code == -1


def test_stream_capture_is_bounded(tmp_path: Path) -> None:
    result = PosixProcessSandbox().run(
        tmp_path,
        SandboxCommand(
            argv=("python3", "-c", f'print("x" * {MAX_STREAM_CHARS * 2})'),
        ),
    )
    assert len(result.stdout) == MAX_STREAM_CHARS


def test_unavailable_program_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SandboxUnavailable):
        PosixProcessSandbox().run(
            tmp_path, SandboxCommand(argv=("definitely-not-a-program",))
        )


def test_unknown_runtime_name_falls_back_to_the_default() -> None:
    assert isinstance(build_sandbox_runtime("containerd"), PosixProcessSandbox)
