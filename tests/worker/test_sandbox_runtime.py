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

# Reports whether the command can open an IP socket at all, and that a Unix socket and
# the filesystem are still its own.
_PROBE = (
    "import socket\n"
    "try:\n"
    "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    "    print('ip-open')\n"
    "except OSError as exc:\n"
    "    print('ip-denied', exc.errno)\n"
    "socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
    "print('unix-open')\n"
)


def test_command_runs_against_the_session_root(tmp_path: Path) -> None:
    result = PosixProcessSandbox().run(
        tmp_path, SandboxCommand(argv=("sh", "-c", "echo written > note.txt"))
    )
    assert result.exit_code == 0
    assert (tmp_path / "note.txt").read_text() == "written\n"


def test_command_cannot_open_an_ip_socket(tmp_path: Path) -> None:
    result = PosixProcessSandbox().run(
        tmp_path, SandboxCommand(argv=("python3", "-c", _PROBE))
    )
    assert result.exit_code == 0, result.stderr
    # Egress is refused at socket creation, so no address can be reached at all.
    assert result.stdout.startswith("ip-denied")
    # A local socket and the filesystem stay usable inside the session's own tree.
    assert "unix-open" in result.stdout


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


def test_sessions_run_their_commands_in_the_posix_runtime() -> None:
    assert isinstance(build_sandbox_runtime(), PosixProcessSandbox)


def test_command_cannot_open_an_io_uring(tmp_path: Path) -> None:
    probe = (
        "import ctypes, os\n"
        "libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
        "params = ctypes.create_string_buffer(120)\n"
        "rc = libc.syscall(425, 8, ctypes.byref(params))\n"
        "print('ring', rc, ctypes.get_errno())\n"
    )
    result = PosixProcessSandbox().run(
        tmp_path, SandboxCommand(argv=("python3", "-c", probe))
    )
    assert result.exit_code == 0, result.stderr
    # A ring would submit socket work in kernel context, unchecked by the filter.
    assert result.stdout.split()[1] == "-1"
