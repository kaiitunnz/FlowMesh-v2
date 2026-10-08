"""The worker-local sandbox fence: what a command may touch, and what it may not."""

import os
import resource
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

from shared.sandbox import (
    MAX_STREAM_CHARS,
    SandboxCommand,
    SandboxDenied,
    SandboxReapUnproved,
    SandboxRuntimeProfile,
)
from tests.worker.processes import recorded_pids, running
from worker.sandbox import runtime as sandbox_runtime
from worker.sandbox._launcher import _NR, _filter_program
from worker.sandbox.runtime import (
    _DRAIN_CHUNK_CHARS,
    PosixProcessSandbox,
    _Streams,
    landlock_abi,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the fence is Linux-only"
)

_HAS_LANDLOCK = landlock_abi() > 0
_needs_landlock = pytest.mark.skipif(
    not _HAS_LANDLOCK, reason="this kernel has no Landlock"
)


@pytest.fixture
def runtime() -> PosixProcessSandbox:
    return PosixProcessSandbox()


@pytest.fixture
def profile() -> SandboxRuntimeProfile:
    return SandboxRuntimeProfile()


def run(runtime, profile, root: Path, *argv: str, egress: bool = False, **kwargs):
    return runtime.run(root, SandboxCommand(argv=argv, **kwargs), profile, egress)


@pytest.fixture
def listener():
    """A loopback endpoint an egress-enabled command can actually reach."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(4)

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            conn.sendall(b"REACHED")
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname()[1]
    server.close()


def _connect(port: int) -> str:
    return (
        "import socket; s = socket.socket(); s.settimeout(5); "
        f"s.connect(('127.0.0.1', {port})); print(s.recv(16).decode())"
    )


def test_a_command_reads_and_writes_its_own_workspace(runtime, profile, tmp_path):
    result = run(
        runtime, profile, tmp_path, "sh", "-c", "echo hello > out.txt && cat out.txt"
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert (tmp_path / "out.txt").read_text().strip() == "hello"


def test_an_interpreter_starts_inside_the_fence(runtime, profile, tmp_path):
    result = run(runtime, profile, tmp_path, sys.executable, "-c", "print('ok')")

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"


@_needs_landlock
def test_a_command_cannot_read_another_activations_tree(runtime, profile, tmp_path):
    workspace = tmp_path / "mine"
    workspace.mkdir()
    peer = tmp_path / "peer"
    peer.mkdir()
    (peer / "secret.txt").write_text("SECRET")

    result = run(runtime, profile, workspace, "cat", (peer / "secret.txt").as_posix())

    assert result.exit_code != 0
    assert "SECRET" not in result.stdout


@_needs_landlock
def test_a_command_cannot_write_another_activations_tree(runtime, profile, tmp_path):
    workspace = tmp_path / "mine"
    workspace.mkdir()
    peer = tmp_path / "peer"
    peer.mkdir()
    target = peer / "secret.txt"
    target.write_text("SECRET")

    result = run(
        runtime, profile, workspace, "sh", "-c", f"echo pwned > {target.as_posix()}"
    )

    assert result.exit_code != 0
    assert target.read_text() == "SECRET"


@_needs_landlock
def test_a_command_cannot_read_a_peer_process_environment(runtime, profile, tmp_path):
    result = run(runtime, profile, tmp_path, "sh", "-c", "cat /proc/1/environ")

    assert result.exit_code != 0


def test_a_command_cannot_open_a_network_connection(runtime, profile, tmp_path):
    result = run(
        runtime,
        profile,
        tmp_path,
        sys.executable,
        "-c",
        "import socket; socket.socket().connect(('1.1.1.1', 80)); print('EGRESS')",
    )

    assert result.exit_code != 0
    assert "EGRESS" not in result.stdout


def test_a_command_past_its_deadline_is_killed(runtime, profile, tmp_path):
    result = run(runtime, profile, tmp_path, "sh", "-c", "sleep 30", timeout_sec=1.0)

    assert result.timed_out
    assert result.exit_code == -1


def test_a_command_deadline_is_capped_by_the_envelope(runtime, tmp_path):
    profile = SandboxRuntimeProfile(command_timeout_sec=1.0)
    result = run(runtime, profile, tmp_path, "sh", "-c", "sleep 5", timeout_sec=60.0)

    assert result.timed_out
    assert result.exit_code == -1


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0.0, -1.0])
def test_a_command_deadline_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError, match="timeout_sec"):
        SandboxCommand(argv=("true",), timeout_sec=timeout)


def test_a_background_process_does_not_outlive_its_command(runtime, profile, tmp_path):
    marker = tmp_path / "alive.txt"
    result = run(
        runtime,
        profile,
        tmp_path,
        "sh",
        "-c",
        f"(while true; do echo x >> {marker.as_posix()}; sleep 0.05; done) &"
        " echo started",
    )

    # The command returns when it finishes, not when the process it left behind gives
    # up its pipes, and the deadline is not what ends it.
    assert not result.timed_out
    assert result.stdout.strip() == "started"
    time.sleep(0.3)
    settled = marker.stat().st_size if marker.exists() else 0
    time.sleep(0.5)

    assert (marker.stat().st_size if marker.exists() else 0) == settled


def test_a_command_naming_no_program_is_denied(runtime, profile, tmp_path):
    with pytest.raises(SandboxDenied):
        runtime.run(tmp_path, SandboxCommand(argv=()), profile)


def test_a_command_naming_an_absent_program_is_denied(runtime, profile, tmp_path):
    with pytest.raises(SandboxDenied):
        run(runtime, profile, tmp_path, "flowmesh-no-such-program")


def test_the_fence_still_denies_egress_without_landlock(profile, tmp_path):
    """A kernel without Landlock keeps the seccomp and envelope layers."""
    runtime = PosixProcessSandbox(abi=0)

    result = run(
        runtime,
        profile,
        tmp_path,
        sys.executable,
        "-c",
        "import socket; socket.socket().connect(('1.1.1.1', 80)); print('EGRESS')",
    )

    assert result.exit_code != 0
    assert "EGRESS" not in result.stdout
    assert run(runtime, profile, tmp_path, "sh", "-c", "echo ok").stdout.strip() == "ok"


def test_a_newline_free_flood_does_not_buffer_unbounded_in_the_worker(
    runtime, profile, tmp_path
):
    """The kept prefix is bounded, and so is what the worker holds to produce it."""
    flood = MAX_STREAM_CHARS * 4
    result = run(
        runtime,
        profile,
        tmp_path,
        sys.executable,
        "-c",
        f"import sys; sys.stdout.write('x' * {flood})",
    )

    assert result.exit_code == 0
    assert len(result.stdout) == MAX_STREAM_CHARS


class _FloodPipe:
    """A pipe with no line breaks, recording the largest read it was asked for."""

    def __init__(self, total: int) -> None:
        self._left = total
        self.largest_read = 0
        self.readline_calls = 0

    def read(self, size: int = -1) -> str:
        # A reader asking for everything is what buffers a flood in worker memory.
        want = self._left if size is None or size < 0 else min(size, self._left)
        self.largest_read = max(self.largest_read, want)
        self._left -= want
        return "x" * want

    def readline(self) -> str:
        self.readline_calls += 1
        self.largest_read = max(self.largest_read, self._left)
        text, self._left = "x" * self._left, 0
        return text

    def close(self) -> None:
        return None


def test_the_drain_reads_in_bounded_chunks():
    """A stream that never breaks a line must not be pulled into memory whole."""
    pipe = _FloodPipe(MAX_STREAM_CHARS * 4)
    kept: list[str] = []

    _Streams._drain(pipe, kept)  # type: ignore[arg-type]

    assert pipe.readline_calls == 0
    assert pipe.largest_read <= _DRAIN_CHUNK_CHARS
    assert len("".join(kept)) == MAX_STREAM_CHARS


def test_an_egress_authorized_command_reaches_the_network(
    runtime, profile, tmp_path, listener
):
    result = run(
        runtime,
        profile,
        tmp_path,
        sys.executable,
        "-c",
        _connect(listener),
        egress=True,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "REACHED"


def test_the_same_command_without_the_opt_in_is_denied(
    runtime, profile, tmp_path, listener
):
    result = run(runtime, profile, tmp_path, sys.executable, "-c", _connect(listener))

    assert result.exit_code != 0
    assert "REACHED" not in result.stdout


def test_egress_is_authorized_without_landlock_too(profile, tmp_path, listener):
    """Relaxing the fence relaxes the seccomp layer, not only the Landlock one."""
    runtime = PosixProcessSandbox(abi=0)

    result = run(
        runtime,
        profile,
        tmp_path,
        sys.executable,
        "-c",
        _connect(listener),
        egress=True,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "REACHED"


@_needs_landlock
def test_an_egress_authorized_command_keeps_every_other_fence(
    runtime, profile, tmp_path
):
    """Only the network layers move: the workspace confinement is unchanged."""
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    result = run(
        runtime,
        profile,
        tmp_path,
        "sh",
        "-c",
        f"cat {outside.as_posix()}",
        egress=True,
    )

    assert result.exit_code != 0
    assert "secret" not in result.stdout


def test_an_egress_authorized_command_still_runs_ordinary_commands(
    runtime, profile, tmp_path
):
    """A filter whose jumps land wrong would deny every syscall, not just sockets."""
    result = run(
        runtime, profile, tmp_path, "sh", "-c", "echo ok > f && cat f", egress=True
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"


def test_the_denying_filter_is_unchanged_byte_for_byte():
    """The denied program is the fence of record; an offset slip must fail here."""
    ld, jeq, ret = 0x20, 0x15, 0x06
    deny, allow = 0x00050000 | 13, 0x7FFF0000
    nr = _NR["x86_64"]

    def stmt(code: int, k: int) -> bytes:
        return struct.pack("HBBI", code, 0, 0, k)

    def jump(k: int, jt: int, jf: int) -> bytes:
        return struct.pack("HBBI", jeq, jt, jf, k)

    expected = b"".join(
        [
            stmt(ld, 4),
            jump(nr["audit"], 0, 6),
            stmt(ld, 0),
            jump(nr["io_uring_setup"], 4, 0),
            jump(nr["socket"], 0, 4),
            stmt(ld, 16),
            jump(2, 1, 0),
            jump(10, 0, 1),
            stmt(ret, deny),
            stmt(ret, allow),
        ]
    )

    assert _filter_program(nr, False) == expected
    assert _filter_program(nr, True) != expected


_RING_PROBE = (
    "import ctypes; libc = ctypes.CDLL('libc.so.6', use_errno=True); "
    "libc.syscall(425, 8, 0); print('errno', ctypes.get_errno())"
)


@pytest.mark.parametrize("egress", [False, True])
def test_io_uring_stays_denied_in_both_modes(runtime, profile, tmp_path, egress):
    """A ring runs its operations in kernel context, where this filter no longer sees
    them, so relaxing the network must not relax it."""
    result = run(
        runtime, profile, tmp_path, sys.executable, "-c", _RING_PROBE, egress=egress
    )

    # EACCES is the filter's own denial; a reachable io_uring_setup fails differently.
    assert result.stdout.strip() == "errno 13"


# Holds the command until its background process has armed itself.
_READY = "while [ ! -s ready ]; do sleep 0.01; done; "


def test_a_command_completes_only_once_a_detached_writer_is_reaped(
    runtime, profile, tmp_path
):
    result = run(
        runtime,
        profile,
        tmp_path,
        "sh",
        "-c",
        "setsid sh -c 'echo x >> log; echo > ready; while :; do echo x >> log; "
        f"sleep 0.02; done' & echo $! > pids; {_READY}exit 0",
    )

    assert result.exit_code == 0
    assert not any(running(pid) for pid in recorded_pids(tmp_path / "pids"))
    size = (tmp_path / "log").stat().st_size
    time.sleep(0.2)
    assert (tmp_path / "log").stat().st_size == size


def test_a_timed_out_command_ignoring_term_is_reaped(profile, tmp_path):
    runtime = PosixProcessSandbox(reap_grace_sec=0.3)

    result = run(
        runtime,
        profile,
        tmp_path,
        "sh",
        "-c",
        "trap '' TERM; setsid sh -c 'trap \"\" TERM; echo $$ >> pids; exec sleep 30' &"
        " echo $$ >> pids; sleep 30",
        timeout_sec=0.5,
    )

    assert result.timed_out
    assert not any(running(pid) for pid in recorded_pids(tmp_path / "pids"))


def test_a_command_whose_tree_is_not_proved_reaped_fails(profile, tmp_path):
    runtime = PosixProcessSandbox(reap_grace_sec=30.0, reap_budget_sec=0.3)

    with pytest.raises(SandboxReapUnproved) as raised:
        run(
            runtime,
            profile,
            tmp_path,
            "sh",
            "-c",
            'setsid sh -c \'trap "" TERM; echo $$ > pids; echo > ready; '
            "exec sleep 30' & "
            f"{_READY}exit 0",
            timeout_sec=0.5,
        )
    retry = raised.value.retry
    assert retry is not None
    assert not retry()
    for pid in recorded_pids(tmp_path / "pids"):
        os.kill(pid, signal.SIGKILL)
    assert retry()


@_needs_landlock
def test_a_supervised_command_still_cannot_read_proc(runtime, profile, tmp_path):
    result = run(runtime, profile, tmp_path, "cat", "/proc/self/status")

    assert result.exit_code != 0
    assert "Pid:" not in result.stdout


def test_a_command_finished_before_its_deadline_is_not_timed_out(profile, tmp_path):
    runtime = PosixProcessSandbox(reap_grace_sec=1.0)

    # The supervisor is draining a TERM-ignoring leftover when the deadline passes.
    result = run(
        runtime,
        profile,
        tmp_path,
        "sh",
        "-c",
        "setsid sh -c 'trap \"\" TERM; echo > ready; exec sleep 30' & "
        f"{_READY}exit 0",
        timeout_sec=0.3,
    )

    assert not result.timed_out
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "script, status",
    [
        ("trap 'kill 0' EXIT; echo done; exit 0", -signal.SIGTERM),
        ("kill -HUP 0; exit 0", -signal.SIGHUP),
        ("kill -USR1 $PPID; kill -ALRM $PPID; kill -TERM $PPID; exit 0", 0),
    ],
)
def test_a_command_signalling_its_group_or_parent_does_not_end_its_supervisor(
    runtime, profile, tmp_path, script, status
):
    result = run(runtime, profile, tmp_path, "sh", "-c", script)

    assert not result.timed_out
    assert result.exit_code == status


def test_a_failure_after_the_command_started_keeps_its_tree_owned(
    runtime, profile, tmp_path, monkeypatch
):
    def lost(proc):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(sandbox_runtime, "_Streams", lost)

    with pytest.raises(SandboxReapUnproved, match="lost track") as raised:
        run(runtime, profile, tmp_path, "sh", "-c", "exit 0")
    assert raised.value.retry is not None
    assert raised.value.retry()


def test_a_command_runs_while_the_worker_holds_more_than_fd_setsize_fds(
    runtime, profile, tmp_path
):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 2048:
        pytest.skip("the hard open-file limit is below 2048")
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, hard))
    held = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
    try:
        result = run(runtime, profile, tmp_path, "true")
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    assert result.exit_code == 0
