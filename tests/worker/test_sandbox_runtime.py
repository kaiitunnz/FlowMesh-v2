"""The worker-local sandbox fence: what a command may touch, and what it may not."""

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
    SandboxRuntimeProfile,
)
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
