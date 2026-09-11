"""The worker-local sandbox fence: what a command may touch, and what it may not."""

import sys
import time
from pathlib import Path

import pytest

from shared.sandbox import (
    SandboxCommand,
    SandboxDenied,
    SandboxRuntimeProfile,
)
from worker.sandbox.runtime import PosixProcessSandbox, landlock_abi

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


def run(runtime, profile, root: Path, *argv: str, **kwargs):
    return runtime.run(root, SandboxCommand(argv=argv, **kwargs), profile)


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
