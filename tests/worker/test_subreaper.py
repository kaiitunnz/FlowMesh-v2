"""The per-tree supervisor: only a REAPED exit proves every descendant gone."""

import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import psutil
import pytest

from tests.worker.processes import recorded_pids, running
from worker.utils import subreaper
from worker.utils.subreaper import (
    REAPED,
    UNSUPPORTED,
    Receipt,
    end_supervised,
    read_receipt,
    reap_proved,
    supervised_argv,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="child subreapers are Linux-only"
)

_SH = "/bin/sh"
# Holds the command until its background process has armed itself.
_READY = "while [ ! -s ready ]; do sleep 0.01; done; "


@pytest.fixture
def strays() -> Iterator[list[int]]:
    """Pids a test may leave behind on purpose; killed afterwards either way."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _supervise(
    script_or_argv: str | list[str], tmp_path: Path, grace_sec: float = 2.0
) -> tuple[subprocess.Popen[bytes], int]:
    """Start a supervisor over a shell script (or an argv), returning it and its
    receipt pipe."""
    argv = (
        [_SH, "-c", script_or_argv]
        if isinstance(script_or_argv, str)
        else script_or_argv
    )
    receipt, receipt_end = os.pipe()
    proc = subprocess.Popen(  # nosec B603 - argv list built by the test
        supervised_argv(argv, receipt_fd=receipt_end, grace_sec=grace_sec),
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(receipt_end,),
    )
    os.close(receipt_end)
    return proc, receipt


def _run(
    script: str, tmp_path: Path, grace_sec: float = 2.0
) -> tuple[int, Receipt | None]:
    proc, receipt = _supervise(script, tmp_path, grace_sec)
    proc.wait(30)
    return proc.returncode, read_receipt(receipt)


def test_a_finished_command_reports_its_own_status_apart_from_the_proof(
    tmp_path: Path,
) -> None:
    returncode, receipt = _run("exit 3", tmp_path)

    assert returncode == REAPED and reap_proved(returncode)
    assert receipt == Receipt(status=3, ended=False)


def test_an_unexecutable_command_is_still_a_proved_reap(tmp_path: Path) -> None:
    proc, receipt = _supervise([(tmp_path / "missing").as_posix()], tmp_path)
    proc.wait(30)

    assert proc.returncode == REAPED
    assert read_receipt(receipt) == Receipt(status=127, ended=False)


def test_the_command_gets_default_dispositions_and_no_receipt_pipe(
    tmp_path: Path,
) -> None:
    returncode, receipt = _run(
        "grep -E '^SigIgn' /proc/self/status > ignored; ls /proc/self/fd > fds",
        tmp_path,
    )

    assert returncode == REAPED and receipt is not None and receipt.status == 0
    ignored = int((tmp_path / "ignored").read_text().split()[1], 16)
    assert ignored & ((1 << (signal.SIGPIPE - 1)) | (1 << (signal.SIGXFSZ - 1))) == 0
    assert sorted(int(fd) for fd in (tmp_path / "fds").read_text().split()) <= [
        0,
        1,
        2,
        3,
    ]


def test_a_detached_writer_is_reaped_before_the_proof(tmp_path: Path) -> None:
    script = (
        "setsid sh -c 'echo x >> log; echo > ready; while :; do echo x >> log; "
        "sleep 0.02; done' & "
        f"echo $! > pids; {_READY}exit 0"
    )

    returncode, _ = _run(script, tmp_path)

    assert returncode == REAPED
    assert not any(running(pid) for pid in recorded_pids(tmp_path / "pids"))
    size = (tmp_path / "log").stat().st_size
    time.sleep(0.2)
    assert (tmp_path / "log").stat().st_size == size


def test_a_descendant_ignoring_term_is_killed_after_the_grace(tmp_path: Path) -> None:
    script = (
        "sh -c 'trap \"\" TERM; echo $$ > pids; echo > ready; exec sleep 30' & "
        f"{_READY}exit 0"
    )

    started = time.monotonic()
    returncode, _ = _run(script, tmp_path, grace_sec=0.3)

    assert returncode == REAPED
    assert time.monotonic() - started < 10
    assert not any(running(pid) for pid in recorded_pids(tmp_path / "pids"))


def test_successive_adoption_through_living_intermediate_parents(
    tmp_path: Path,
) -> None:
    # Each level detaches into its own session and leaves a child behind; the middle
    # level outlives its parent and ignores TERM, so its own child is adopted only once
    # it is killed.
    leaf = "echo $$ >> pids; exec sleep 30"
    middle = (
        f"trap '' TERM; echo $$ >> pids; setsid sh -c \"{leaf}\" & "
        "while :; do sleep 1; done"
    )
    script = f"setsid sh -c '{middle}' & echo $! >> pids; sleep 0.3; exit 0"

    returncode, _ = _run(script, tmp_path, grace_sec=0.3)

    assert returncode == REAPED
    pids = recorded_pids(tmp_path / "pids")
    assert len(pids) >= 3
    assert not any(running(pid) for pid in pids)


def test_a_tree_still_draining_is_unproved_until_it_is_reaped(
    tmp_path: Path, strays: list[int]
) -> None:
    script = (
        "sh -c 'trap \"\" TERM; echo $$ > pids; echo > ready; exec sleep 30' & "
        f"{_READY}exit 0"
    )
    proc = subprocess.Popen(  # nosec B603 - argv list built by the test
        supervised_argv([_SH, "-c", script], grace_sec=30.0), cwd=tmp_path
    )
    deadline = time.monotonic() + 10
    while not (tmp_path / "pids").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    strays.extend(recorded_pids(tmp_path / "pids"))

    assert not end_supervised(proc, 0.5)
    # The supervisor still owns the tree, so a later attempt can still prove it.
    assert proc.poll() is None
    os.kill(strays[0], signal.SIGKILL)
    assert end_supervised(proc, 10)


def test_a_killed_supervisor_proves_nothing(tmp_path: Path, strays: list[int]) -> None:
    script = "setsid sh -c 'echo $$ > pids; exec sleep 30' & wait"
    proc = subprocess.Popen(  # nosec B603 - argv list built by the test
        supervised_argv([_SH, "-c", script]), cwd=tmp_path
    )
    deadline = time.monotonic() + 10
    while not (tmp_path / "pids").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    strays.extend(recorded_pids(tmp_path / "pids"))

    proc.kill()
    proc.wait(10)

    assert not reap_proved(proc.returncode)
    assert running(strays[0])


def test_ending_a_running_supervisor_reaps_its_command(tmp_path: Path) -> None:
    script = "setsid sh -c 'echo $$ > pids; exec sleep 30' & sleep 30"
    proc, receipt = _supervise(script, tmp_path, grace_sec=0.3)
    deadline = time.monotonic() + 10
    while not (tmp_path / "pids").exists() and time.monotonic() < deadline:
        time.sleep(0.02)

    assert end_supervised(proc, 10)
    assert read_receipt(receipt) == Receipt(status=-signal.SIGTERM, ended=True)
    assert not any(running(pid) for pid in recorded_pids(tmp_path / "pids"))


def test_only_a_drain_or_an_exit_before_the_fork_proves_a_tree() -> None:
    # Before the supervisor blocks SIGTERM nothing has forked, so dying of it is proof.
    assert reap_proved(-signal.SIGTERM)
    assert not reap_proved(-signal.SIGKILL)
    assert not reap_proved(1)


def test_a_supervisor_the_kernel_will_not_make_a_subreaper_runs_nothing(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran"
    runner = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('s', {subreaper.__file__!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module._become_subreaper = lambda libc: False\n"
        f"sys.argv = ['subreaper', '--', {_SH!r}, '-c', 'touch {marker}']\n"
        "module.main()\n"
    )
    proc = subprocess.run(  # nosec B603 - argv list built by the test
        [sys.executable, "-I", "-c", runner], capture_output=True, timeout=30
    )

    assert proc.returncode == UNSUPPORTED
    assert not marker.exists()


def test_a_pid_whose_pidfd_cannot_be_opened_is_still_signalled() -> None:
    real = ctypes.CDLL(None, use_errno=True)

    class _Refusing:
        """A libc whose pidfd_open fails as a seccomp profile without it does."""

        def syscall(self, number: int, *args: Any) -> int:
            if number == subreaper._SYS_PIDFD_OPEN:
                ctypes.set_errno(errno.EPERM)
                return -1
            return int(real.syscall(number, *args))

    child = subprocess.Popen(["/bin/sleep", "30"])  # nosec B603 - fixed argv
    try:
        # The child is not the test's own descendant, so the tree check is bypassed.
        with patch.object(subreaper, "_descendants", lambda: {child.pid}):
            subreaper._signal_tree(
                cast(ctypes.CDLL, _Refusing()), {child.pid}, signal.SIGKILL
            )
        assert child.wait(5) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()


def _run_without_child_lists(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run a supervisor over ``argv`` as on a kernel that lists no task's children."""
    runner = (
        "import importlib.util, pathlib, sys\n"
        "read = pathlib.Path.read_text\n"
        "def no_children(self, *args, **kwargs):\n"
        "    if self.name == 'children':\n"
        "        raise FileNotFoundError(self)\n"
        "    return read(self, *args, **kwargs)\n"
        "pathlib.Path.read_text = no_children\n"
        f"spec = importlib.util.spec_from_file_location('s', {subreaper.__file__!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"sys.argv = ['subreaper', '--grace', '0.3', '--', *{argv!r}]\n"
        "module.main()\n"
    )
    return subprocess.run(  # nosec B603 - argv list built by the test
        [sys.executable, "-I", "-c", runner], capture_output=True, timeout=60
    )


def test_a_kernel_without_child_lists_still_drains_the_tree() -> None:
    proc = _run_without_child_lists([_SH, "-c", "sleep 30 & setsid sleep 30 & exit 0"])

    assert proc.returncode == REAPED


def test_a_pid_hopping_group_is_ended_without_child_lists() -> None:
    # Each generation forks its successor and exits at once, for 3000 generations;
    # bash, since dash caps the recursion at 1000.
    hopper = (
        "i=0; hop() { i=$((i+1)); [ $i -lt 3000 ] && { hop & }; exit 0; }; hop & exit 0"
    )
    started = time.monotonic()
    proc = _run_without_child_lists(["/bin/bash", "-c", hopper])

    assert proc.returncode == REAPED
    assert time.monotonic() - started < 5


def test_orphans_that_exit_while_the_command_runs_are_reaped_at_once(
    tmp_path: Path,
) -> None:
    script = "for i in $(seq 40); do (sleep 0 &); done; echo > ready; sleep 3"
    proc, receipt = _supervise(script, tmp_path)
    deadline = time.monotonic() + 10
    while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.5)

    zombies = [
        child
        for child in psutil.Process(proc.pid).children()
        if child.status() == psutil.STATUS_ZOMBIE
    ]
    proc.wait(30)
    os.close(receipt)

    assert len(zombies) < 5
