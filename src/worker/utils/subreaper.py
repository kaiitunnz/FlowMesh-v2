"""A per-tree supervisor that runs one command and proves its whole process tree reaped.

Run as a fresh, single-threaded process: ``python -I subreaper.py [--receipt-fd FD]
[--grace SEC] -- ARGV...``. The supervisor marks itself a child subreaper, so every
descendant the command leaves behind is re-parented to it, whatever
session or group that descendant moved to. It forks and execs the command, waits for it
to exit (or for its own SIGTERM, which ends the command early), then terminates and
reaps whatever remains until it has no child left, escalating from TERM to KILL after
the grace. It keeps draining for as long as anything remains, so the tree never loses
its owner; the caller bounds how long it waits.

Only that drain to no-children exits :data:`REAPED`; the command's own exit status, and
whether the supervisor ended it, go to the receipt instead. Any other exit after the
fork, or death by a signal, proves nothing about the tree.
"""

import argparse
import contextlib
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

REAPED = 86  # the drain reached no children: the whole tree is gone
UNSUPPORTED = 88  # the kernel would not make the supervisor a child subreaper

DEFAULT_GRACE_SEC = 2.0

_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_WAIT_STEP_SEC = 0.05
# Numbered alike on every Linux architecture.
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434
_SCRIPT = Path(__file__).resolve()


def supervised_argv(
    argv: list[str],
    *,
    receipt_fd: int | None = None,
    grace_sec: float = DEFAULT_GRACE_SEC,
) -> list[str]:
    """The argv that runs ``argv`` under a fresh supervisor.

    ``receipt_fd`` is a pipe the supervisor inherits and writes its receipt to; the
    command never holds it.
    """
    head = [sys.executable, "-I", _SCRIPT.as_posix()]
    if receipt_fd is not None:
        head += ["--receipt-fd", str(receipt_fd)]
    return [*head, "--grace", str(grace_sec), "--", *argv]


def reap_proved(returncode: int | None) -> bool:
    """Whether a supervisor's exit status proves its tree reaped.

    A supervisor the kernel would not make a subreaper, or one a SIGTERM ended, never
    forked: the signal is blocked from before the fork, so it can end the supervisor
    only while there is no tree.
    """
    return returncode in (REAPED, UNSUPPORTED, -signal.SIGTERM)


class Receipt(NamedTuple):
    """What a supervisor reports about the command it ran."""

    status: int | None  # the command's exit status, negative for a signal
    ended: bool  # whether the supervisor ended the command before it exited


def read_receipt(fd: int) -> Receipt | None:
    """Read and close a receipt pipe once its supervisor exited; None without one."""
    chunks: list[bytes] = []
    try:
        while chunk := os.read(fd, 4096):
            chunks.append(chunk)
        raw = json.loads(b"".join(chunks))
        return Receipt(status=raw["status"], ended=bool(raw["ended"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    finally:
        os.close(fd)


def end_supervised(proc: subprocess.Popen[Any], timeout_sec: float) -> bool:
    """Ask a running supervisor to end its tree and wait for its proof.

    The supervisor is never killed here: killing it would orphan the tree it is draining
    and destroy the proof. A supervisor that does not finish within ``timeout_sec``
    leaves the tree unproved and the caller keeps ``proc`` to try again.
    """
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout_sec)
    return reap_proved(proc.returncode)


def primary_child(supervisor_pid: int) -> int | None:
    """The command a supervisor runs: its earliest-started child, ahead of any child it
    adopted later."""
    children: list[tuple[int, int]] = []
    for pid, (ppid, started) in _process_table().items():
        if ppid == supervisor_pid:
            children.append((started, pid))
    return min(children)[1] if children else None


def _become_subreaper(libc: ctypes.CDLL) -> bool:
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        return False
    flag = ctypes.c_int(0)
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(flag), 0, 0, 0) != 0:
        return False
    return flag.value == 1


def _process_table() -> dict[int, tuple[int, int]]:
    """Every visible process's parent pid and start time."""
    table: dict[int, tuple[int, int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
            # A command name may hold spaces or parentheses; fields follow the last ")".
            fields = stat[stat.rfind(")") + 2 :].split()
            table[int(entry)] = (int(fields[1]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue
    return table


def _descendants() -> set[int]:
    parents = {pid: ppid for pid, (ppid, _) in _process_table().items()}
    found: set[int] = set()
    frontier = [os.getpid()]
    while frontier:
        parent = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == parent and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return found


def _signal_tree(libc: ctypes.CDLL, pids: set[int], sig: int) -> None:
    """Signal every pid of ``pids`` that is still in the tree.

    Pinning each pid first, then confirming it is still in the tree, keeps a recycled
    pid that left the tree from being signalled. The raw syscalls stand in for
    ``os.pidfd_open``, which an interpreter built against an older libc omits; a kernel
    without pidfds gets a plain kill of a pid found in the tree a moment before.
    """
    pinned: dict[int, int] = {}
    unpinned: set[int] = set()
    for pid in pids:
        if (fd := libc.syscall(_SYS_PIDFD_OPEN, pid, 0)) >= 0:
            pinned[pid] = fd
        elif ctypes.get_errno() == errno.ENOSYS:
            unpinned.add(pid)
    try:
        members = _descendants()
        for pid, fd in pinned.items():
            if pid in members:
                libc.syscall(_SYS_PIDFD_SEND_SIGNAL, fd, sig, None, 0)
        for pid in unpinned & members:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)
    finally:
        for fd in pinned.values():
            os.close(fd)


class _Supervisor:
    def __init__(self, libc: ctypes.CDLL, primary: int) -> None:
        self.libc = libc
        self.primary = primary
        self.status: int | None = None
        self.ended = False

    def reap_ready(self) -> bool:
        """Reap every exited child; return whether any child remains."""
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return False
            if pid == 0:
                return True
            if pid == self.primary:
                self.status = os.waitstatus_to_exitcode(status)

    def await_primary(self) -> None:
        """Wait for the command to exit or for the supervisor's own SIGTERM."""
        while self.reap_ready() and self.status is None:
            got = signal.sigtimedwait({signal.SIGCHLD, signal.SIGTERM}, 1.0)
            if got is not None and got.si_signo == signal.SIGTERM:
                self.ended = True
                return

    def drain(self, grace_sec: float) -> None:
        """Terminate and reap the remaining tree until no child is left."""
        start = time.monotonic()
        terminated: set[int] = set()
        while self.reap_ready():
            pids = _descendants()
            if time.monotonic() - start >= grace_sec:
                _signal_tree(self.libc, pids, signal.SIGKILL)
            elif fresh := pids - terminated:
                _signal_tree(self.libc, fresh, signal.SIGTERM)
                terminated |= fresh
            signal.sigtimedwait({signal.SIGCHLD}, _WAIT_STEP_SEC)


def _write_receipt(fd: int, supervisor: _Supervisor) -> None:
    payload = {"status": supervisor.status, "ended": supervisor.ended}
    with os.fdopen(fd, "wb") as pipe:
        pipe.write(json.dumps(payload).encode())


def _exec_child(argv: list[str], mask: Iterable[int]) -> None:
    try:
        # The interpreter ignores these at startup and an exec keeps that, so the
        # command gets the defaults any process starts with.
        for sig in (signal.SIGPIPE, signal.SIGXFSZ):
            signal.signal(sig, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        os.execv(argv[0], argv)  # nosec B606 - argv list from the worker, no shell
    except OSError as exc:
        os.write(2, f"subreaper: cannot exec {argv[0]}: {exc}\n".encode())
    os._exit(127)


def main() -> None:
    parser = argparse.ArgumentParser(prog="subreaper")
    parser.add_argument("--receipt-fd", type=int)
    parser.add_argument("--grace", type=float, default=DEFAULT_GRACE_SEC)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        parser.error("a command to supervise is required")
    libc = ctypes.CDLL(None, use_errno=True)
    if not _become_subreaper(libc):
        os.write(2, b"subreaper: the kernel refused PR_SET_CHILD_SUBREAPER\n")
        os._exit(UNSUPPORTED)
    if args.receipt_fd is not None:
        os.set_inheritable(args.receipt_fd, False)
    with contextlib.suppress(PermissionError):
        os.setsid()
    # Blocked before the fork, so neither signal is lost or takes its default action
    # between the fork and the wait; the command gets the original mask back.
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD, signal.SIGTERM})
    primary = os.fork()
    if primary == 0:
        _exec_child(argv, mask)
    # The command alone owns the stdio pipes, so its reader sees end-of-file when the
    # command closes them.
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.close(devnull)
    supervisor = _Supervisor(libc, primary)
    supervisor.await_primary()
    supervisor.drain(args.grace)
    if args.receipt_fd is not None:
        # The tree is gone either way; a missing receipt reads as an unknown status.
        with contextlib.suppress(OSError):
            _write_receipt(args.receipt_fd, supervisor)
    os._exit(REAPED)


if __name__ == "__main__":
    main()
