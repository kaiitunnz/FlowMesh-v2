"""A per-tree supervisor that runs one command and proves its whole process tree reaped.

Run as a fresh, single-threaded process: ``python -I -S subreaper.py [--receipt-fd FD]
[--grace SEC] -- ARGV...``. The supervisor marks itself a child subreaper, so every
descendant the command leaves behind is re-parented to it, whatever session or group
that descendant moved to. It forks and execs the command as the leader of its own
process group, waits for it to exit (or for a SIGTERM from the process that launched
the supervisor, which ends the command early), then terminates and reaps whatever
remains until it has no child left, escalating from TERM to KILL after the grace. It
keeps draining for as long as anything remains, so the tree never loses its owner; the
caller bounds how long it waits.

Only that drain to no children exits :data:`REAPED`; the receipt carries the command's
own exit status and whether the supervisor ended it. Any other exit after the fork, or
death by a signal, proves nothing about the tree.
"""

import argparse
import contextlib
import ctypes
import errno
import json
import os
import select
import signal
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

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
# Whether children are read from per-task files or found in a scan of every process;
# a kernel built without the files is scanned. Probed when the supervisor starts.
_CHILDREN_FILES = True
# Held blocked in the supervisor: SIGCHLD and SIGTERM it waits on, the rest it would
# otherwise die of when the command signals its parent.
_HELD = frozenset(
    {
        signal.SIGCHLD,
        signal.SIGTERM,
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGUSR1,
        signal.SIGUSR2,
        signal.SIGALRM,
        signal.SIGPIPE,
    }
)


def supervised_argv(
    argv: list[str],
    *,
    receipt_fd: int | None = None,
    grace_sec: float = DEFAULT_GRACE_SEC,
) -> list[str]:
    """Build the argv that runs ``argv`` under a fresh supervisor.

    ``receipt_fd`` is a pipe the supervisor inherits and writes its receipt to; the
    command never holds it.
    """
    head = [sys.executable, "-I", "-S", _SCRIPT.as_posix()]
    if receipt_fd is not None:
        head += ["--receipt-fd", str(receipt_fd)]
    return [*head, "--grace", str(grace_sec), "--", *argv]


def reap_proved(returncode: int | None) -> bool:
    """Map a supervisor's exit status to whether it proves its tree reaped.

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


def wait_supervisor(proc: subprocess.Popen[Any], timeout_sec: float) -> bool:
    """Wait up to ``timeout_sec`` for a supervisor to exit; return whether it did.

    The wait blocks on the process's pidfd, so it returns as soon as the supervisor
    exits.
    """
    if proc.poll() is not None:
        return True
    libc = ctypes.CDLL(None, use_errno=True)
    if (fd := libc.syscall(_SYS_PIDFD_OPEN, proc.pid, 0)) < 0:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout_sec)
        return proc.poll() is not None
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        poller.poll(timeout_sec * 1000)
    finally:
        os.close(fd)
    # A pidfd is readable once the process exits, which a reap may briefly trail.
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(_WAIT_STEP_SEC)
    return proc.poll() is not None


def end_supervised(proc: subprocess.Popen[Any], timeout_sec: float) -> bool:
    """Ask a running supervisor to end its tree and wait for its proof.

    The supervisor is never killed here: killing it would orphan the tree it is draining
    and destroy the proof. A supervisor that does not finish within ``timeout_sec``
    leaves the tree unproved and the caller keeps ``proc`` to try again.
    """
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(signal.SIGTERM)
        wait_supervisor(proc, timeout_sec)
    return reap_proved(proc.returncode)


def primary_child(supervisor_pid: int) -> int | None:
    """Return the command a supervisor runs: its earliest-started child, ahead of any
    child it adopted later."""
    children = [
        (started, pid)
        for pid, (ppid, started) in _process_table().items()
        if ppid == supervisor_pid
    ]
    return min(children)[1] if children else None


def _become_subreaper(libc: ctypes.CDLL) -> bool:
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        return False
    flag = ctypes.c_int(0)
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(flag), 0, 0, 0) != 0:
        return False
    return flag.value == 1


def _process_table() -> dict[int, tuple[int, int]]:
    """Map every visible process to its parent pid and start time."""
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


def _children_files_available() -> bool:
    """Return whether this kernel lists a task's children under ``/proc``."""
    me = os.getpid()
    try:
        Path(f"/proc/{me}/task/{me}/children").read_text()
    except OSError:
        return False
    return True


def _children(pid: int) -> set[int]:
    """Return a process's direct children, read from each of its threads."""
    if not _CHILDREN_FILES:
        return {child for child, (ppid, _) in _process_table().items() if ppid == pid}
    found: set[int] = set()
    with contextlib.suppress(OSError):
        for task in os.listdir(f"/proc/{pid}/task"):
            with contextlib.suppress(OSError, ValueError):
                found.update(
                    int(child)
                    for child in Path(f"/proc/{pid}/task/{task}/children")
                    .read_text()
                    .split()
                )
    return found


def _descendants() -> set[int]:
    children: Callable[[int], set[int]] = _children
    if not _CHILDREN_FILES:
        by_parent: defaultdict[int, set[int]] = defaultdict(set)
        for child, (ppid, _) in _process_table().items():
            by_parent[ppid].add(child)
        children = by_parent.__getitem__
    found: set[int] = set()
    frontier = [os.getpid()]
    while frontier:
        for child in children(frontier.pop()) - found:
            found.add(child)
            frontier.append(child)
    return found


def _signal_tree(libc: ctypes.CDLL, pids: set[int], sig: int) -> None:
    """Signal every pid of ``pids`` that is still in the tree.

    Pinning each pid first, then confirming it is still in the tree, keeps a recycled
    pid that left the tree from being signalled. The raw syscalls stand in for
    ``os.pidfd_open``, which an interpreter built against an older libc omits; a pid
    that cannot be pinned gets a plain kill if it was in the tree a moment before.
    """
    pinned: dict[int, int] = {}
    unpinned: set[int] = set()
    for pid in pids:
        if (fd := libc.syscall(_SYS_PIDFD_OPEN, pid, 0)) >= 0:
            pinned[pid] = fd
        elif ctypes.get_errno() != errno.ESRCH:
            unpinned.add(pid)
    try:
        members = _descendants()
        for pid, fd in pinned.items():
            if pid in members:
                libc.syscall(_SYS_PIDFD_SEND_SIGNAL, fd, sig, None, 0)
        for pid in unpinned & members:
            with contextlib.suppress(ProcessLookupError, PermissionError):
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
        # The command stays an unreaped zombie until the rest of the tree is gone, so
        # its process group id is not recycled while the drain signals that group.
        self.primary_reaped = False

    def reap_others(self) -> set[int]:
        """Reap every exited child but the command; return the others left."""
        others = _children(os.getpid()) - {self.primary}
        for pid in list(others):
            with contextlib.suppress(ChildProcessError):
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    others.discard(pid)
        return others

    def primary_exited(self) -> bool:
        if self.status is None:
            flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
            if (info := os.waitid(os.P_PID, self.primary, flags)) is None:
                return False
            exited = info.si_code == os.CLD_EXITED
            self.status = info.si_status if exited else -info.si_status
        return True

    def await_primary(self) -> None:
        """Wait for the command to exit or for its launcher's SIGTERM."""
        launcher = os.getppid()
        while True:
            got = signal.sigtimedwait({signal.SIGCHLD, signal.SIGTERM}, 1.0)
            self.reap_others()
            if self.primary_exited():
                return
            if (
                got is not None
                and got.si_signo == signal.SIGTERM
                and got.si_pid == launcher
            ):
                self.ended = True
                return

    def drain(self, grace_sec: float) -> None:
        """Terminate and reap the remaining tree until no child is left."""
        start = time.monotonic()
        terminated: set[int] = set()
        while True:
            others = self.reap_others()
            if not self.primary_reaped and not others and self.primary_exited():
                # A snapshot can miss a member forking its successor; the zombie still
                # holds the group id, so one last group kill catches whatever it missed.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.primary, signal.SIGKILL)
                os.waitpid(self.primary, 0)
                self.primary_reaped = True
            if self.primary_reaped:
                try:
                    os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    return
            kill = time.monotonic() - start >= grace_sec
            sig = signal.SIGKILL if kill else signal.SIGTERM
            if not self.primary_reaped and (kill or self.primary not in terminated):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.primary, sig)
            pids = _descendants()
            if targets := (pids if kill else pids - terminated):
                _signal_tree(self.libc, targets, sig)
            terminated |= pids | {self.primary}
            signal.sigtimedwait({signal.SIGCHLD}, _WAIT_STEP_SEC)


def _write_receipt(fd: int, supervisor: _Supervisor) -> None:
    payload = {"status": supervisor.status, "ended": supervisor.ended}
    with os.fdopen(fd, "wb") as pipe:
        pipe.write(json.dumps(payload).encode())


def _exec_child(argv: list[str], mask: Iterable[int]) -> NoReturn:
    try:
        # The command leads a group of its own, so a signal it sends its group never
        # reaches the supervisor, and the drain can end that whole group at once.
        os.setpgid(0, 0)
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
    global _CHILDREN_FILES
    _CHILDREN_FILES = _children_files_available()
    libc = ctypes.CDLL(None, use_errno=True)
    if not _become_subreaper(libc):
        os.write(2, b"subreaper: the kernel refused PR_SET_CHILD_SUBREAPER\n")
        os._exit(UNSUPPORTED)
    if args.receipt_fd is not None:
        os.set_inheritable(args.receipt_fd, False)
    with contextlib.suppress(PermissionError):
        os.setsid()
    # Blocked before the fork, so none is lost or takes its default action between the
    # fork and the wait; the command gets the original mask back.
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, _HELD)
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
