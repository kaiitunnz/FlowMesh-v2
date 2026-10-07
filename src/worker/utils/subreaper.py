"""A per-tree supervisor that runs one command and proves its whole process tree reaped.

Run as a fresh, single-threaded process: ``python -I subreaper.py [--receipt PATH]
[--grace SEC] [--budget SEC] -- ARGV...``. The supervisor marks itself a child
subreaper, so every descendant the command leaves behind is re-parented to it, whatever
session or group that descendant moved to. It forks and execs the command, waits for it
to exit (or for its own SIGTERM, which ends the command early), then terminates and
reaps whatever remains until it has no child left. Escalation goes TERM, then KILL after
the grace, within the cleanup budget.

Only that drain to no-children exits :data:`REAPED`; the command's own exit status goes
to the receipt instead. Any other exit, or death by a signal, proves nothing about the
tree. The module needs only the standard library, so the worker imports its helpers and
runs the same file as the supervisor.
"""

import argparse
import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

REAPED = 86  # the drain reached no children: the whole tree is gone
UNPROVED = 87  # the cleanup budget ran out with descendants still alive
UNSUPPORTED = 88  # the kernel would not make the supervisor a child subreaper

DEFAULT_GRACE_SEC = 2.0
DEFAULT_BUDGET_SEC = 10.0

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
    receipt: Path | None = None,
    grace_sec: float = DEFAULT_GRACE_SEC,
    budget_sec: float = DEFAULT_BUDGET_SEC,
) -> list[str]:
    """The argv that runs ``argv`` under a fresh supervisor."""
    head = [sys.executable, "-I", _SCRIPT.as_posix()]
    if receipt is not None:
        head += ["--receipt", receipt.as_posix()]
    return [*head, "--grace", str(grace_sec), "--budget", str(budget_sec), "--", *argv]


def reap_proved(returncode: int | None) -> bool:
    """Whether a supervisor's exit status proves its tree reaped."""
    return returncode == REAPED


def read_receipt(path: Path) -> int | None:
    """The supervised command's exit status from a receipt, or None without one."""
    try:
        return int(json.loads(path.read_text())["status"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def end_supervised(proc: subprocess.Popen[str], timeout_sec: float) -> bool:
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
        except OSError:
            continue
        # A command name may hold spaces or parentheses; fields follow the last ")".
        fields = stat[stat.rfind(")") + 2 :].split()
        table[int(entry)] = (int(fields[1]), int(fields[19]))
    return table


def _parents() -> dict[int, int]:
    return {pid: ppid for pid, (ppid, _) in _process_table().items()}


def _descendants() -> set[int]:
    parents = _parents()
    found: set[int] = set()
    frontier = [os.getpid()]
    while frontier:
        parent = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == parent and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return found


def _signal_descendant(libc: ctypes.CDLL, pid: int, sig: int) -> None:
    # Pinning the pid first, then confirming it is still in the tree, keeps a recycled
    # pid that left the tree from being signalled. The raw syscalls stand in for
    # ``os.pidfd_open``, which an interpreter built against an older libc omits.
    fd = libc.syscall(_SYS_PIDFD_OPEN, pid, 0)
    if fd < 0:
        return
    try:
        if pid in _descendants():
            libc.syscall(_SYS_PIDFD_SEND_SIGNAL, fd, sig, None, 0)
    finally:
        os.close(fd)


class _Supervisor:
    def __init__(self, libc: ctypes.CDLL, primary: int) -> None:
        self.libc = libc
        self.primary = primary
        self.status: int | None = None

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
                return

    def drain(self, grace_sec: float, budget_sec: float) -> bool:
        """Terminate and reap the remaining tree; return whether it reached no
        children within the budget."""
        start = time.monotonic()
        terminated: set[int] = set()
        while self.reap_ready():
            elapsed = time.monotonic() - start
            if elapsed > budget_sec:
                return False
            kill = elapsed >= grace_sec
            for pid in _descendants():
                if kill or pid not in terminated:
                    _signal_descendant(
                        self.libc, pid, signal.SIGKILL if kill else signal.SIGTERM
                    )
                    terminated.add(pid)
            signal.sigtimedwait({signal.SIGCHLD}, _WAIT_STEP_SEC)
        return True


def _write_receipt(path: Path, status: int | None) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"status": status}, handle)
    staged.replace(path)


def _exec_child(argv: list[str], mask: Iterable[int]) -> None:
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        os.execv(argv[0], argv)  # nosec B606 - argv list from the worker, no shell
    except OSError as exc:
        os.write(2, f"subreaper: cannot exec {argv[0]}: {exc}\n".encode())
    os._exit(127)


def main() -> None:
    parser = argparse.ArgumentParser(prog="subreaper")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--grace", type=float, default=DEFAULT_GRACE_SEC)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SEC)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        parser.error("a command to supervise is required")
    libc = ctypes.CDLL(None, use_errno=True)
    if not _become_subreaper(libc):
        os.write(2, b"subreaper: the kernel refused PR_SET_CHILD_SUBREAPER\n")
        os._exit(UNSUPPORTED)
    if os.getsid(0) != os.getpid():
        os.setsid()
    # Blocked before the fork, so neither signal is lost or takes its default action
    # between the fork and the wait; the command gets the original mask back.
    mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, {signal.SIGCHLD, signal.SIGTERM, signal.SIGINT}
    )
    primary = os.fork()
    if primary == 0:
        _exec_child(argv, mask)
    # The command alone owns the stdio pipes, so its reader sees end-of-file when the
    # command closes them rather than when the supervisor finally exits.
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.close(devnull)
    supervisor = _Supervisor(libc, primary)
    supervisor.await_primary()
    if not supervisor.drain(args.grace, args.budget):
        os._exit(UNPROVED)
    if args.receipt is not None:
        _write_receipt(args.receipt, supervisor.status)
    os._exit(REAPED)


if __name__ == "__main__":
    main()
