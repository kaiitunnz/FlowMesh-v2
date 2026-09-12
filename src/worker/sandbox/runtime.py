"""Running one agent-local command inside its fenced workspace.

A runtime executes a command against the activation's own workspace and returns its
bounded result. The fence is kernel-enforced and unprivileged, so it holds in an
ordinary worker container: Landlock denies every path outside the workspace and the
read-only runtime, a seccomp filter denies IP sockets and io_uring, the envelope's
resource limits bound the command, and its process group is killed and reaped before
the action completes. A dispatch whose capability carries the egress opt-in relaxes the
two network layers and nothing else: the workspace confinement, the envelope, and the
reaping bound it as they bound any other command.

What the fence does not provide, because an unprivileged container cannot: no mount
namespace or private root view, no PID or IPC isolation (processes on one worker remain
visible to each other), and no cgroup accounting beyond process limits. Agent-local
execution is therefore a trusted-development posture on single-tenant workers, not
multi-tenant isolation; a confining runtime binds behind this seam and needs a
deployment that grants it more than a default container has.
"""

import contextlib
import ctypes
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

from shared.sandbox import (
    MAX_STREAM_CHARS,
    SandboxCommand,
    SandboxCommandResult,
    SandboxDenied,
    SandboxRuntimeProfile,
    SandboxUnavailable,
)

_LOG = logging.getLogger("sandbox-runtime")
# A drain that outlives its reaped process group is a lost thread, not a lost result.
_DRAIN_JOIN_SEC = 5.0
_DRAIN_CHUNK_CHARS = 8192
_REAP_WAIT_SEC = 5.0
_LAUNCHER = Path(__file__).with_name("_launcher.py")
_LANDLOCK_CREATE_RULESET = {"x86_64": 444, "aarch64": 444}


class SandboxRuntime(ABC):
    """Executes agent commands inside a fenced workspace."""

    name: str

    @abstractmethod
    def run(
        self,
        root: Path,
        command: SandboxCommand,
        profile: SandboxRuntimeProfile,
        egress: bool = False,
    ) -> SandboxCommandResult: ...


def landlock_abi() -> int:
    """The Landlock ABI this kernel supports, or 0 when it supports none.

    Probing is a pure query, so a kernel without Landlock reports zero rather than
    failing; the caller degrades to the remaining layers and logs the posture it got.
    """
    if (nr := _LANDLOCK_CREATE_RULESET.get(os.uname().machine)) is None:
        return 0
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.syscall.restype = ctypes.c_long
        abi = libc.syscall(nr, None, 0, 1)
    except OSError:
        return 0
    return abi if abi > 0 else 0


# The device nodes an interpreter needs to start and to discard output. Granting the
# whole of /dev would hand the command the worker's disks and terminals.
_DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom")


def _readonly_roots() -> list[str]:
    """The runtime paths a command may read and execute, and nothing else.

    ``/proc`` is deliberately absent: it is the one path that would expose another
    activation's command line and environment on a shared worker.
    """
    roots = {"/usr", "/usr/local", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt"}
    roots.add(Path(sys.executable).parent.as_posix())
    roots.add(sys.base_prefix)
    roots.add(sys.prefix)
    return sorted(roots)


def _sandbox_env(root: Path) -> dict[str, str]:
    """The command's whole environment: no inherited credentials, no proxy handles."""
    return {
        "PATH": os.defpath.lstrip(os.pathsep),
        "HOME": root.as_posix(),
        "TMPDIR": root.as_posix(),
        "LANG": "C.UTF-8",
    }


class PosixProcessSandbox(SandboxRuntime):
    """Runs each command as a fenced process rooted in the activation's workspace."""

    name = "posix_process"

    def __init__(self, abi: int | None = None) -> None:
        self._abi = landlock_abi() if abi is None else abi
        _LOG.info(
            "sandbox fence: %s, seccomp egress denial, envelope limits",
            (
                f"landlock ABI {self._abi} filesystem confinement"
                if self._abi
                else "NO landlock on this kernel, workspace scoping only"
            ),
        )

    def run(
        self,
        root: Path,
        command: SandboxCommand,
        profile: SandboxRuntimeProfile,
        egress: bool = False,
    ) -> SandboxCommandResult:
        if not command.argv:
            raise SandboxDenied("a sandbox command names no program")
        if (program := shutil.which(command.argv[0])) is None:
            raise SandboxDenied(f"{command.argv[0]!r} is not available")
        spec = {
            "landlock_abi": self._abi,
            "rw": [root.as_posix()],
            # The program's own directory too: an interpreter installed outside the
            # standard roots must still be readable to exec.
            "ro": [*_readonly_roots(), Path(program).resolve().parent.as_posix()],
            "devices": list(_DEVICES),
            # Only an authorized dispatch relaxes the network layers; every other fence
            # is identical either way.
            "egress": egress,
            "memory_bytes": profile.memory_bytes,
            "cpu_seconds": profile.cpu_seconds,
            "file_size_bytes": profile.file_size_bytes,
            "open_files": profile.open_files,
        }
        argv = [
            sys.executable,
            _LAUNCHER.as_posix(),
            json.dumps(spec),
            program,
            *command.argv[1:],
        ]
        deadline = command.timeout_sec or profile.command_timeout_sec
        try:
            # Its own session makes the command's descendants one killable group, so the
            # tree is reaped before the action completes rather than outliving it.
            proc = subprocess.Popen(  # nosec B603 - argv list, no shell, absolute program via shutil.which()
                argv,
                cwd=root,
                env=_sandbox_env(root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise SandboxUnavailable(f"the sandbox could not start a command: {exc}")
        streams = _Streams(proc)
        try:
            proc.wait(timeout=deadline)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # The command itself has finished, so anything still holding its pipes is a
            # process it left behind: kill the group, which also ends the drain.
            _reap(proc)
            # A killed group normally reaps at once; a child stuck in the kernel would
            # otherwise hold this lane, and the result is already decided either way.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(_REAP_WAIT_SEC)
        stdout, stderr = streams.collect()
        return SandboxCommandResult(
            exit_code=-1 if timed_out else proc.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )


class _Streams:
    """Drain a command's pipes off the waiting thread and keep a bounded prefix.

    Waiting on the process rather than on end-of-pipe is what keeps a stray background
    writer from holding the episode open to the deadline; draining concurrently is what
    keeps a chatty command from blocking on a full pipe before it can exit.
    """

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._out: list[str] = []
        self._err: list[str] = []
        self._threads = [
            threading.Thread(target=self._drain, args=(proc.stdout, self._out)),
            threading.Thread(target=self._drain, args=(proc.stderr, self._err)),
            threading.Thread(target=self._close, args=(proc.stdin,)),
        ]
        for thread in self._threads:
            thread.daemon = True
            thread.start()

    def collect(self) -> tuple[str, str]:
        for thread in self._threads:
            thread.join(_DRAIN_JOIN_SEC)
        return "".join(self._out), "".join(self._err)

    @staticmethod
    def _drain(pipe: IO[str] | None, into: list[str]) -> None:
        """Keep a bounded prefix, then keep reading without keeping.

        Reading in fixed chunks rather than by line is what bounds the worker's own
        memory: a command emitting one newline-free stream would otherwise buffer the
        whole thing here while the reader waited for a terminator that never comes.
        """
        if pipe is None:
            return
        kept = 0
        with contextlib.closing(pipe):
            while chunk := pipe.read(_DRAIN_CHUNK_CHARS):
                if kept < MAX_STREAM_CHARS:
                    into.append(chunk[: MAX_STREAM_CHARS - kept])
                    kept += len(chunk)

    @staticmethod
    def _close(pipe: IO[str] | None) -> None:
        """Close the command's stdin so one that reads it sees end-of-input at once."""
        if pipe is None:
            return
        with contextlib.suppress(OSError):
            pipe.close()


def _reap(proc: subprocess.Popen[str]) -> None:
    """Kill the command's whole process group, including anything it left running."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def build_sandbox_runtime() -> SandboxRuntime:
    """Instantiate the runtime an agent's local commands run in."""
    return PosixProcessSandbox()
