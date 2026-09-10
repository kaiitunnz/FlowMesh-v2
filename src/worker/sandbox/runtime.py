"""Running one session command inside its private sandbox tree.

A runtime executes a command against the session's own filesystem root and returns its
bounded result. The declared-safe path is filesystem-only: the command reaches nothing
outside its root and cannot open a network connection, so its mutation is private-state
backing rather than an external effect.
"""

import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Bounds one command's captured streams so a runaway writer cannot unbound a result.
MAX_STREAM_CHARS = 64 * 1024


class SandboxUnavailable(Exception):
    """The runtime cannot give a command the isolation the sandbox declares."""


@dataclass(frozen=True)
class SandboxCommand:
    """One command to run against a session's sandbox root."""

    argv: tuple[str, ...]
    timeout_sec: float = 60.0
    stdin: str | None = None


@dataclass(frozen=True)
class SandboxCommandResult:
    """A command's bounded terminal result."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxRuntime(ABC):
    """Executes session commands inside a private sandbox root."""

    name: str

    @abstractmethod
    def run(self, root: Path, command: SandboxCommand) -> SandboxCommandResult: ...


# Installs a seccomp filter denying IP socket creation, then becomes the command, so it
# cannot open a network connection rather than being trusted not to. A filter needs no
# privilege, which a namespace does: a worker container is normally denied one. Unix
# sockets and the filesystem are untouched. An architecture whose syscall numbers this
# does not know refuses the command instead of running it unfiltered, and the filter is
# installed in this freshly started interpreter rather than between fork and exec, where
# a threaded worker could deadlock.
_ISOLATING_LAUNCHER = """
import ctypes, os, struct, sys

_ARCH = {"x86_64": (0xC000003E, 41, 317), "aarch64": (0xC00000B7, 198, 277)}
audit, nr_socket, nr_seccomp = _ARCH[os.uname().machine]
_LD, _JEQ, _RET, _DENY, _ALLOW = 0x20, 0x15, 0x06, 0x00050000 | 13, 0x7FFF0000


def _stmt(code, k):
    return struct.pack("HBBI", code, 0, 0, k)


def _jeq(k, jt, jf):
    return struct.pack("HBBI", _JEQ, jt, jf, k)


prog = b"".join(
    [
        _stmt(_LD, 4),                 # seccomp_data.arch
        _jeq(audit, 0, 5),             # a foreign arch denies everything
        _stmt(_LD, 0),                 # seccomp_data.nr
        _jeq(nr_socket, 0, 4),         # anything but socket() runs
        _stmt(_LD, 16),                # socket() domain
        _jeq(2, 1, 0),                 # AF_INET
        _jeq(10, 0, 1),                # AF_INET6
        _stmt(_RET, _DENY),
        _stmt(_RET, _ALLOW),
    ]
)


class _Prog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


buf = ctypes.create_string_buffer(prog, len(prog))
libc = ctypes.CDLL("libc.so.6", use_errno=True)
if libc.prctl(38, 1, 0, 0, 0) != 0:      # PR_SET_NO_NEW_PRIVS
    raise OSError(ctypes.get_errno(), "cannot drop privilege escalation")
libc.syscall.argtypes = [ctypes.c_long, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
fprog = _Prog(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p))
if libc.syscall(nr_seccomp, 1, 0, ctypes.byref(fprog)) != 0:
    raise OSError(ctypes.get_errno(), "cannot install the sandbox filter")
os.execv(sys.argv[1], sys.argv[1:])
"""


def _sandbox_env(root: Path) -> dict[str, str]:
    """The command's whole environment: no inherited credentials, no proxy handles."""
    return {
        "PATH": os.defpath.lstrip(os.pathsep),
        "HOME": root.as_posix(),
        "TMPDIR": root.as_posix(),
        "LANG": "C.UTF-8",
    }


def _clip(stream: str | None) -> str:
    text = stream or ""
    return text if len(text) <= MAX_STREAM_CHARS else text[:MAX_STREAM_CHARS]


class PosixProcessSandbox(SandboxRuntime):
    """Runs each command as a filtered process rooted in the session tree.

    The command's working directory, home, and temporary directory are the session's
    own root, and its environment carries nothing the worker holds.
    """

    name = "posix_process"

    def run(self, root: Path, command: SandboxCommand) -> SandboxCommandResult:
        if not command.argv:
            raise SandboxUnavailable("a sandbox command names no program")
        if (program := shutil.which(command.argv[0])) is None:
            raise SandboxUnavailable(f"{command.argv[0]!r} is not available")
        argv = [
            sys.executable,
            "-c",
            _ISOLATING_LAUNCHER,
            program,
            *command.argv[1:],
        ]
        try:
            completed = subprocess.run(  # nosec B603 - argv list, no shell, absolute program via shutil.which()
                argv,
                cwd=root,
                env=_sandbox_env(root),
                input=command.stdin,
                capture_output=True,
                text=True,
                timeout=command.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            return SandboxCommandResult(
                exit_code=-1,
                stdout=_clip(_as_text(expired.stdout)),
                stderr=_clip(_as_text(expired.stderr)),
                timed_out=True,
            )
        except OSError as exc:
            raise SandboxUnavailable(
                f"the sandbox could not isolate a command: {exc}"
            ) from exc
        return SandboxCommandResult(
            exit_code=completed.returncode,
            stdout=_clip(completed.stdout),
            stderr=_clip(completed.stderr),
        )


def _as_text(stream: str | bytes | None) -> str:
    """A timed-out command's partial stream, which arrives as bytes on some paths."""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return stream or ""


_RUNTIMES: dict[str, type[SandboxRuntime]] = {
    PosixProcessSandbox.name: PosixProcessSandbox
}

DEFAULT_SANDBOX_RUNTIME = PosixProcessSandbox.name


def build_sandbox_runtime(name: str | None) -> SandboxRuntime:
    """Instantiate a named sandbox runtime, falling back to the default."""
    return _RUNTIMES.get(name or DEFAULT_SANDBOX_RUNTIME, PosixProcessSandbox)()
