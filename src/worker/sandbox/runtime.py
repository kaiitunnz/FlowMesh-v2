"""Running one session command inside its private sandbox tree.

A runtime executes a command against the session's own filesystem root and returns its
bounded result. The declared-safe path is filesystem-only: the command reaches nothing
outside its root and has no network egress, so its mutation is private-state backing
rather than an external effect.
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


# Detaches into its own user and network namespaces, then becomes the command. An
# unprivileged user namespace carries a network namespace holding only a down loopback,
# so the command cannot reach a network at all rather than being trusted not to. The
# unshare runs in this freshly started interpreter rather than between fork and exec,
# where a threaded worker could deadlock.
_ISOLATING_LAUNCHER = (
    "import os, sys; "
    "os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET); "
    "os.execv(sys.argv[1], sys.argv[1:])"
)


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
    """Runs each command as a namespace-isolated process rooted in the session tree.

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
