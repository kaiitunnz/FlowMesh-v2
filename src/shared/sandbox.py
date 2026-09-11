"""The agent-local sandbox contract shared by the compiler, the fabric, and the worker.

An agent that may execute code declares the ``sandbox.execute`` interface in its
authority ceiling and carries a pinned sandbox binding. The fabric mints one
:class:`LocalSandboxCapability` per agent-episode dispatch, fenced to the same holder
and write epoch as the episode's private-state attachment, and the worker-local runtime
validates every command against it. A command is a fenced private-state transition, not
a mediated invocation: it raises no invocation, claim, route, or permit, and its result
becomes durable at the agent's ordinary boundary seal.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

SANDBOX_EXECUTE_INTERFACE = "sandbox.execute"

# Bounds one command's captured streams so a runaway writer cannot unbound a result.
MAX_STREAM_CHARS = 64 * 1024


class SandboxRuntimeProfile(BaseModel):
    """A policy-approved local runtime and the envelope its commands run under.

    The envelope is the bounded CPU, memory, PID, disk, and wallclock cost one command
    may take. It is ordinary worker resource isolation held for the dispatch, not an
    admission credit.
    """

    model_config = ConfigDict(frozen=True)

    runtime: str = "posix_process"
    command_timeout_sec: float = 60.0
    cpu_seconds: int = 60
    memory_bytes: int = 2 * 1024**3
    max_processes: int = 64
    file_size_bytes: int = 512 * 1024**2
    open_files: int = 1024


class LocalSandboxCapability(BaseModel):
    """The dispatch-scoped authority the worker-local runtime validates per command.

    It carries the fences of the episode's private-state attachment, so a command runs
    only under the holder and write epoch that currently owns the workspace it mutates.
    A superseded holder fails the same fence its seal would fail.
    """

    model_config = ConfigDict(frozen=True)

    attachment_id: str
    reference_id: str
    worker_id: str
    incarnation: int
    write_epoch: int
    profile: SandboxRuntimeProfile


class SandboxCommand(BaseModel):
    """One command to run against an activation's workspace."""

    model_config = ConfigDict(frozen=True)

    argv: tuple[str, ...]
    timeout_sec: float | None = None  # None takes the capability's envelope deadline
    stdin: str | None = None


class SandboxCommandResult(BaseModel):
    """A command's bounded terminal result."""

    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxDenied(Exception):
    """A command the capability, the envelope, or the runtime refuses to run.

    A denial is a declared terminal outcome of the action, never a retryable failure and
    never an escalation into a mediated boundary.
    """


class LocalSandboxExecutor(ABC):
    """The seam a harness adapter hands a local code action to.

    An adapter reaches the sandbox only through this: it cannot run unrestricted host
    code, and the action never becomes an egress or tool invocation.
    """

    @abstractmethod
    def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        """Run one command under the dispatch's capability and return its result."""
