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
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

SANDBOX_EXECUTE_INTERFACE = "sandbox.execute"
# Egress is a separate interface, never implied by the authority to run code: an
# activation reaches the network only where its own effective grant carries this.
SANDBOX_EGRESS_INTERFACE = "sandbox.egress"

# The local runtimes a deployment may name. A confining runtime joins this set when it
# exists; until then naming one is refused rather than silently run as the weaker fence.
SANDBOX_RUNTIMES = frozenset({"posix_process"})

# Bounds one command's captured streams so a runaway writer cannot unbound a result.
MAX_STREAM_CHARS = 64 * 1024


class SandboxEgressMode(StrEnum):
    """Whether a binding's commands may reach the network, and under what contract.

    ``DENY`` is the default and keeps a command an activation-private transition. The
    opt-in is named for what recovery does rather than for what it permits: a command
    that egressed before the episode's seal may run again on a re-drive, so the author
    owns idempotency and reconciliation. The fabric mints no per-command receipt to
    make it once, and the name is not a delivery guarantee.
    """

    DENY = "deny"
    AUTHOR_OWNED_AT_LEAST_ONCE = "author_owned_at_least_once"

    @property
    def allows_egress(self) -> bool:
        return self is SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE


class SandboxRuntimeProfile(BaseModel):
    """A policy-approved local runtime and the envelope its commands run under.

    The envelope is the bounded CPU, memory, disk, descriptor, and wallclock cost one
    command may take. It is ordinary worker resource isolation held for the dispatch,
    not an admission credit.
    """

    model_config = ConfigDict(frozen=True)

    runtime: str = "posix_process"
    command_timeout_sec: float = 60.0
    cpu_seconds: int = 60
    memory_bytes: int = 2 * 1024**3
    file_size_bytes: int = 512 * 1024**2
    open_files: int = 1024


class LocalSandboxCapability(BaseModel):
    """The dispatch-scoped authority the worker-local runtime validates per command.

    It carries the fences of the episode's private-state attachment, so a command runs
    only under the holder and write epoch that currently owns the workspace it mutates.
    A superseded holder fails the same fence its seal would fail. It is immutable and
    minted per dispatch, so a grant revoked or attenuated between dispatches takes
    effect at the next one, under the fences of that dispatch's own attachment.
    """

    model_config = ConfigDict(frozen=True)

    attachment_id: str
    reference_id: str
    worker_id: str
    incarnation: int
    write_epoch: int
    profile: SandboxRuntimeProfile
    # The effective mode, resolved against the activation's own attenuated grant rather
    # than copied from the pinned binding, so a child never inherits an egress its
    # parent withheld.
    network_egress: SandboxEgressMode = SandboxEgressMode.DENY

    @property
    def egress_allowed(self) -> bool:
        return self.network_egress.allows_egress


class SandboxCommand(BaseModel):
    """One command to run against an activation's workspace."""

    model_config = ConfigDict(frozen=True)

    argv: tuple[str, ...]
    timeout_sec: float | None = None  # None takes the capability's envelope deadline


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


class SandboxUnavailable(Exception):
    """The runtime cannot give a command the fence the sandbox declares.

    Like a denial this settles the action rather than escalating it: a command the
    runtime could not start under its fence never runs unfenced instead.
    """


class LocalSandboxExecutor(ABC):
    """The seam a harness adapter hands a local code action to.

    An adapter reaches the sandbox only through this: it cannot run unrestricted host
    code, and the action never becomes an egress or tool invocation.
    """

    @abstractmethod
    def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        """Run one command under the dispatch's capability and return its result."""
