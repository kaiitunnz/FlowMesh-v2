"""The capability-gated seam a harness hands an agent's local code action to."""

import logging

from shared.private_state import PrivateStateAttachment
from shared.sandbox import (
    LocalSandboxCapability,
    LocalSandboxExecutor,
    SandboxCommand,
    SandboxCommandResult,
    SandboxDenied,
)

from ..private_state import MaterializedState
from .runtime import SandboxRuntime

_LOG = logging.getLogger("agent-sandbox")


class AgentSandboxRuntime(LocalSandboxExecutor):
    """Run an agent's commands in its own workspace under one dispatch's capability.

    Every command is validated against the capability the dispatch minted and the
    attachment the episode opened its private state with, so a command runs only under
    the holder and write epoch that owns the workspace it mutates. The command mutates
    the agent's ``workspace_fs`` and becomes durable at the episode's ordinary boundary
    seal; it raises no invocation, claim, route, or permit.
    """

    def __init__(
        self,
        capability: LocalSandboxCapability,
        attachment: PrivateStateAttachment,
        state: MaterializedState,
        runtime: SandboxRuntime,
    ) -> None:
        self._capability = capability
        self._attachment = attachment
        self._state = state
        self._runtime = runtime
        self._executed = 0

    @property
    def executed(self) -> int:
        """How many commands this dispatch has run, for the episode's own accounting."""
        return self._executed

    def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        self._check_fence()
        self._executed += 1
        _LOG.info("[sandbox] %s", " ".join(command.argv)[:200])
        result = self._runtime.run(
            self._state.workspace, command, self._capability.profile
        )
        _LOG.info(
            "[sandbox] exit=%s%s",
            result.exit_code,
            " (timed out)" if result.timed_out else "",
        )
        return result

    def _check_fence(self) -> None:
        """Refuse a command whose capability is not this dispatch's write authority."""
        capability, attachment = self._capability, self._attachment
        if (
            capability.attachment_id != attachment.attachment_id
            or capability.reference_id != attachment.reference_id
            or capability.worker_id != attachment.worker_id
            or capability.incarnation != attachment.incarnation
            or capability.write_epoch != attachment.write_epoch
        ):
            raise SandboxDenied(
                "the sandbox capability is not this dispatch's write authority"
            )
