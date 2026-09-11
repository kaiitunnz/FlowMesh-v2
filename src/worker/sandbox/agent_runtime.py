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

    def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        self._check_fence()
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
        """Refuse a command whose capability is not this dispatch's write authority.

        Both sides are minted for one dispatch, so this is a consistency check rather
        than the security boundary: the load-bearing owner and epoch fences are the
        holder's, applied when it opens and seals the generation.
        """
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
