"""The capability-gated seam a harness hands an agent's local code action to."""

import logging
import threading
from collections.abc import Callable

from shared.private_state import PrivateStateAttachment
from shared.sandbox import (
    LocalSandboxCapability,
    LocalSandboxExecutor,
    SandboxCommand,
    SandboxCommandResult,
    SandboxDenied,
    SandboxReapUnproved,
)
from shared.telemetry.config import TelemetryLevel
from shared.telemetry.provider import payload_free_span
from shared.telemetry.semconv import PHYSICAL_WORKER_ID, SPAN_SANDBOX_COMMAND

from ..private_state import MaterializedState
from ..telemetry import otel
from .runtime import SandboxRuntime

_LOG = logging.getLogger("agent-sandbox")
# Beyond a command's own deadline: its supervisor's reap and the result's collection.
_DRAIN_SLACK_SEC = 30.0


class AgentSandboxRuntime(LocalSandboxExecutor):
    """Run an agent's commands in its own workspace under one dispatch's capability.

    Every command is validated against the capability the dispatch minted and the
    attachment the episode opened its private state with, so a command runs only under
    the holder and write epoch that owns the workspace it mutates. The command mutates
    the agent's ``workspace_fs`` and becomes durable at the episode's ordinary boundary
    seal; it raises no invocation, claim, route, or permit.

    The runtime is also the dispatch's command fence: once the step ends it admits no
    further command and waits out the ones in flight, and a command whose tree was not
    proved reaped leaves the dispatch unable to seal.
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
        self._admission = threading.Condition()
        self._open = True
        self._running = 0
        self._unproved = False
        self._unreaped: list[Callable[[], bool]] = []

    @property
    def reap_unproved(self) -> bool:
        with self._admission:
            return self._unproved

    def close(self) -> None:
        """Close admission, so a later command is denied."""
        with self._admission:
            self._open = False

    def drain(self) -> bool:
        """Close admission, wait out the commands in flight, and return whether every
        command of the dispatch was proved reaped."""
        bound = self._capability.profile.command_timeout_sec + _DRAIN_SLACK_SEC
        with self._admission:
            self._open = False
            settled = self._admission.wait_for(lambda: self._running == 0, bound)
        self.finish_reaps()
        with self._admission:
            return settled and not self._unproved

    def finish_reaps(self) -> bool:
        """Retry each unproved reap; return whether nothing is left running."""
        with self._admission:
            unreaped, self._unreaped = self._unreaped, []
        still = [retry for retry in unreaped if not retry()]
        with self._admission:
            self._unreaped.extend(still)
            return not self._unreaped and self._running == 0

    def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        with self._admission:
            if self._unproved:
                raise SandboxReapUnproved(
                    "an earlier command of this dispatch was not proved reaped"
                )
            if not self._open:
                raise SandboxDenied("the step that owns this sandbox has ended")
            self._running += 1
        try:
            return self._execute(command)
        except SandboxReapUnproved as exc:
            with self._admission:
                self._unproved = True
                if exc.retry is not None:
                    self._unreaped.append(exc.retry)
            raise
        finally:
            with self._admission:
                self._running -= 1
                self._admission.notify_all()

    def _execute(self, command: SandboxCommand) -> SandboxCommandResult:
        self._check_fence()
        _LOG.info("[sandbox] %s", " ".join(command.argv)[:200])
        if otel.emits(TelemetryLevel.FULL):
            with payload_free_span(
                otel.get_tracer(),
                SPAN_SANDBOX_COMMAND,
                attributes=otel.new_span_attributes(
                    {PHYSICAL_WORKER_ID: self._capability.worker_id}
                ),
            ):
                result = self._run(command)
        else:
            result = self._run(command)
        _LOG.info(
            "[sandbox] exit=%s%s",
            result.exit_code,
            " (timed out)" if result.timed_out else "",
        )
        return result

    def _run(self, command: SandboxCommand) -> SandboxCommandResult:
        return self._runtime.run(
            self._state.workspace,
            command,
            self._capability.profile,
            self._capability.egress_allowed,
        )

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
