"""The run-to-yield executor for one private sandbox session.

A session runs its declared commands one per dispatch against its own filesystem, which
it materializes from the generation its binding names and seals again at the end of each
step. The sealed generation is also the session's progress: a step runs the command its
generation indexes, so a resume never repeats or skips one. A generation the holder
cannot supply in full fails the step closed rather than starting from an empty tree.
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from shared.harness import (
    BoundaryEventKind,
    BoundaryRequest,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
)
from shared.private_state import PrivateStateUnavailable
from shared.schemas.result import SandboxCommandItem, SandboxResult
from shared.tasks.specs import SandboxSpecStrict
from shared.tasks.task_type import TaskType

from ..private_state import PrivateStateHolder
from ..sandbox import SandboxCommand, SandboxUnavailable, build_sandbox_runtime
from .base_executor import ExecutionError, Executor, ExecutorTask
from .episode_support import EpisodeStepResult

_LOG = logging.getLogger("sandbox-session-executor")

_DEFAULT_COMMAND_TIMEOUT_SEC = 60.0

# The session's own continuation binding: its capsule carries the results of the
# commands already run and is readable only by this substrate.
_CAPSULE_BACKEND = HarnessBackendKey(backend="sandbox_session", version="1")


class SandboxSessionExecutor(Executor):
    """Run one command of a sandbox session against its private filesystem."""

    name = "sandbox_session"
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset({TaskType.SANDBOX})

    def run(self, task: ExecutorTask, out_dir: Path) -> EpisodeStepResult:
        spec = self.require_spec(task, SandboxSpecStrict)
        dispatch = task.sandbox_session
        if dispatch is None:
            raise ExecutionError(
                f"{task.task_id} routed to the sandbox-session executor without a "
                "session dispatch context"
            )
        binding, attachment = dispatch.private_state, dispatch.private_state_attachment
        if binding is None or attachment is None:
            raise ExecutionError(
                f"{task.task_id} carries no private-state authority for its session"
            )
        index = dispatch.command_index
        if not 0 <= index < len(spec.commands):
            raise ExecutionError(
                f"{task.task_id} resumed at command {index}, outside its declared "
                f"{len(spec.commands)}"
            )

        holder = PrivateStateHolder(self._config.private_state_dir)
        try:
            state = holder.open(binding, attachment)
        except PrivateStateUnavailable as exc:
            raise ExecutionError(f"PrivateStateUnavailable: {exc}") from exc

        declared = spec.commands[index]
        runtime = build_sandbox_runtime(dispatch.runtime)
        try:
            outcome = runtime.run(
                state.sandbox,
                SandboxCommand(
                    argv=tuple(declared.argv),
                    stdin=declared.stdin,
                    timeout_sec=declared.timeoutSeconds or _DEFAULT_COMMAND_TIMEOUT_SEC,
                ),
            )
        except SandboxUnavailable as exc:
            raise ExecutionError(
                f"sandbox command {index} could not run: {exc}"
            ) from exc

        item = SandboxCommandItem(
            argv=list(declared.argv),
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            timed_out=outcome.timed_out,
        )
        _LOG.info(
            "sandbox command %d of %s exited %d",
            index,
            task.task_id,
            outcome.exit_code,
        )
        history = [*_history(dispatch.capsule_blob), item]
        sealed = holder.seal(state, attachment)

        if index + 1 < len(spec.commands):
            return EpisodeStepResult(
                harness_result=HarnessResult(
                    kind=HarnessResultKind.YIELD,
                    request=BoundaryRequest(kind=BoundaryEventKind.YIELD),
                    capsule=HarnessCapsule(
                        backend=_CAPSULE_BACKEND, blob=_encode(history)
                    ),
                ),
                private_state=sealed,
            )
        value = SandboxResult(commands=history).model_dump_json()
        return EpisodeStepResult(
            harness_result=HarnessResult(
                kind=HarnessResultKind.COMPLETION, value=value
            ),
            value=value,
            private_state=sealed,
        )


def _history(blob: str | None) -> list[SandboxCommandItem]:
    """The results of the commands already run, from the session's continuation."""
    if not blob:
        return []
    try:
        recorded = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ExecutionError(
            f"sandbox session continuation is unreadable: {exc}"
        ) from exc
    if not isinstance(recorded, list):
        raise ExecutionError("sandbox session continuation is not a command list")
    return [SandboxCommandItem.model_validate(entry) for entry in recorded]


def _encode(history: list[SandboxCommandItem]) -> str:
    items: list[dict[str, Any]] = [item.model_dump(mode="json") for item in history]
    return json.dumps(items)
