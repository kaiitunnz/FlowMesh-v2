"""The run-to-yield agent-episode executor.

A v2 agent whose dispatch carries a harness backend key runs here rather than through
the legacy UTU path. One ``run`` is one adapter step: it resumes the backend from the
durable capsule and delivered outcomes the fabric shipped, takes a single run-to-yield
step, and returns the step's :class:`HarnessResult`. The lane releases after the step;
the server routes any boundary and re-dispatches with the next capsule and outcomes.
"""

from pathlib import Path
from typing import ClassVar

from shared.harness import (
    HarnessAdapter,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
)
from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from worker.harness import build_adapter

from .base_executor import ExecutionError, Executor, ExecutorTask


class AgentEpisodeResult(BaseExecutorResult):
    """One agent-episode step's result: the harness step plus its terminal value.

    ``harness_result`` carries the step back to the server through the success metadata;
    ``value`` is the agent's declared output on a completion step, readable over REST.
    """

    harness_result: HarnessResult
    value: str | None = None


class AgentEpisodeExecutor(Executor):
    """Drive one run-to-yield step of an agent's harness backend."""

    name = "agent_episode"
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset({TaskType.AGENT})

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._adapter: HarnessAdapter | None = None

    def run(self, task: ExecutorTask, out_dir: Path) -> AgentEpisodeResult:
        dispatch = task.agent_episode
        if dispatch is None:
            raise ExecutionError(
                f"{task.task_id} routed to the agent-episode executor without an "
                "agent-episode dispatch context"
            )
        adapter = build_adapter(dispatch.backend, task, self._config)
        if not adapter.bypass_disabled():
            raise ExecutionError(
                f"harness backend {dispatch.backend.backend!r} does not disable native "
                "tool and subagent bypass paths"
            )
        self._adapter = adapter
        capsule = (
            HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
            if dispatch.capsule_blob is not None
            else None
        )
        result = adapter.start(
            task.task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes
        )
        value = result.value if result.kind is HarnessResultKind.COMPLETION else None
        return AgentEpisodeResult(harness_result=result, value=value)

    def cancel(self, task_id: str) -> None:
        if self._adapter is not None:
            self._adapter.cancel(task_id)

    def cleanup_after_run(self) -> None:
        self._adapter = None
