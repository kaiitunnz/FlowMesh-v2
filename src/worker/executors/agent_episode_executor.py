"""The run-to-yield agent-episode executor.

Every agent dispatches here through its resolved harness backend key. One ``run`` is
one adapter step: it resumes the backend from the
durable capsule and delivered outcomes the fabric shipped, takes a single run-to-yield
step, and returns the step's :class:`HarnessResult`. The lane releases after the step;
the server routes any boundary and re-dispatches with the next capsule and outcomes.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

from shared.harness import (
    REQUIRED_MEDIATED_FACADES,
    DeliveredOutcome,
    HarnessAdapter,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
)
from shared.outcome import ContentStoreError, FabricContentStore
from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType

from ..content_store import build_content_store
from .base_executor import ExecutionError, Executor, ExecutorTask
from .harness import build_adapter

_LOG = logging.getLogger("agent-episode-executor")


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._adapter: HarnessAdapter | None = None

    def run(self, task: ExecutorTask, out_dir: Path) -> AgentEpisodeResult:
        dispatch = task.agent_episode
        if dispatch is None:
            raise ExecutionError(
                f"{task.task_id} routed to the agent-episode executor without an "
                "agent-episode dispatch context"
            )
        adapter = build_adapter(dispatch.backend, task, self._config)
        missing = REQUIRED_MEDIATED_FACADES - adapter.mediated_facades()
        if missing:
            raise ExecutionError(
                f"harness backend {dispatch.backend.backend!r} does not mediate "
                + ", ".join(sorted(missing))
            )
        self._adapter = adapter
        capsule = (
            HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
            if dispatch.capsule_blob is not None
            else None
        )
        outcomes = self._hydrate_outcomes(dispatch.delivered_outcomes)
        for outcome in outcomes:
            _LOG.info(
                "[fabric] injecting %s outcome at call %s",
                outcome.kind.value,
                outcome.call_correlation,
            )
        result = adapter.start(task.task_id, capsule=capsule, outcomes=outcomes)
        if result.kind is HarnessResultKind.BOUNDARY and result.request is not None:
            _LOG.info(
                "[fabric] episode yielded a %s boundary (interface=%s)",
                result.request.kind.value,
                result.request.interface or "-",
            )
        value = result.value if result.kind is HarnessResultKind.COMPLETION else None
        return AgentEpisodeResult(harness_result=result, value=value)

    def _hydrate_outcomes(
        self, outcomes: tuple[DeliveredOutcome, ...]
    ) -> tuple[DeliveredOutcome, ...]:
        """Resolve any reference-backed outcome into its injected value.

        A manifest is fetched from the content store and digest-verified before
        injection; a hydration failure fails the step for a physical retry of the same
        reference, never a re-run of the invocation, so an unverified value is never
        injected. An inline outcome passes through unchanged.
        """
        if not any(o.outcome_ref is not None for o in outcomes):
            return outcomes
        store = build_content_store(self._config.server_base_url)
        if store is None:
            raise ExecutionError("cannot hydrate a reference-backed outcome: no store")
        return tuple(self._hydrate(o, store) for o in outcomes)

    @staticmethod
    def _hydrate(
        outcome: DeliveredOutcome, store: FabricContentStore
    ) -> DeliveredOutcome:
        if outcome.outcome_ref is None:
            return outcome
        try:
            value = store.hydrate(outcome.outcome_ref).decode()
        except (ContentStoreError, UnicodeDecodeError) as exc:
            raise ExecutionError(
                f"outcome hydration failed at {outcome.call_correlation}: {exc}"
            ) from exc
        return outcome.model_copy(update={"value": value, "outcome_ref": None})

    def cancel(self, task_id: str) -> None:
        if self._adapter is not None:
            self._adapter.cancel(task_id)

    def cleanup_after_run(self) -> None:
        self._adapter = None
