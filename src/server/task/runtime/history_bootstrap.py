"""Classifying workflows stored before the open-workflow index existed.

A restart restores only the workflows the open set names. A store holding workflows
registered before that set was kept has each of them classified once, a bounded batch
at a time: a settled one nothing still needs is closed and freed behind its certified
history, and every other joins the open set. A cursor recorded after each batch lets a
crash resume where it stopped.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Collection
from typing import Protocol

from ...registries.workflow import WorkflowRegistry
from .settled_history import SettledHistory

INDEX_COMPLETE = "complete:v1"
_INDEX_BUILDING = "building:v1:"
BATCH_SIZE = 64


class ScratchRuntime(Protocol):
    """A runtime that restores stored workflows to classify them, delivering
    nothing."""

    @property
    def settled_history(self) -> SettledHistory: ...

    async def restore(self, workflow_ids: Collection[str]) -> list[str]: ...

    def reclaim_invocations(self, workflow_id: str) -> frozenset[str] | None: ...

    def workflow_task_ids(self, workflow_id: str) -> list[str]: ...

    def shutdown(self) -> None: ...


class HistoryBootstrap[R: ScratchRuntime]:
    """Builds the open-workflow index over every registered workflow, once."""

    def __init__(
        self,
        registry: WorkflowRegistry,
        scratch_runtime: Callable[[], R],
        close: Callable[[R, str], bool],
        claims_terminal: Callable[[Collection[str]], bool] | None,
        archived: Callable[[str], bool | None] | None,
        logger: logging.Logger,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        # ``close`` closes a settled workflow restored in the given runtime, or finds
        # it closed, and returns whether it is. ``claims_terminal`` runs on the event
        # loop that owns the resident claims; ``archived`` answers whether a finished
        # task's logs need no more archiving, None when that cannot be told.
        self._registry = registry
        self._scratch_runtime = scratch_runtime
        self._close = close
        self._claims_terminal = claims_terminal
        self._archived = archived
        self._logger = logger
        self._batch_size = batch_size

    async def run(self) -> None:
        """Classify every registered workflow not yet classified, or every one again
        when the registered, open and freed workflows disagree."""
        state = await self._registry.history_index_state_async()
        cursor = 0
        if state == INDEX_COMPLETE:
            census = await self._registry.history_census_async()
            if census.consistent:
                return
            self._logger.warning(
                "%d registered workflows are not %d open and %d freed; classifying "
                "them again",
                census.registered,
                census.open,
                census.freed,
            )
        elif (resumed := _cursor(state)) is not None:
            cursor = resumed
        elif state is not None:
            self._logger.warning(
                "Rebuilding the open-workflow index over unrecognized state %r", state
            )
        self._logger.info("Classifying stored workflows for the open-workflow index")
        outcomes: Counter[str] = Counter()
        while True:
            cursor, workflow_ids = await self._registry.scan_workflow_ids_async(
                cursor, self._batch_size
            )
            if workflow_ids:
                outcomes.update(await self._classify(sorted(set(workflow_ids))))
            if cursor == 0:
                break
            await self._registry.set_history_index_state_async(
                f"{_INDEX_BUILDING}{cursor}"
            )
        await self._registry.settle_open_set_async()
        await self._registry.set_history_index_state_async(INDEX_COMPLETE)
        self._logger.info(
            "Classified stored workflows: %s",
            ", ".join(f"{count} {why}" for why, count in sorted(outcomes.items()))
            or "none",
        )

    async def _classify(self, workflow_ids: list[str]) -> Counter[str]:
        outcomes: Counter[str] = Counter()
        certified = await self._registry.history_certified_async(workflow_ids)
        freed = [w for w, done in zip(workflow_ids, certified) if done]
        pending = [w for w, done in zip(workflow_ids, certified) if not done]
        await self._registry.close_workflows_async(*freed)
        outcomes["already freed"] = len(freed)
        if not pending:
            return outcomes
        scratch = self._scratch_runtime()
        try:
            scratch.settled_history.start(
                await self._registry.history_seq_async(), 0, 0
            )
            restored = set(await scratch.restore(pending))
            reopened: list[str] = []
            for workflow_id in pending:
                why = (
                    await self._free(scratch, workflow_id)
                    if workflow_id in restored
                    else "unrestored"
                )
                outcomes[why] += 1
                if why != "freed":
                    reopened.append(workflow_id)
            await self._registry.reopen_workflows_async(*reopened)
        finally:
            scratch.shutdown()
        return outcomes

    async def _free(self, scratch: R, workflow_id: str) -> str:
        """Free one restored workflow; returns ``"freed"`` or what keeps it open."""
        invocation_ids = scratch.reclaim_invocations(workflow_id)
        if invocation_ids is None:
            return "in use"
        if (
            invocation_ids
            and self._claims_terminal is not None
            and not self._claims_terminal(invocation_ids)
        ):
            return "holding resident claims"
        if self._archived is not None and not await asyncio.to_thread(
            self._all_archived, scratch.workflow_task_ids(workflow_id)
        ):
            return "archiving logs"
        if not await asyncio.to_thread(self._close, scratch, workflow_id):
            return "not closed"
        if not await asyncio.to_thread(scratch.settled_history.evaluate, workflow_id):
            return "not certified"
        return "freed"

    def _all_archived(self, task_ids: list[str]) -> bool:
        assert self._archived is not None
        return all(self._archived(task_id) for task_id in task_ids)


def _cursor(state: str | None) -> int | None:
    if state is None or not state.startswith(_INDEX_BUILDING):
        return None
    try:
        return int(state.removeprefix(_INDEX_BUILDING))
    except ValueError:
        return None


__all__ = ["BATCH_SIZE", "INDEX_COMPLETE", "HistoryBootstrap", "ScratchRuntime"]
