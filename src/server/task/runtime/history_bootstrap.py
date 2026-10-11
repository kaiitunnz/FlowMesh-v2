"""Classifying the stored workflows of a store without the open-workflow index.

A restart restores only the workflows the open set names. A store without that set, or
one whose registered workflows are not each open or freed, has every stored workflow
classified, a bounded batch at a time: a settled one nothing still needs is closed and
freed behind its certified history, and every other joins the open set. A cursor
recorded after each batch lets a crash resume where it stopped.
"""

import asyncio
import gc
import logging
from collections import Counter
from collections.abc import Callable, Collection
from enum import StrEnum
from typing import Protocol

from ...registries.history import WorkflowHistoryStore
from .settled_history import SettledHistory

INDEX_COMPLETE = "complete:v1"
_INDEX_BUILDING = "building:v1:"
BATCH_SIZE = 64
# The most stored tasks one scratch runtime restores at a time; a workflow holding more
# is restored alone.
BATCH_TASKS = 2_000


class Classification(StrEnum):
    """What classifying a stored workflow found."""

    ALREADY_FREED = "already freed"
    FREED = "freed"
    IN_USE = "in use"
    HOLDING_CLAIMS = "holding resident claims"
    ARCHIVING = "archiving logs"
    NOT_CLOSED = "not closed"
    NOT_CERTIFIED = "not certified"
    UNRESTORED = "unrestored"


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
    """Builds the open-workflow index over every registered workflow.

    ``close`` closes a settled workflow restored in the given runtime, or finds it
    closed, and returns whether it is. ``claims_terminal`` runs on the event loop that
    owns the resident claims. ``archived`` answers whether a finished task's logs need
    no more archiving, None when that cannot be told.
    """

    def __init__(
        self,
        registry: WorkflowHistoryStore,
        scratch_runtime: Callable[[], R],
        close: Callable[[R, str], bool],
        claims_terminal: Callable[[Collection[str]], bool] | None,
        archived: Callable[[str], bool | None] | None,
        logger: logging.Logger,
        batch_tasks: int = BATCH_TASKS,
    ) -> None:
        self._registry = registry
        self._scratch_runtime = scratch_runtime
        self._close = close
        self._claims_terminal = claims_terminal
        self._archived = archived
        self._logger = logger
        self._batch_tasks = batch_tasks

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
        outcomes: Counter[Classification] = Counter()
        while True:
            cursor, workflow_ids = await self._registry.scan_workflow_ids_async(
                cursor, BATCH_SIZE
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

    async def _classify(self, workflow_ids: list[str]) -> Counter[Classification]:
        outcomes: Counter[Classification] = Counter()
        certified = await self._registry.history_certified_async(workflow_ids)
        freed = [w for w, done in zip(workflow_ids, certified) if done]
        pending = [w for w, done in zip(workflow_ids, certified) if not done]
        await self._registry.close_workflows_async(*freed)
        outcomes[Classification.ALREADY_FREED] = len(freed)
        counts = await self._registry.stored_task_counts_async(pending)
        for batch in _batches(pending, counts, self._batch_tasks):
            outcomes.update(await self._classify_restored(batch))
            # A scratch runtime's state is cyclic, so it is collected before the next
            # batch is restored beside it.
            gc.collect()
        return outcomes

    async def _classify_restored(self, pending: list[str]) -> Counter[Classification]:
        """Restore workflows into one scratch runtime and free or reopen each."""
        outcomes: Counter[Classification] = Counter()
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
                    else Classification.UNRESTORED
                )
                outcomes[why] += 1
                if why is not Classification.FREED:
                    reopened.append(workflow_id)
            await self._registry.reopen_workflows_async(*reopened)
        finally:
            scratch.shutdown()
        return outcomes

    async def _free(self, scratch: R, workflow_id: str) -> Classification:
        """Free one restored workflow; returns whether it is freed or what keeps it
        open."""
        invocation_ids = scratch.reclaim_invocations(workflow_id)
        if invocation_ids is None:
            return Classification.IN_USE
        if (
            invocation_ids
            and self._claims_terminal is not None
            and not self._claims_terminal(invocation_ids)
        ):
            return Classification.HOLDING_CLAIMS
        if (archived := self._archived) is not None and not await asyncio.to_thread(
            _all_archived, archived, scratch.workflow_task_ids(workflow_id)
        ):
            return Classification.ARCHIVING
        if not await asyncio.to_thread(self._close, scratch, workflow_id):
            return Classification.NOT_CLOSED
        if not await asyncio.to_thread(scratch.settled_history.evaluate, workflow_id):
            return Classification.NOT_CERTIFIED
        return Classification.FREED


def _all_archived(archived: Callable[[str], bool | None], task_ids: list[str]) -> bool:
    return all(archived(task_id) for task_id in task_ids)


def _batches(
    workflow_ids: list[str], task_counts: list[int], limit: int
) -> list[list[str]]:
    """Split workflows into batches holding at most ``limit`` tasks between them, a
    workflow holding more in a batch of its own."""
    batches: list[list[str]] = []
    batch: list[str] = []
    tasks = 0
    for workflow_id, count in zip(workflow_ids, task_counts, strict=True):
        if batch and tasks + count > limit:
            batches.append(batch)
            batch, tasks = [], 0
        batch.append(workflow_id)
        tasks += count
    if batch:
        batches.append(batch)
    return batches


def _cursor(state: str | None) -> int | None:
    if state is None or not state.startswith(_INDEX_BUILDING):
        return None
    try:
        return int(state.removeprefix(_INDEX_BUILDING))
    except ValueError:
        return None


__all__ = [
    "BATCH_SIZE",
    "BATCH_TASKS",
    "INDEX_COMPLETE",
    "Classification",
    "HistoryBootstrap",
    "ScratchRuntime",
]
