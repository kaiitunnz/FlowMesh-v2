"""The worker each dispatched task holds."""

import logging
from collections.abc import Sequence

from ...orchestration import OrchestrationEngine
from ...registries.worker import WorkerRegistry
from ..models import TaskRecord, TaskStatus
from . import episode_dispatch
from .reports import membership


class WorkerReservations:
    """Holds the worker each dispatched task reserves until its dispatch ends, and
    releases ended reservations off the lock."""

    def __init__(
        self,
        tasks: dict[str, TaskRecord],
        engines: dict[str, OrchestrationEngine],
        worker_registry: WorkerRegistry,
        logger: logging.Logger,
    ) -> None:
        self._tasks = tasks
        self._engines = engines
        self._worker_registry = worker_registry
        self._logger = logger
        # The worker and dispatch holding each dispatched task, until a commit moves the
        # task off it and releases the worker's reservation for it.
        self.held_dispatches: dict[str, tuple[str, str]] = {}
        # Reservations of dispatches that ended, released after the lock; one whose
        # release failed stays here for the next release.
        self.ended_dispatches: list[tuple[str, str]] = []

    def take_ended(self) -> list[tuple[str, str]]:
        """Take every reservation whose dispatch ended."""
        ended, self.ended_dispatches = self.ended_dispatches, []
        return ended

    def requeue_ended(self, failed: list[tuple[str, str]]) -> None:
        """Keep the reservations whose release failed for the next release."""
        self.ended_dispatches[:0] = failed

    def hold_dispatch_locked(self, task_id: str, held: tuple[str, str]) -> None:
        """Reserve a dispatch's worker for its task, ending the one it replaces."""
        earlier = self.held_dispatches.get(task_id)
        if earlier is not None and earlier != held:
            self.ended_dispatches.append(earlier)
        self.held_dispatches[task_id] = held

    def seed_held_dispatches_locked(self) -> None:
        """Reserve each restored dispatch's worker, as its dispatch did."""
        self.held_dispatches.update(
            (record.task_id, (record.assigned_worker, record.dispatch_id))
            for record in self._tasks.values()
            if membership(record) == TaskStatus.DISPATCHED
            and record.assigned_worker is not None
            and record.dispatch_id is not None
            and not episode_dispatch.dispatch_ended_at_suspension_locked(
                self._engines, record
            )
        )

    def release_ended_dispatches_locked(self, task_ids: Sequence[str]) -> None:
        """Release each worker reserved for a dispatch that ended: its task moved off
        it, or it ended at a suspension.

        A worker reporting its status names the dispatch it concerns, and its IDLE
        clears the reservation itself; one that names none is fenced while reserved, so
        the end of the dispatch is what frees it. The release is a no-op once the
        worker's own IDLE cleared it, and never frees a later reservation.
        """
        for task_id in dict.fromkeys(task_ids):
            held = self.held_dispatches.get(task_id)
            record = self._tasks.get(task_id)
            if held is None or (
                record is not None
                and membership(record) == TaskStatus.DISPATCHED
                and record.dispatch_id == held[1]
                and not episode_dispatch.dispatch_ended_at_suspension_locked(
                    self._engines, record
                )
            ):
                continue
            del self.held_dispatches[task_id]
            self.ended_dispatches.append(held)

    def release_workers(
        self, reservations: Sequence[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Release each worker from its dispatch; returns those that failed."""
        failed: list[tuple[str, str]] = []
        for worker_id, dispatch_id in reservations:
            try:
                self._worker_registry.release_worker(worker_id, dispatch_id)
            except Exception as exc:
                self._logger.warning(
                    "Failed to release worker %s from dispatch %s: %s",
                    worker_id,
                    dispatch_id,
                    exc,
                )
                failed.append((worker_id, dispatch_id))
        return failed
