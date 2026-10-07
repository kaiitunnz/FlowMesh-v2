"""What a terminated workflow or task still owes its workers."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from shared.schemas.command import InterruptMessage, RevokeMessage

from ...registries.worker import WorkerRegistry
from ..models import TaskRecord, TaskStatus
from .boundary_router import MediatedBoundaryRouter


@dataclass(frozen=True)
class _Revoke:
    message: RevokeMessage
    # The node the dispatch went to, when known apart from the worker's record, which
    # the worker's unregister deletes.
    node_id: str | None = None


@dataclass
class Termination:
    """What control still owes workers off its lock: interrupts and revokes to send,
    operations to reap, and resident credits to release."""

    interrupts: list[InterruptMessage]
    # Each pending mediated operation's worker, agent task, and call.
    reaps: list[tuple[str, str, str]]
    resident_invocation_ids: list[str] = field(default_factory=list)
    revokes: list[_Revoke] = field(default_factory=list)


class TerminationRelease:
    """Holds each termination until the ledger save it waits on succeeds, queues
    interrupts and revokes, and sends them to workers off the lock."""

    def __init__(
        self,
        router: MediatedBoundaryRouter,
        tasks: dict[str, TaskRecord],
        worker_registry: WorkerRegistry,
        logger: logging.Logger,
    ) -> None:
        self._router = router
        self._tasks = tasks
        self._worker_registry = worker_registry
        self._logger = logger
        self.pending_terminations: list[Termination] = []
        # Terminations whose workflow's ledger has not been saved since: each releases
        # only after that save succeeds.
        self.undurable_terminations: dict[str, list[Termination]] = {}

    def ledger_saved(self, workflow_id: str) -> None:
        """Queue each termination that waited on the workflow's ledger save."""
        self.pending_terminations += self.undurable_terminations.pop(workflow_id, [])

    def has_undurable(self, workflow_id: str) -> bool:
        return workflow_id in self.undurable_terminations

    def take_terminations(self) -> list[Termination]:
        """Take every termination queued for release."""
        pending, self.pending_terminations = self.pending_terminations, []
        return pending

    def hold_termination_locked(
        self, workflow_id: str, termination: Termination
    ) -> None:
        """Hold what a termination releases until the workflow's next ledger save
        succeeds; ``_release_pending_terminations`` releases it after the lock."""
        self.undurable_terminations.setdefault(workflow_id, []).append(termination)

    @staticmethod
    def interrupt_for(record: TaskRecord, reason: str) -> InterruptMessage | None:
        # A merged child's batch keeps running for siblings from other workflows.
        if not record.assigned_worker or record.merged_parent_id:
            return None
        return InterruptMessage(
            task_id=record.task_id,
            worker_id=record.assigned_worker,
            reason=reason,
            dispatch_id=record.dispatch_id,
        )

    def revoke_locked(
        self,
        task_id: str,
        worker_id: str,
        dispatch_id: str | None,
        node_id: str | None = None,
    ) -> None:
        """Queue the revocation of a dispatch that resolved without its worker ending
        it."""
        if dispatch_id is None:
            return
        message = RevokeMessage(
            task_id=task_id, worker_id=worker_id, dispatch_id=dispatch_id
        )
        self.pending_terminations.append(
            Termination([], [], revokes=[_Revoke(message, node_id)])
        )

    def queue_interrupts_locked(self, interrupts: list[InterruptMessage]) -> None:
        if interrupts:
            self.pending_terminations.append(Termination(interrupts, []))

    def cancelling_interrupts_locked(
        self, include: Callable[[TaskRecord], bool]
    ) -> list[InterruptMessage]:
        """An interrupt for each task being cancelled that ``include`` selects."""
        return [
            interrupt
            for record in self._tasks.values()
            if record.status == TaskStatus.CANCELLING
            and include(record)
            and (interrupt := self.interrupt_for(record, record.error or "cancelled"))
        ]

    def interrupt_cancelling_locked(self, workflow_id: str) -> None:
        """Queue an interrupt for each task a restart found still being cancelled,
        whose worker may never have received one."""
        self.queue_interrupts_locked(
            self.cancelling_interrupts_locked(
                lambda record: record.workflow_id == workflow_id
            )
        )

    def release_terminated_work(self, termination: Termination) -> None:
        """Send what ``termination`` owes workers, best effort: each resident credit,
        interrupt, revoke, and reap is attempted however the others fare."""
        # The fenced terminal releases each in-flight resident invocation's credit, so a
        # lost or draining replica is not held forever.
        for invocation_id in termination.resident_invocation_ids:
            try:
                self._router.release_resident_credit(invocation_id, failed=True)
            except Exception:
                self._logger.exception(
                    "Releasing the resident credit of %s failed", invocation_id
                )
        for interrupt in termination.interrupts:
            try:
                worker = self._worker_registry.get_worker(interrupt.worker_id)
                if worker is None:
                    self._logger.warning(
                        "Cannot publish interrupt for %s; worker %s missing",
                        interrupt.task_id,
                        interrupt.worker_id,
                    )
                else:
                    self._worker_registry.publish_interrupt(worker, interrupt)
            except Exception:
                self._logger.exception(
                    "Interrupting %s on %s failed",
                    interrupt.task_id,
                    interrupt.worker_id,
                )
        for revoke in termination.revokes:
            message = revoke.message
            try:
                node_id = revoke.node_id
                if node_id is None and (
                    worker := self._worker_registry.get_worker(message.worker_id)
                ):
                    node_id = worker.node_id
                if node_id is None:
                    self._logger.warning(
                        "Cannot revoke dispatch %s of %s; worker %s missing",
                        message.dispatch_id,
                        message.task_id,
                        message.worker_id,
                    )
                else:
                    self._worker_registry.publish_revoke(node_id, message)
            except Exception:
                self._logger.exception(
                    "Revoking dispatch %s on %s failed",
                    message.dispatch_id,
                    message.worker_id,
                )
        # The worker drops a reaped operation and its custody.
        for worker_id, agent_task_id, call in termination.reaps:
            try:
                self._router.reap_mediated_op(worker_id, agent_task_id, call)
            except Exception:
                self._logger.exception(
                    "Reaping the operation of %s on %s failed", agent_task_id, worker_id
                )
