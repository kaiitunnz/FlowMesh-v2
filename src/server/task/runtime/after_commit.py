"""What a runtime transition owes workers and other consumers once it is durable."""

from collections.abc import Callable
from dataclasses import dataclass, field

from shared.schemas.command import InterruptMessage, RevokeMessage
from shared.tools.contract import AgentModelTurnProposal

from ..models import TaskRecord, TaskStatus


@dataclass(frozen=True)
class CreditRelease:
    """Release the resident credit of a boundary invocation whose terminal is
    committed, with that terminal's outcome."""

    invocation_id: str
    failed: bool


@dataclass(frozen=True)
class Reap:
    """Tell a worker to drop the request it holds for a boundary that will not run
    again: a mediated operation's custody, or a captured resident request."""

    worker_id: str
    task_id: str
    call_correlation: str
    resident: bool = False


@dataclass(frozen=True)
class Interrupt:
    message: InterruptMessage


@dataclass(frozen=True)
class Revoke:
    message: RevokeMessage
    # The node the dispatch went to, when known apart from the worker's record, which
    # the worker's unregister deletes.
    node_id: str | None = None


@dataclass(frozen=True)
class Issue:
    """Hand a recorded mediated boundary to its handler, under the invocation it was
    recorded with."""

    task_id: str
    call_correlation: str
    invocation_id: str | None


@dataclass(frozen=True)
class Purge:
    """Purge a workflow's vaulted credentials, only once it has settled when
    ``settled_only``."""

    workflow_id: str
    settled_only: bool = True


@dataclass(frozen=True)
class Settled:
    """Tell the completion finalizer a workflow may have reached its end."""

    workflow_id: str


@dataclass(frozen=True)
class Cleanup:
    """Run a caller's teardown of what a task's dispatch exposed, unless a later
    dispatch holds the task by then."""

    task_id: str
    dispatch_id: str | None
    name: str
    run: Callable[[], None] = field(compare=False)


@dataclass(frozen=True)
class AuthorizeTurn:
    """Authorize a held model turn proposed while its workflow's writes were held,
    unless a later proposal of the turn superseded it or ``deadline_epoch`` passed."""

    proposal: AgentModelTurnProposal
    proposer_id: str
    incarnation: int
    deadline_epoch: float


AfterCommit = (
    CreditRelease
    | Reap
    | Interrupt
    | Revoke
    | Issue
    | Purge
    | Settled
    | Cleanup
    | AuthorizeTurn
)


class AfterCommitActions:
    """Holds each workflow's actions until its transition commits, queues committed
    actions for delivery off the lock, and keeps each action whose delivery failed for
    a retry."""

    def __init__(self, tasks: dict[str, TaskRecord]) -> None:
        self._tasks = tasks
        # Actions whose workflow has writes not yet durable, or whose transition is
        # still running.
        self.parked: dict[str, list[AfterCommit]] = {}
        self.ready: list[tuple[str | None, AfterCommit]] = []
        # Committed actions whose delivery failed, kept for the workflow's retry.
        self.failed: dict[str, list[AfterCommit]] = {}
        # Each cleanup filed and not yet delivered.
        self.cleanups: set[Cleanup] = set()

    def file_locked(self, workflow_id: str, *actions: AfterCommit) -> None:
        """Hold actions until the workflow's transition commits, each once."""
        if actions:
            parked = self.parked.setdefault(workflow_id, [])
            for action in actions:
                if action not in parked and self._owe(action):
                    parked.append(action)

    def queue_locked(self, *actions: AfterCommit) -> None:
        """Queue actions whose cause is already durable for delivery."""
        self.ready.extend((None, action) for action in actions if self._owe(action))

    def _owe(self, action: AfterCommit) -> bool:
        """Note a cleanup as owed; returns whether the action is to be held, which a
        cleanup already owed is not."""
        if not isinstance(action, Cleanup):
            return True
        if action in self.cleanups:
            return False
        self.cleanups.add(action)
        return True

    def cleanup_delivered_locked(self, cleanup: Cleanup) -> None:
        self.cleanups.discard(cleanup)

    def cleanup_owed(self, task_id: str) -> bool:
        """Whether a cleanup of the task is filed and not yet delivered."""
        return any(cleanup.task_id == task_id for cleanup in self.cleanups)

    def release_locked(self, durable: Callable[[str], bool]) -> None:
        """Queue each parked action whose workflow ``durable`` reports committed."""
        for workflow_id in [w for w in self.parked if durable(w)]:
            self.ready.extend(
                (workflow_id, action) for action in self.parked.pop(workflow_id)
            )

    def take_ready(self) -> list[tuple[str | None, AfterCommit]]:
        ready, self.ready = self.ready, []
        return ready

    def retain_failed_locked(
        self, workflow_id: str | None, action: AfterCommit
    ) -> None:
        """Keep an action whose delivery failed for its workflow's retry; one with no
        workflow is queued again."""
        if workflow_id is None:
            self.ready.append((None, action))
        else:
            self.failed.setdefault(workflow_id, []).append(action)

    def retry_failed_locked(self, workflow_id: str) -> None:
        """Queue the workflow's failed actions for delivery again."""
        self.ready.extend(
            (workflow_id, action) for action in self.failed.pop(workflow_id, [])
        )

    def has_failed(self, workflow_id: str) -> bool:
        return workflow_id in self.failed

    def forget_workflow_locked(self, workflow_id: str) -> None:
        self.parked.pop(workflow_id, None)
        self.failed.pop(workflow_id, None)

    @staticmethod
    def interrupt_for(record: TaskRecord, reason: str) -> Interrupt | None:
        # A merged child's batch keeps running for siblings from other workflows.
        if not record.assigned_worker or record.merged_parent_id:
            return None
        return Interrupt(
            InterruptMessage(
                task_id=record.task_id,
                worker_id=record.assigned_worker,
                reason=reason,
                dispatch_id=record.dispatch_id,
            )
        )

    @staticmethod
    def revoke(
        task_id: str, worker_id: str, dispatch_id: str | None, node_id: str | None
    ) -> Revoke | None:
        """The revocation of a dispatch that resolved without its worker ending it."""
        if dispatch_id is None:
            return None
        return Revoke(
            RevokeMessage(
                task_id=task_id, worker_id=worker_id, dispatch_id=dispatch_id
            ),
            node_id,
        )

    def cancelling_interrupts_locked(
        self, include: Callable[[TaskRecord], bool]
    ) -> list[tuple[str, Interrupt]]:
        """An interrupt for each task being cancelled that ``include`` selects, with
        its workflow."""
        return [
            (record.workflow_id, interrupt)
            for record in self._tasks.values()
            if record.status == TaskStatus.CANCELLING
            and include(record)
            and (interrupt := self.interrupt_for(record, record.error or "cancelled"))
        ]
