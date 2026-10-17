"""What a runtime transition owes workers and other consumers once it is durable."""

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from shared.schemas.command import InterruptMessage, RevokeMessage
from shared.tasks.artifact_upload import UploadSettled
from shared.tools.contract import AgentModelTurnProposal

from ..models import TaskRecord, TaskStatus


@dataclass(frozen=True)
class CreditRelease:
    """Release a committed invocation terminal's resident credit."""

    invocation_id: str
    failed: bool


@dataclass(frozen=True)
class Reap:
    """Tell a worker to drop a request it holds for a boundary that will not run."""

    worker_id: str
    task_id: str
    call_correlation: str
    # Whether the request went to resident capacity; None when that is not known.
    resident: bool | None = False


@dataclass(frozen=True)
class Interrupt:
    """Interrupt a task on its worker."""

    message: InterruptMessage


@dataclass(frozen=True)
class Revoke:
    """Revoke a dispatch on its worker."""

    message: RevokeMessage
    # The dispatch's node, which outlives the worker's record.
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
    """Purge a workflow's vaulted credentials, once it has settled when
    ``settled_only``."""

    workflow_id: str
    settled_only: bool = True


@dataclass(frozen=True)
class Settled:
    """Tell the completion finalizer a workflow may have reached its end."""

    workflow_id: str


@dataclass(frozen=True)
class AuthorizeTurn:
    """Authorize a held model turn proposed while its workflow's writes were held,
    unless a later proposal of the turn superseded it or ``deadline_epoch`` passed."""

    proposal: AgentModelTurnProposal
    proposer_id: str
    incarnation: int
    deadline_epoch: float


@dataclass(frozen=True)
class UploadAck:
    """Tell the worker a dispatch's artifact upload of one task is owed or settled."""

    worker_id: str
    settled: UploadSettled


@dataclass(frozen=True)
class TransferMultipart:
    """Move the multipart uploads a dispatch owns in a scope to the upload obligation
    its success left owed."""

    scope: str
    dispatch_id: str


type AfterCommit = (
    CreditRelease
    | Reap
    | Interrupt
    | Revoke
    | Issue
    | Purge
    | Settled
    | AuthorizeTurn
    | UploadAck
    | TransferMultipart
)


class _Filing(threading.local):
    # The workflows this thread's open outermost transition filed actions of.
    filed: set[str] | None = None


class AfterCommitActions:
    """Holds each workflow's actions until its transition commits, queues committed
    actions for delivery off the lock, and keeps each action whose delivery failed for
    a retry."""

    def __init__(self) -> None:
        # Actions whose workflow has writes not yet durable, or whose transition is
        # still running.
        self.parked: dict[str, list[AfterCommit]] = {}
        self.ready: list[tuple[str | None, AfterCommit]] = []
        # Committed actions whose delivery failed, kept for the workflow's retry.
        self.failed: dict[str, list[AfterCommit]] = {}
        # How many open transitions filed actions of each workflow.
        self._filing: dict[str, int] = {}
        # How many of each workflow's actions are filed and not yet delivered, whether
        # parked, queued, handed out for delivery or kept for a retry.
        self._owed: dict[str, int] = {}
        self._scope = _Filing()

    def open_scope(self) -> None:
        """Start an outermost transition on this thread: what it files stays parked
        until it closes, whichever thread releases."""
        self._scope.filed = set()

    def close_scope_locked(self) -> None:
        filed, self._scope.filed = self._scope.filed or set(), None
        for workflow_id in filed:
            if (count := self._filing[workflow_id] - 1) > 0:
                self._filing[workflow_id] = count
            else:
                del self._filing[workflow_id]

    def file_locked(self, workflow_id: str, *actions: AfterCommit) -> None:
        """Hold actions until the workflow's transition commits, each once."""
        if actions:
            if (filed := self._scope.filed) is not None and workflow_id not in filed:
                filed.add(workflow_id)
                self._filing[workflow_id] = self._filing.get(workflow_id, 0) + 1
            parked = self.parked.setdefault(workflow_id, [])
            for action in actions:
                if action not in parked:
                    parked.append(action)
                    self._owed[workflow_id] = self._owed.get(workflow_id, 0) + 1

    def queue_locked(self, *actions: AfterCommit) -> None:
        """Queue actions whose cause is already durable for delivery."""
        self.ready.extend((None, action) for action in actions)

    def release_locked(self, durable: Callable[[str], bool]) -> None:
        """Queue each parked action whose workflow ``durable`` reports committed and
        no open transition filed for."""
        for workflow_id in [
            w for w in self.parked if w not in self._filing and durable(w)
        ]:
            self.ready.extend(
                (workflow_id, action) for action in self.parked.pop(workflow_id)
            )

    def take_ready(self) -> list[tuple[str | None, AfterCommit]]:
        ready, self.ready = self.ready, []
        return ready

    def retain_failed_locked(self, workflow_id: str, action: AfterCommit) -> None:
        """Keep an action whose delivery failed for its workflow's retry."""
        self.failed.setdefault(workflow_id, []).append(action)

    def retry_failed_locked(self, workflow_id: str) -> None:
        """Queue the workflow's failed actions for delivery again."""
        self.ready.extend(
            (workflow_id, action) for action in self.failed.pop(workflow_id, [])
        )

    def has_failed(self, workflow_id: str) -> bool:
        return workflow_id in self.failed

    def delivered_locked(self, workflow_id: str) -> bool:
        """Count one of the workflow's actions delivered; returns whether it is owed
        none."""
        if (count := self._owed[workflow_id] - 1) > 0:
            self._owed[workflow_id] = count
            return False
        del self._owed[workflow_id]
        return True

    def owes(self, workflow_id: str) -> bool:
        """Whether any action of the workflow is still to be delivered."""
        return workflow_id in self._owed

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
                dispatch_id=record.internal.dispatch_id,
            )
        )

    @staticmethod
    def revoke_for(
        task_id: str, worker_id: str, dispatch_id: str | None, node_id: str | None
    ) -> Revoke | None:
        """Build the revocation of a dispatch resolved without its worker ending it."""
        if dispatch_id is None:
            return None
        return Revoke(
            RevokeMessage(
                task_id=task_id, worker_id=worker_id, dispatch_id=dispatch_id
            ),
            node_id,
        )

    @classmethod
    def cancelling_interrupts(
        cls, records: Iterable[TaskRecord]
    ) -> list[tuple[str, Interrupt]]:
        """Build an interrupt for each of ``records`` being cancelled, with its
        workflow."""
        return [
            (record.workflow_id, interrupt)
            for record in records
            if record.status == TaskStatus.CANCELLING
            and (interrupt := cls.interrupt_for(record, record.error or "cancelled"))
        ]
