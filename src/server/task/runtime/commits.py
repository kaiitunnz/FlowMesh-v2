"""Durable commits of task transitions and workflow ledgers."""

import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from redis.exceptions import (
    BusyLoadingError,
    ClusterDownError,
    OutOfMemoryError,
    ReadOnlyError,
    ResponseError,
    TryAgainError,
)

from server.telemetry.tracing import ControlPlaneTracer

from ...clients.redis import REDIS_CONN_ERRORS
from ...orchestration import OrchestrationEngine
from ...registries.workflow import (
    PersistedTask,
    WorkflowControl,
    WorkflowRegistry,
    WorkflowSched,
)
from ..models import (
    SETTLING_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    FailureOutcome,
    SettleOutcome,
    TaskRecord,
    TaskStatus,
    WorkflowSettlement,
)
from .after_commit import AfterCommitActions, Purge, Settled
from .record_failures import RecordFailures
from .reports import membership
from .reservations import WorkerReservations
from .resident_tasks import ResidentServeTasks
from .scheduling import EpochFrontier
from .task_table import TaskTable

_UNAVAILABLE_ERRORS: tuple[type[BaseException], ...] = (
    *REDIS_CONN_ERRORS,
    BusyLoadingError,
    ClusterDownError,
    OutOfMemoryError,
    ReadOnlyError,
    TryAgainError,
)
_UNAVAILABLE_REPLIES = frozenset(
    {
        "BUSY",
        "CLUSTERDOWN",
        "LOADING",
        "MASTERDOWN",
        "MISCONF",
        "NOREPLICAS",
        "OOM",
        "READONLY",
        "TRYAGAIN",
    }
)

_PIPELINE_ERROR_NOTE = "caused error: "


def store_unavailable(error: BaseException) -> bool:
    """Whether a write failed because the store could not take writes for now.

    Any other error a write raises, an argument the client cannot encode or a reply
    the command always gets, is a fault in the transition itself. A transaction the
    store discarded raises the reply of the command that discarded it, after the
    pipeline's note of which command that was.
    """
    if isinstance(error, _UNAVAILABLE_ERRORS):
        return True
    if not isinstance(error, ResponseError):
        return False
    reply = str(error).rpartition(_PIPELINE_ERROR_NOTE)[2]
    return reply.split(" ", 1)[0] in _UNAVAILABLE_REPLIES


class TransitionNotDurable(Exception):
    """Raised to a caller that acknowledges a request when writes its handling made
    are held for a retry.

    The handling applied in memory and is not to be applied again: the same request
    handled again finds it applied, and makes what it holds durable.
    """

    def __init__(
        self, held: dict[str, BaseException], stopped_partway: bool = False
    ) -> None:
        self.held = held
        # Whether the handling also stopped partway on an error of its own.
        self.stopped_partway = stopped_partway
        first = next(iter(held.values()))
        super().__init__(
            f"writes of workflow(s) {', '.join(sorted(held))} are not durable: {first}"
        )


def _control(engine: OrchestrationEngine) -> WorkflowControl:
    return WorkflowControl(
        open=engine.is_open(),
        failure=engine.control_failure(),
        cancelled=engine.instance_cancelled(),
    )


@dataclass(frozen=True)
class _Records:
    """A write of task records, with their status-set moves when ``membership``, and
    the workflow's schedule when ``sched``."""

    task_ids: tuple[str, ...]
    membership: bool = True
    sched: bool = False


@dataclass(frozen=True)
class _Ledger:
    """A write of the workflow's ledger changes, with the records of the children it
    materialized and has not written, removing ``retire`` from its remaining set."""

    retire: tuple[str, ...] = ()


type _Write = _Records | _Ledger


@dataclass
class _Debt:
    """What a workflow's writes the store has not taken still owe. Each is made from
    current state, so a write owed twice is made once."""

    # Each owed task record, and whether its status-set membership is owed with it.
    records: dict[str, bool] = field(default_factory=dict)
    sched: bool = False
    ledger: bool = False
    retire: set[str] = field(default_factory=set)

    def owe(self, write: _Write) -> None:
        match write:
            case _Records(task_ids=task_ids, membership=moves, sched=sched):
                for task_id in task_ids:
                    self.records[task_id] = self.records.get(task_id, False) or moves
                self.sched |= sched
            case _Ledger(retire=retire):
                self.ledger = True
                self.retire.update(retire)

    def copy(self) -> "_Debt":
        return _Debt(self.records.copy(), self.sched, self.ledger, self.retire.copy())

    def __bool__(self) -> bool:
        return bool(self.records or self.sched or self.ledger or self.retire)


class _Scope(threading.local):
    depth: int = 0
    held: dict[str, BaseException]
    # The writes held while an acknowledging caller's handling runs, by workflow, and
    # the scope depth that handling started at.
    acknowledging: dict[str, BaseException] | None = None
    acknowledging_depth: int = 0
    # The tasks whose reports that handling took up.
    guarded: set[str]
    reporting: bool = False

    def __init__(self) -> None:
        self.held = {}
        self.guarded = set()


@dataclass
class _Unacknowledged:
    """A report its caller has not acknowledged because writes of its handling are
    held: the workflows owing them, and what the handling answered, made and answered
    again by the report handled again."""

    report: str
    worker_id: str
    dispatch_id: str | None
    workflows: set[str]
    outcome: SettleOutcome | FailureOutcome


class TransitionCommitter:
    """Commits task records, their status-set moves and each workflow's ledger.

    A write the store does not take is held in its workflow's debt, and every later
    write of that workflow is made with the debt, so none lands ahead of it. Every
    write is made from current state, and the debt is made in one order: the task
    records, then the ledger with the records and status-set membership of the children
    it materialized. A ledger that lands therefore never leads the task records it
    reflects, and a child's record never lands ahead of the ledger that recorded it.
    """

    def __init__(
        self,
        epochs: EpochFrontier,
        resident_tasks: ResidentServeTasks,
        reservations: WorkerReservations,
        actions: AfterCommitActions,
        record_failures: RecordFailures,
        tasks: TaskTable,
        original_deps: dict[str, set[str]],
        engines: dict[str, OrchestrationEngine],
        workflow_registry: WorkflowRegistry,
        control: ControlPlaneTracer,
        logger: logging.Logger,
        lock: threading.RLock,
    ) -> None:
        self._epochs = epochs
        self._resident_tasks = resident_tasks
        self._reservations = reservations
        self._actions = actions
        self._record_failures = record_failures
        self._tasks = tasks
        self._original_deps = original_deps
        self._engines = engines
        self._workflow_registry = workflow_registry
        self._control = control
        self._logger = logger
        self._lock = lock
        self._scope = _Scope()
        # What each workflow's writes the store has not taken owe.
        self.debt: dict[str, _Debt] = {}
        self._making: set[str] = set()
        # Children materialized in memory whose records the ledger seam has not written.
        self._unwritten_children: dict[str, set[str]] = {}
        self.unacknowledged: dict[str, _Unacknowledged] = {}
        # How many acknowledging handlings have taken up a report of each task.
        self._reports: dict[str, int] = {}
        self.on_workflow_settled: Callable[[str], None] | None = None
        self.on_debt: Callable[[str], None] | None = None
        # Workflows owing a rewrite after a write raised a fault of its own, and what
        # runs once a write of one is made again.
        self.faulted: set[str] = set()
        self.on_fault_cleared: Callable[[str], None] | None = None

    # ------------------------------------------------------------------ #
    # Transition scopes and debt
    # ------------------------------------------------------------------ #

    def enter_scope(self) -> bool:
        """Open a transition scope on this thread; returns whether it is outermost."""
        self._scope.depth += 1
        return self._scope.depth == 1

    def exit_scope(self) -> tuple[dict[str, BaseException], dict[str, BaseException]]:
        """Close a transition scope, returning the errors of the writes the outermost
        one held, by workflow, and, when it is a runtime entry an acknowledging caller
        made, those of the writes its handling holds still."""
        entry = (
            self._scope.acknowledging is not None
            and self._scope.depth == self._scope.acknowledging_depth + 2
        )
        self._scope.depth -= 1
        refused = self._still_held(self._scope.acknowledging or {}) if entry else {}
        if self._scope.depth:
            return {}, refused
        held, self._scope.held = self._scope.held, {}
        return self._still_held(held), refused

    def open_acknowledging(self) -> bool:
        """Start collecting the writes this thread's handling holds, for a caller that
        acknowledges it; returns whether this is the outermost such handling."""
        if self._scope.acknowledging is not None:
            return False
        self._scope.acknowledging = {}
        self._scope.acknowledging_depth = self._scope.depth
        return True

    def close_acknowledging(self) -> dict[str, BaseException]:
        """Stop collecting, returning the errors of the writes the handling held that
        are still not durable, by workflow."""
        held, self._scope.acknowledging = self._scope.acknowledging or {}, None
        return self._still_held(held)

    def reporting(self, task_id: str) -> bool:
        """Whether an acknowledging caller is handling a report of the task."""
        return task_id in self._reports

    def end_reports_locked(self, acknowledged: bool) -> list[str]:
        """Release the reports this thread's handling took up, and when the caller
        acknowledged it, what each waits to be handled again; returns the tasks no
        handling holds any more."""
        guarded, self._scope.guarded = self._scope.guarded, set()
        ended = []
        for task_id in guarded:
            if acknowledged:
                self.unacknowledged.pop(task_id, None)
            if (count := self._reports[task_id] - 1) > 0:
                self._reports[task_id] = count
            else:
                del self._reports[task_id]
                ended.append(task_id)
        return ended

    def _still_held(self, held: dict[str, BaseException]) -> dict[str, BaseException]:
        # A write held and made later within the same handling left nothing owed.
        with self._lock:
            return {
                workflow_id: error
                for workflow_id, error in held.items()
                if workflow_id in self.debt
            }

    def durable(self, workflow_id: str) -> bool:
        """Whether every write of the workflow made so far is durable."""
        return workflow_id not in self.debt

    def close_locked(self, workflow_id: str) -> bool:
        """Make a workflow's held writes; returns whether it is durable."""
        if (owed := self.debt.get(workflow_id)) is None:
            return True
        return self._make_debt_locked(workflow_id, owed.copy())

    def mark_dirty_locked(self, workflow_id: str) -> None:
        """Owe a write of every record and the ledger of a workflow whose in-memory
        state may hold changes no write carried, and schedule it."""
        self._owe_rewrite_locked(workflow_id)
        if self.on_debt is not None:
            self.on_debt(workflow_id)

    def _owe_rewrite_locked(self, workflow_id: str) -> None:
        owed = self.debt.setdefault(workflow_id, _Debt())
        owed.owe(
            _Records(
                tuple(self._tasks.ids_of(workflow_id)),
                sched=True,
            )
        )
        owed.owe(_Ledger())
        if (engine := self._engines.get(workflow_id)) is not None:
            engine.owe_ledger_rewrite()

    def _write_locked(self, workflow_id: str, write: _Write) -> bool:
        """Make one durable write of a workflow with the writes it holds, or hold it
        with them; returns whether it was made."""
        if workflow_id in self._making:
            # The writes being made pick it up.
            self.debt[workflow_id].owe(write)
            return True
        before = self.debt.get(workflow_id)
        owed = before.copy() if before is not None else _Debt()
        owed.owe(write)
        self.debt[workflow_id] = owed
        return self._make_debt_locked(workflow_id, before)

    def _make_debt_locked(self, workflow_id: str, before: _Debt | None) -> bool:
        """Make what a workflow owes; returns whether all of it was made.

        A write the store cannot take for now stays owed and is held. Any other error
        is the transition's own and is raised. The workflow's in-memory state may then
        hold what no write carried, so a rewrite of all of it is owed, made with the
        workflow's next write rather than retried on its own, so a recurring fault
        never spins.
        """
        owed = self.debt[workflow_id]
        self._making.add(workflow_id)
        try:
            while owed:
                self._make_locked(workflow_id, owed)
        except Exception as exc:
            if not store_unavailable(exc):
                if before is None:
                    self.debt.pop(workflow_id, None)
                else:
                    self.debt[workflow_id] = before
                self._owe_rewrite_locked(workflow_id)
                if workflow_id not in self.faulted:
                    self.faulted.add(workflow_id)
                    self._logger.error(
                        "A write of workflow %s raised; its next write rewrites it: %s",
                        workflow_id,
                        exc,
                    )
                raise
            self._note_held(workflow_id, exc)
            return False
        finally:
            self._making.discard(workflow_id)
        del self.debt[workflow_id]
        if workflow_id in self.faulted:
            self.faulted.discard(workflow_id)
            if self.on_fault_cleared is not None:
                self.on_fault_cleared(workflow_id)
        return True

    def _make_locked(self, workflow_id: str, owed: _Debt) -> None:
        """Make one pass over what a workflow owes, records before the ledger."""
        unwritten = (
            self._unwritten_children.get(workflow_id, set())
            if workflow_id in self._engines
            else set()
        )
        if (ids := [t for t in owed.records if t not in unwritten]) or owed.sched:
            self._commit_records_raw(
                workflow_id, ids, [t for t in ids if owed.records[t]], owed.sched
            )
            for task_id in ids:
                del owed.records[task_id]
            owed.sched = False
        if not (owed.ledger or owed.retire or owed.records):
            return
        retire = sorted(owed.retire)
        # The ledger writes the children it materialized from their current state,
        # membership included, so it settles what they owe.
        for task_id in self._save_ledger_raw(workflow_id, retire):
            owed.records.pop(task_id, None)
        owed.ledger = False
        owed.retire.difference_update(retire)

    def _note_held(self, workflow_id: str, error: BaseException) -> None:
        if self.on_debt is not None:
            self.on_debt(workflow_id)
        self._scope.held[workflow_id] = error
        if self._scope.acknowledging is not None:
            self._scope.acknowledging[workflow_id] = error

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def _persisted_task_locked(self, task_id: str) -> PersistedTask | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        return PersistedTask(
            record=record,
            depends_on=self._original_deps.get(task_id) or set(),
            epoch_index=self._epochs.task_epoch_index.get(task_id),
        )

    def _records_locked(self, *task_ids: str) -> list[PersistedTask]:
        return [
            persisted
            for task_id in dict.fromkeys(task_ids)
            if (persisted := self._persisted_task_locked(task_id))
        ]

    def _sched_locked(self, workflow_id: str) -> WorkflowSched:
        return WorkflowSched(
            in_epoch_order=self._epochs.workflow_in_epoch_order.get(workflow_id, False),
            epoch_frontier=self._epochs.workflow_epoch_frontier.get(workflow_id, 0),
        )

    def _commit_records_raw(
        self,
        workflow_id: str,
        task_ids: Sequence[str],
        moves: Sequence[str],
        sched: bool,
    ) -> None:
        """Commit task records of one workflow, the status-set membership of ``moves``
        and its schedule, as one atomic transaction."""
        ids = {
            task_id: None
            for task_id in task_ids
            if (record := self._tasks.get(task_id)) is not None
            and record.workflow_id == workflow_id
        }
        by_status: dict[str, list[str]] = defaultdict(list)
        for task_id in dict.fromkeys(moves):
            if task_id in ids:
                by_status[membership(self._tasks[task_id])].append(task_id)
        records = self._records_locked(*ids)
        engine = self._engines.get(workflow_id)
        self._workflow_registry.commit_transition(
            workflow_id,
            records=records,
            dispatched=by_status[TaskStatus.DISPATCHED],
            pending=by_status[TaskStatus.PENDING],
            done=by_status[TaskStatus.DONE],
            failed=by_status[TaskStatus.FAILED],
            cancelled=by_status[TaskStatus.CANCELLED],
            sched=self._sched_locked(workflow_id) if sched else None,
            control=_control(engine) if engine is not None else None,
        )
        self._after_records_locked(workflow_id, records, by_status)

    def _after_records_locked(
        self,
        workflow_id: str,
        records: Sequence[PersistedTask],
        by_status: dict[str, list[str]],
    ) -> None:
        """Report each committed resident task that ended, file the completion notice
        of a terminal, and release the worker of each dispatch the commit ended."""
        self._observe_resident_locked(records)
        if any(by_status[status] for status in TERMINAL_TASK_STATUSES):
            self._actions.file_locked(workflow_id, Settled(workflow_id))
        self._reservations.release_ended_dispatches_locked(
            [persisted.record.task_id for persisted in records]
        )

    def _observe_resident_locked(self, records: Sequence[PersistedTask]) -> None:
        """Report each committed resident task that ended or reported an update under
        its current dispatch."""
        for persisted in records:
            if persisted.record.resident:
                self._resident_tasks.observe_resident_locked(persisted.record)

    def _commit_children_raw(
        self, workflow_id: str, retire: Sequence[str]
    ) -> list[str]:
        """Commit a workflow's unwritten children, with their status-set membership and
        the schedule, with its ledger changes and the retire, as one atomic
        transaction; returns the children it wrote."""
        engine = self._engines.get(workflow_id)
        children = sorted(self._unwritten_children.get(workflow_id, ()))
        if engine is None or not (children or retire):
            return []
        records = self._records_locked(*children)
        by_status: dict[str, list[str]] = defaultdict(list)
        for persisted in records:
            by_status[membership(persisted.record)].append(persisted.record.task_id)
        changes = engine.ledger_changes()
        self._workflow_registry.commit_dynamic_tasks(
            workflow_id,
            records,
            changes,
            retire,
            dispatched=by_status[TaskStatus.DISPATCHED],
            done=by_status[TaskStatus.DONE],
            failed=by_status[TaskStatus.FAILED],
            cancelled=by_status[TaskStatus.CANCELLED],
            sched=self._sched_locked(workflow_id) if children else None,
            control=_control(engine),
        )
        engine.ledger_written(changes)
        self._after_records_locked(workflow_id, records, by_status)
        if unwritten := self._unwritten_children.get(workflow_id):
            unwritten.difference_update(children)
            if not unwritten:
                del self._unwritten_children[workflow_id]
        return children

    def _save_ledger_raw(self, workflow_id: str, retire: Sequence[str]) -> list[str]:
        """Save a workflow's ledger changes; returns the children it wrote with them."""
        engine = self._engines.get(workflow_id)
        if engine is None:
            return []
        if retire or self._unwritten_children.get(workflow_id):
            return self._commit_children_raw(workflow_id, retire)
        with self._control.ledger_snapshot(workflow_id):
            changes = engine.ledger_changes()
            self._workflow_registry.save_ledger(workflow_id, changes, _control(engine))
        engine.ledger_written(changes)
        return []

    def _by_workflow_locked(self, task_ids: Sequence[str]) -> dict[str, list[str]]:
        by_workflow: dict[str, list[str]] = defaultdict(list)
        for task_id in dict.fromkeys(task_ids):
            if (record := self._tasks.get(task_id)) is not None:
                by_workflow[record.workflow_id].append(task_id)
        return by_workflow

    def commit_records_locked(
        self,
        workflow_id: str,
        task_ids: Sequence[str],
        *,
        with_membership: bool = True,
        sched: bool = False,
    ) -> bool:
        """Commit task records of one workflow, with their status-set membership and
        the workflow schedule when asked; returns whether the write was made."""
        return self._write_locked(
            workflow_id,
            _Records(tuple(dict.fromkeys(task_ids)), with_membership, sched),
        )

    def persist_locked(self, *task_ids: str) -> None:
        """Commit task records (no membership change), per workflow."""
        for workflow_id, ids in self._by_workflow_locked(task_ids).items():
            self.commit_records_locked(workflow_id, ids, with_membership=False)

    def commit_locked(self, *task_ids: str, sched: bool = True) -> None:
        """Commit each task's record and its status-set membership, and the workflow
        schedule, as one atomic transaction per workflow and the single last step of a
        transition.

        A terminal task moves to its done/failed/cancelled set, a pending one leaves the
        dispatched set, and a dispatched one joins it. Committing only after all
        in-memory mutations means a failed or crashed write can't leave durable state
        half-applied: the transaction commits in full or not at all. Assumes the
        in-memory mutations never raise, which holds while ordered tasks carry
        ``position_in_epoch`` (so the ready-queue helpers never hit their guards).
        """
        for workflow_id, ids in self._by_workflow_locked(task_ids).items():
            self.commit_records_locked(workflow_id, ids, sched=sched)

    def save_ledger_locked(self, workflow_id: str) -> None:
        """Save a workflow's ledger after the writes it holds. A workflow with no
        ledger has its terminal in its task records, committed before."""
        engine = self._engines.get(workflow_id)
        if engine is not None:
            self._persist_declared_failures_locked(engine)
        elif workflow_id not in self.debt:
            return
        self._write_locked(workflow_id, _Ledger())

    def note_child_locked(self, workflow_id: str, child_task_id: str) -> None:
        """Mark a child materialized in memory, so the ledger write that carries its
        work item also writes its record."""
        self._unwritten_children.setdefault(workflow_id, set()).add(child_task_id)

    def commit_cancelled_locked(
        self, workflow_id: str, touched: list[str], returned: list[str]
    ) -> None:
        """Commit the tasks a cancel moved, then the merged children it returned to the
        queue."""
        self.commit_locked(*touched)
        self.commit_locked(*(task_id for task_id in returned if task_id not in touched))

    def _persist_declared_failures_locked(self, engine: OrchestrationEngine) -> None:
        """Fail and persist each task the engine settled as a declared failure whose
        record has not settled, ahead of a ledger write that reflects it.

        A failure whose task has no record yet is applied by a later write; a failed
        record is committed, and owed when its write is held, as any other is.
        """
        failed: list[str] = []
        for task_id, reason in engine.unapplied_failures():
            if (record := self._tasks.get(task_id)) is None:
                continue
            if record.status not in SETTLING_TASK_STATUSES:
                self._record_failures.fail_record_locked(record, reason)
                failed.append(task_id)
            engine.mark_failure_applied(task_id)
        if failed:
            self.commit_locked(*failed)

    def recommit_terminal_locked(self, record: TaskRecord) -> None:
        """Re-commit a task a replayed terminal event reports, with the merged
        children its settlement settled.

        A write the transition made and the store refused is held and retried, and a
        transition that stopped partway owes its whole workflow, so the replay owes
        only the reported task's own records.
        """
        self.commit_locked(
            record.task_id,
            *(
                child_id
                for child_id in record.merged_children or ()
                if (child := self._tasks.get(child_id)) is not None
                and child.status in TERMINAL_TASK_STATUSES
            ),
        )

    # ------------------------------------------------------------------ #
    # Settlement
    # ------------------------------------------------------------------ #

    def settle_if_done_locked(self, workflow_id: str) -> None:
        """Once a workflow has settled, have its finalizer close it and purge its
        vaulted credentials, each once the settlement is durable.

        Called at the end of every transition that can settle a workflow, after its
        advance materializes any new children, so a producer that fans out does not
        settle its workflow while its children are still pending.
        """
        if self.workflow_settlement_locked(workflow_id).settled:
            self._actions.file_locked(
                workflow_id, Settled(workflow_id), Purge(workflow_id)
            )

    def workflow_settlement_locked(self, workflow_id: str) -> WorkflowSettlement:
        """Whether every task of a workflow has settled in memory, durable or not, and
        the last of their finishes.

        A workflow whose ledger still waits on a value read to route a branch or fan
        out a spawn has not settled, though none of its tasks holds it open.
        """
        engine = self._engines.get(workflow_id)
        if (
            not self._tasks.holds(workflow_id)
            or self._tasks.has_unsettled(workflow_id)
            or (engine is not None and engine.awaits_control_reads())
        ):
            return WorkflowSettlement(settled=False, finished_ts=None)
        return WorkflowSettlement(
            settled=True, finished_ts=self._tasks.last_finish(workflow_id)
        )

    def notify_terminal_transition(self, workflow_id: str) -> None:
        """Tell the completion finalizer a workflow may have reached its end.

        It carries a workflow id and nothing else: the finalizer decides whether the
        workflow is complete, and does so off this thread.
        """
        if self.on_workflow_settled is None:
            return
        try:
            self.on_workflow_settled(workflow_id)
        except Exception as exc:
            self._logger.debug(
                "Failed to notify workflow completion for %s: %s", workflow_id, exc
            )

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #

    def reported[O: (SettleOutcome, FailureOutcome)](
        self,
        report: str,
        task_id: str,
        worker_id: str | None,
        dispatch_id: str | None,
        transition: Callable[[], O],
    ) -> O:
        """Apply a worker's report to its task through ``transition``.

        A report an acknowledging caller handles holds its task's next publication
        until the handling ends, so what the caller does after the report runs before
        the task can be dispatched again. A handling whose writes are held is
        remembered with the workflows owing them and its answer. The report handled
        again is not applied again: it makes those writes and gives the same answer.
        ``transition`` runs in the caller's transition scope.
        """
        # A report naming no worker has nothing to replay against, and a nested one
        # runs under the outer report.
        if worker_id is None or self._scope.reporting:
            return transition()
        acknowledging = self._scope.acknowledging is not None
        with self._lock:
            if acknowledging and task_id not in self._scope.guarded:
                self._scope.guarded.add(task_id)
                self._reports[task_id] = self._reports.get(task_id, 0) + 1
            # A stash matches only the same report from the same worker, naming its
            # dispatch or none.
            pending = self.unacknowledged.get(task_id)
            if pending is not None and (
                pending.report != report
                or pending.worker_id != worker_id
                or dispatch_id not in (None, pending.dispatch_id)
            ):
                pending = None
            if pending is not None:
                for workflow_id in sorted(pending.workflows):
                    self.close_locked(workflow_id)
                if not (held := {w for w in pending.workflows if w in self.debt}):
                    del self.unacknowledged[task_id]
                    return cast(O, pending.outcome)
                pending.workflows = held
                raise self._not_durable(held)
        self._scope.reporting = True
        try:
            outcome = transition()
        finally:
            self._scope.reporting = False
        with self._lock:
            held = {w for w in self._scope.held if w in self.debt}
            if held and acknowledging:
                # Refused in the same lock hold, so no thread that makes the writes
                # durable meanwhile leaves the report acknowledged with its stash.
                self.unacknowledged[task_id] = _Unacknowledged(
                    report, worker_id, dispatch_id, held, outcome
                )
                raise self._not_durable(held)
        return outcome

    def _not_durable(self, workflows: set[str]) -> TransitionNotDurable:
        return TransitionNotDurable(
            {
                workflow_id: self._scope.held.get(workflow_id)
                or ConnectionError("store unavailable")
                for workflow_id in sorted(workflows)
            }
        )

    def drop_unacknowledged(self, task_id: str) -> None:
        del self.unacknowledged[task_id]
