"""Durable commits of task transitions and workflow ledgers."""

import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from redis.exceptions import RedisError

from server.telemetry.tracing import ControlPlaneTracer

from ...orchestration import OrchestrationEngine
from ...registries.workflow import PersistedTask, WorkflowRegistry, WorkflowSched
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

# What a durable write raises when the store did not take it. Anything else a write
# raises is a fault in the transition itself.
PERSISTENCE_ERRORS: tuple[type[BaseException], ...] = (
    RedisError,
    OSError,
)


class TransitionNotDurable(Exception):
    """A transition applied in memory whose durable writes are held for a retry.

    The transition is not to be applied again: a retry of the same request finds it
    applied and makes it durable.
    """

    def __init__(self, held: dict[str, BaseException]) -> None:
        self.held = held
        first = next(iter(held.values()))
        super().__init__(
            f"writes of workflow(s) {', '.join(sorted(held))} are not durable: {first}"
        )


@dataclass(frozen=True)
class _Records:
    """Commit task records, with their status-set moves when ``membership``, and the
    workflow's schedule when ``sched``."""

    task_ids: tuple[str, ...]
    membership: bool = True
    sched: bool = False


@dataclass(frozen=True)
class _Snapshot:
    """Save the workflow's ledger, through the child seam with any child it has not
    written, retiring ``retire`` from its remaining set."""

    retire: tuple[str, ...] = ()


_Write = _Records | _Snapshot


class _Scope(threading.local):
    depth: int = 0
    held: dict[str, BaseException]

    def __init__(self) -> None:
        self.held = {}


@dataclass
class _Unacknowledged:
    """A report whose transition applied in memory with writes still held: what it did
    to the task, answered to the report handled again."""

    report: str
    worker_id: str
    dispatch_id: str | None
    outcome: SettleOutcome | FailureOutcome


class TransitionCommitter:
    """Commits task records, their status-set moves and each workflow's ledger.

    A write the store does not take is held in its workflow's ordered debt, and every
    later write of that workflow repays the debt first, so no write overtakes one held
    before it. The writes are state-derived, so repaying one writes current state.
    """

    def __init__(
        self,
        epochs: EpochFrontier,
        resident_tasks: ResidentServeTasks,
        reservations: WorkerReservations,
        actions: AfterCommitActions,
        record_failures: RecordFailures,
        tasks: dict[str, TaskRecord],
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
        # Each workflow's writes the store has not taken, in the order they were made.
        self.debt: dict[str, list[_Write]] = {}
        self.held_errors: dict[str, BaseException] = {}
        self._repaying: set[str] = set()
        # Children materialized in memory whose records the ledger seam has not written.
        self.unwritten_children: dict[str, set[str]] = {}
        self.unacknowledged: dict[str, _Unacknowledged] = {}
        self.retired_region_templates: dict[str, set[str]] = {}
        self.on_workflow_settled: Callable[[str], None] | None = None
        self.on_debt: Callable[[str], None] | None = None

    # ------------------------------------------------------------------ #
    # Transition scopes and debt
    # ------------------------------------------------------------------ #

    def enter_scope(self) -> bool:
        """Open a transition scope on this thread; returns whether it is outermost."""
        self._scope.depth += 1
        return self._scope.depth == 1

    def exit_scope(self) -> dict[str, BaseException]:
        """Close a transition scope, returning the errors of the writes the outermost
        one held, by workflow."""
        self._scope.depth -= 1
        if self._scope.depth:
            return {}
        held, self._scope.held = self._scope.held, {}
        # A write held and repaid within the same transition left nothing owed.
        return {
            workflow_id: error
            for workflow_id, error in held.items()
            if workflow_id in self.debt
        }

    def in_scope(self) -> bool:
        return self._scope.depth > 0

    def durable(self, workflow_id: str) -> bool:
        """Whether every write of the workflow made so far is durable."""
        return workflow_id not in self.debt

    def close_locked(self, workflow_id: str) -> bool:
        """Repay a workflow's held writes; returns whether it is durable."""
        if workflow_id not in self.debt:
            return True
        if self._repay_locked(workflow_id):
            return True
        self._note_held(workflow_id, self.held_errors[workflow_id])
        return False

    def mark_dirty_locked(self, workflow_id: str) -> None:
        """Owe a write of every record and the ledger of a workflow whose in-memory
        state may hold changes no write carried."""
        if workflow_id in self.debt:
            return
        task_ids = tuple(
            task_id
            for task_id, record in self._tasks.items()
            if record.workflow_id == workflow_id
        )
        self.debt[workflow_id] = [_Records(task_ids, sched=True), _Snapshot()]
        if self.on_debt is not None:
            self.on_debt(workflow_id)

    def forget_workflow_locked(self, workflow_id: str) -> None:
        self.debt.pop(workflow_id, None)
        self.held_errors.pop(workflow_id, None)
        self.unwritten_children.pop(workflow_id, None)
        self.retired_region_templates.pop(workflow_id, None)

    def _write_locked(
        self, workflow_id: str, entry: _Write, write: Callable[[], Any]
    ) -> bool:
        """Make one durable write of a workflow after the writes it holds, or hold it
        behind them; returns whether it was made."""
        if workflow_id in self._repaying:
            write()
            return True
        if workflow_id in self.debt and not self._repay_locked(workflow_id):
            self._hold(workflow_id, entry, self.held_errors[workflow_id])
            return False
        try:
            write()
        except PERSISTENCE_ERRORS as exc:
            self._hold(workflow_id, entry, exc)
            return False
        return True

    def _hold(self, workflow_id: str, entry: _Write, error: BaseException) -> None:
        held = self.debt.setdefault(workflow_id, [])
        # Repaying a write writes current state, so a write already owed is owed once.
        if entry not in held:
            held.append(entry)
        self.held_errors[workflow_id] = error
        self._note_held(workflow_id, error)

    def _note_held(self, workflow_id: str, error: BaseException) -> None:
        if self.on_debt is not None:
            self.on_debt(workflow_id)
        if not self._scope.depth:
            raise TransitionNotDurable({workflow_id: error})
        self._scope.held[workflow_id] = error

    def _repay_locked(self, workflow_id: str) -> bool:
        """Make a workflow's held writes in order; returns whether all were made."""
        held = self.debt[workflow_id]
        self._repaying.add(workflow_id)
        try:
            while held:
                entry = held[0]
                if isinstance(entry, _Records):
                    self._commit_records_raw(
                        workflow_id, entry.task_ids, entry.membership, entry.sched
                    )
                    held.pop(0)
                elif written := self._save_snapshot_raw(workflow_id, entry.retire):
                    # The children may have moved since they were materialized.
                    held[0] = _Records(tuple(written), sched=True)
                else:
                    held.pop(0)
        except PERSISTENCE_ERRORS as exc:
            self.held_errors[workflow_id] = exc
            return False
        finally:
            self._repaying.discard(workflow_id)
        del self.debt[workflow_id]
        self.held_errors.pop(workflow_id, None)
        return True

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def persisted_task_locked(self, task_id: str) -> PersistedTask | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        return PersistedTask(
            record=record,
            depends_on=self._original_deps.get(task_id) or set(),
            epoch_index=self._epochs.task_epoch_index.get(task_id),
        )

    def records_locked(self, *task_ids: str) -> list[PersistedTask]:
        return [
            persisted
            for task_id in dict.fromkeys(task_ids)
            if (persisted := self.persisted_task_locked(task_id))
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
        with_membership: bool,
        sched: bool,
    ) -> None:
        """Commit task records of one workflow, with their membership and its schedule,
        as one atomic transaction; then report each resident task it ended, note a
        terminal, and release the worker of each dispatch it ended."""
        ids = [
            task_id
            for task_id in dict.fromkeys(task_ids)
            if (record := self._tasks.get(task_id)) is not None
            and record.workflow_id == workflow_id
        ]
        if self.unwritten_children.get(workflow_id, set()) & set(ids):
            # A child's own record never lands before the seam that records it.
            self._commit_children_raw(workflow_id, ())
        by_status: dict[str, list[str]] = defaultdict(list)
        if with_membership:
            for task_id in ids:
                by_status[membership(self._tasks[task_id])].append(task_id)
        records = self.records_locked(*ids)
        self._workflow_registry.commit_transition(
            workflow_id,
            records=records,
            dispatched=by_status[TaskStatus.DISPATCHED],
            pending=by_status[TaskStatus.PENDING],
            done=by_status[TaskStatus.DONE],
            failed=by_status[TaskStatus.FAILED],
            cancelled=by_status[TaskStatus.CANCELLED],
            sched=self._sched_locked(workflow_id) if sched else None,
        )
        self._observe_resident_locked(records)
        if any(by_status[status] for status in TERMINAL_TASK_STATUSES):
            self._actions.file_locked(workflow_id, Settled(workflow_id))
        self._reservations.release_ended_dispatches_locked(ids)

    def _observe_resident_locked(self, records: Sequence[PersistedTask]) -> None:
        """Report each committed resident task that ended or reported an update under
        its current dispatch."""
        for persisted in records:
            if persisted.record.resident:
                self._resident_tasks.observe_resident_locked(persisted.record)

    def _commit_children_raw(
        self, workflow_id: str, retire: Sequence[str]
    ) -> list[str]:
        """Commit a workflow's unwritten children with its ledger snapshot and the
        retire, as one atomic transaction; returns the children it wrote."""
        engine = self._engines.get(workflow_id)
        children = sorted(self.unwritten_children.get(workflow_id, ()))
        if engine is None or not (children or retire):
            return []
        self._workflow_registry.commit_dynamic_tasks(
            workflow_id, self.records_locked(*children), engine.to_snapshot(), retire
        )
        if unwritten := self.unwritten_children.get(workflow_id):
            unwritten.difference_update(children)
            if not unwritten:
                del self.unwritten_children[workflow_id]
        return children

    def _save_snapshot_raw(self, workflow_id: str, retire: Sequence[str]) -> list[str]:
        """Save a workflow's ledger snapshot; returns the children it wrote with it."""
        engine = self._engines.get(workflow_id)
        if engine is None:
            return []
        if retire or self.unwritten_children.get(workflow_id):
            return self._commit_children_raw(workflow_id, retire)
        with self._control.ledger_snapshot(workflow_id):
            self._workflow_registry.save_ledger_snapshot(
                workflow_id, engine.to_snapshot()
            )
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
        ids = tuple(dict.fromkeys(task_ids))
        return self._write_locked(
            workflow_id,
            _Records(ids, with_membership, sched),
            lambda: self._commit_records_raw(workflow_id, ids, with_membership, sched),
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
        self._write_locked(
            workflow_id,
            _Snapshot(),
            lambda: self._save_snapshot_raw(workflow_id, ()),
        )

    def note_child_locked(self, workflow_id: str, child_task_id: str) -> None:
        """Mark a child materialized in memory, so the ledger snapshot that carries its
        work item also writes its record."""
        self.unwritten_children.setdefault(workflow_id, set()).add(child_task_id)

    def commit_new_children_locked(
        self,
        workflow_id: str,
        engine: OrchestrationEngine,
        child_task_ids: list[str],
        retire: Sequence[str] = (),
    ) -> None:
        """Persist new child records atomically with the ledger snapshot they belong to.

        Persisting the child records and the snapshot in one transaction keeps a
        dynamically materialized child from being durably half-recorded — a ledger work
        item without its task record, or a task record with no ledger work item — across
        a crash. ``retire`` drops the sealed spawn's child template from the remaining
        set in the same transaction, so the children replace it without a window in
        which the workflow reads as complete.
        """
        if not (child_task_ids or retire):
            return
        for child_task_id in child_task_ids:
            self.note_child_locked(workflow_id, child_task_id)
        self._persist_declared_failures_locked(engine, child_task_ids)
        retired = tuple(retire)
        self._write_locked(
            workflow_id,
            _Snapshot(retired),
            lambda: self._commit_children_raw(workflow_id, retired),
        )
        if retire:
            self.retired_region_templates.setdefault(workflow_id, set()).update(retire)
            # A retire drains the remaining set as a terminal does, and can drain its
            # last entry: a spawn that seals with no children leaves the workflow
            # complete with no task terminal behind it.
            self._actions.file_locked(workflow_id, Settled(workflow_id))

    def retire_sealed_region_templates_locked(
        self, workflow_id: str, engine: OrchestrationEngine
    ) -> None:
        """Retire an agent-region child template once its spawn region has sealed.

        A dynamic spawn region's child body is a template, never dispatched as a task;
        once the region seals it no longer holds the workflow open, so it is dropped
        from the remaining set (idempotently, tracked per workflow) with the ledger.
        """
        already = self.retired_region_templates.setdefault(workflow_id, set())
        if pending := engine.sealed_region_child_templates() - already:
            self.commit_new_children_locked(
                workflow_id, engine, [], retire=sorted(pending)
            )

    def commit_cancelled_locked(
        self, workflow_id: str, touched: list[str], returned: list[str]
    ) -> None:
        """Commit the tasks a cancel moved, then the merged children it returned to the
        queue."""

        def commit() -> None:
            records = self.records_locked(*touched)
            self._workflow_registry.commit_transition(
                workflow_id,
                records=records,
                dispatched=[
                    task_id
                    for task_id in touched
                    if membership(self._tasks[task_id]) == TaskStatus.DISPATCHED
                ],
                done=[
                    task_id
                    for task_id in touched
                    if membership(self._tasks[task_id]) == TaskStatus.DONE
                ],
                cancelled=[
                    task_id
                    for task_id in touched
                    if membership(self._tasks[task_id]) == TaskStatus.CANCELLED
                ],
                sched=self._sched_locked(workflow_id),
            )
            self._observe_resident_locked(records)

        self._write_locked(workflow_id, _Records(tuple(touched), sched=True), commit)
        self.commit_locked(*(task_id for task_id in returned if task_id not in touched))

    def _persist_declared_failures_locked(
        self, engine: OrchestrationEngine, new: Sequence[str] = ()
    ) -> None:
        """Fail and persist each task the engine settled as a declared failure whose
        record has not settled, ahead of a ledger write that reflects it. ``new`` are
        records the write itself creates."""
        failed: list[str] = []
        for task_id, reason in engine.declared_failures().items():
            record = self._tasks.get(task_id)
            if (
                record is None
                or record.status in SETTLING_TASK_STATUSES
                or task_id in new
            ):
                continue
            self._record_failures.fail_record_locked(record, reason)
            failed.append(task_id)
        if failed:
            self.commit_locked(*failed)

    def repersist_terminal_workflow_locked(self, workflow_id: str) -> None:
        """Re-commit the workflow's already-terminal tasks and schedule state.

        The idempotency guard calls this on a replayed terminal event: the original
        transition may have failed its persist after committing in memory, so re-
        committing makes the durable state current before the consumer's cursor advances
        past the event (else the task re-runs after a restart). It covers the whole
        workflow, not just the replayed task, because a cascade's other affected tasks
        aren't identifiable here. Idempotent; only on a rare duplicate replay.
        """
        terminal_ids = [
            task_id
            for task_id, record in self._tasks.items()
            if record.workflow_id == workflow_id
            and record.status in TERMINAL_TASK_STATUSES
        ]
        self.commit_records_locked(workflow_id, terminal_ids, sched=True)

    # ------------------------------------------------------------------ #
    # Settlement
    # ------------------------------------------------------------------ #

    def reclaim_vault_if_settled_locked(self, workflow_id: str) -> None:
        """Purge a workflow's vaulted credentials once its last task has settled and
        the settlement is durable.

        Called after an event's advance materializes any new children, so a producer
        that fans out is not reclaimed while its children are still pending.
        """
        if self.workflow_settlement_locked(workflow_id).settled:
            self._actions.file_locked(workflow_id, Purge(workflow_id))

    def workflow_settlement_locked(self, workflow_id: str) -> WorkflowSettlement:
        # A retired task -- a sealed spawn's child template, replaced by the children it
        # instantiated -- no longer holds the workflow open, and its record stays
        # PENDING forever because it is never dispatched. Counting it would leave every
        # workflow with a spawn region permanently unsettled.
        retired = self.retired_region_templates.get(workflow_id) or set()
        records = [
            r
            for r in self._tasks.values()
            if r.workflow_id == workflow_id and r.task_id not in retired
        ]
        if not records or any(r.status not in TERMINAL_TASK_STATUSES for r in records):
            return WorkflowSettlement(settled=False, finished_ts=None)
        finishes = [r.finished_ts for r in records if r.finished_ts is not None]
        return WorkflowSettlement(
            settled=True, finished_ts=max(finishes) if finishes else None
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

    def set_completion_notifier(self, notify: Callable[[str], None]) -> None:
        """Install the callback that a terminal transition notifies."""
        self.on_workflow_settled = notify

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

        A transition whose durable write the store does not take completes in memory,
        holds that write, and raises ``TransitionNotDurable``. The report handled again
        makes what was held and returns what the first handling did; it keeps doing so
        until one handling leaves nothing held. ``transition`` runs in the caller's
        transition scope.
        """
        # A report naming no worker has nothing to replay against, and a nested one
        # runs under the outer report.
        if worker_id is None or self._scope.depth > 1:
            return transition()
        with self._lock:
            # A stash matches only the same report from the same worker, naming its
            # dispatch or none.
            pending = self.unacknowledged.get(task_id)
            if pending is not None and (
                pending.report != report
                or pending.worker_id != worker_id
                or dispatch_id not in (None, pending.dispatch_id)
            ):
                pending = None
            if pending is not None and (record := self._tasks.get(task_id)):
                self.close_locked(record.workflow_id)
        outcome = transition()
        # A replay sees its own event as stale, so it answers with what the first
        # handling did.
        if pending is not None:
            outcome = cast(O, pending.outcome)
        with self._lock:
            if any(workflow_id in self.debt for workflow_id in self._scope.held):
                if pending is None:
                    self.unacknowledged[task_id] = _Unacknowledged(
                        report, worker_id, dispatch_id, outcome
                    )
            elif pending is not None and self.unacknowledged.get(task_id) is pending:
                del self.unacknowledged[task_id]
        return outcome

    def drop_unacknowledged(self, task_id: str) -> None:
        del self.unacknowledged[task_id]
