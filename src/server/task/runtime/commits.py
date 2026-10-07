"""Durable commits of task transitions and workflow ledgers."""

import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Self, cast

from server.telemetry.tracing import ControlPlaneTracer

from ...orchestration import OrchestrationEngine
from ...registries.workflow import PersistedTask, WorkflowRegistry, WorkflowSched
from ...services.credential_vault import CredentialVault
from ..models import (
    SETTLING_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    FailureOutcome,
    SettleOutcome,
    TaskRecord,
    TaskStatus,
    WorkflowSettlement,
)
from .record_failures import RecordFailures
from .reports import membership
from .reservations import WorkerReservations
from .resident_tasks import ResidentServeTasks
from .scheduling import EpochFrontier
from .terminations import TerminationRelease


@dataclass
class HeldWrites:
    """The durable writes a report's transition holds back once one of them fails:
    the tasks to commit, the spawned children to commit, and the workflows whose
    ledger to save and credentials to reclaim."""

    task_ids: list[str] = field(default_factory=list)
    children: list[tuple[str, list[str], list[str]]] = field(default_factory=list)
    workflow_ids: list[str] = field(default_factory=list)
    error: Exception | None = None

    def follow(self, earlier: Self) -> None:
        """Hold another report's held writes ahead of this one's."""
        self.task_ids[:0] = earlier.task_ids
        self.children[:0] = earlier.children
        self.workflow_ids[:0] = earlier.workflow_ids


class _ReportWrites(threading.local):
    held: HeldWrites | None = None


@dataclass
class _Unacknowledged:
    """A report whose transition completed in memory and failed a durable write: what
    it did to the task, and the writes it held back."""

    report: str
    worker_id: str
    dispatch_id: str | None
    held: HeldWrites
    outcome: SettleOutcome | FailureOutcome


class TransitionCommitter:
    """Commits task records, their status-set moves and each workflow's ledger, holds
    back a report's writes once one fails so its replay completes them, and notifies
    each workflow's terminal transition."""

    def __init__(
        self,
        epochs: EpochFrontier,
        resident_tasks: ResidentServeTasks,
        reservations: WorkerReservations,
        terminations: TerminationRelease,
        record_failures: RecordFailures,
        tasks: dict[str, TaskRecord],
        original_deps: dict[str, set[str]],
        engines: dict[str, OrchestrationEngine],
        workflow_registry: WorkflowRegistry,
        credential_vault: CredentialVault,
        control: ControlPlaneTracer,
        logger: logging.Logger,
        lock: threading.RLock,
    ) -> None:
        self._epochs = epochs
        self._resident_tasks = resident_tasks
        self._reservations = reservations
        self._terminations = terminations
        self._record_failures = record_failures
        self._tasks = tasks
        self._original_deps = original_deps
        self._engines = engines
        self._workflow_registry = workflow_registry
        self._credential_vault = credential_vault
        self._control = control
        self._logger = logger
        self._lock = lock
        self.report_writes = _ReportWrites()
        self.unacknowledged: dict[str, _Unacknowledged] = {}
        self.retired_region_templates: dict[str, set[str]] = {}
        self.on_workflow_settled: Callable[[str], None] | None = None

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

    def commit_transition_locked(
        self,
        workflow_id: str,
        *,
        records: Sequence[PersistedTask] = (),
        dispatched: Sequence[str] = (),
        pending: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
    ) -> None:
        """Apply one workflow state delta, then report each resident task it ended or
        that reported an update under its current dispatch."""
        self._workflow_registry.commit_transition(
            workflow_id,
            records=records,
            dispatched=dispatched,
            pending=pending,
            done=done,
            failed=failed,
            cancelled=cancelled,
            sched=sched,
        )
        for persisted in records:
            if persisted.record.resident:
                self._resident_tasks.observe_resident_locked(persisted.record)

    def persist_locked(self, *task_ids: str) -> None:
        """Commit task records (no membership change) atomically, per workflow."""
        by_workflow: dict[str, list[str]] = defaultdict(list)
        for task_id in dict.fromkeys(task_ids):
            if record := self._tasks.get(task_id):
                by_workflow[record.workflow_id].append(task_id)
        for workflow_id, ids in by_workflow.items():
            self.commit_transition_locked(
                workflow_id, records=self.records_locked(*ids)
            )

    def commit_locked(self, *task_ids: str, sched: bool = True) -> None:
        """Commit each task's record and its status-set membership, and the workflow
        schedule, as one atomic transaction per workflow and the single last step of a
        transition.

        A terminal task moves to its done/failed/cancelled set, a pending one leaves the
        dispatched set, and a dispatched one joins it. Committing only after all
        in-memory mutations means a failed or crashed write can't leave durable state
        half-applied: the transaction commits in full or not at all. Event-driven
        callers additionally heal via the at-least-once replay
        (``_repersist_terminal_workflow_locked``); the API-driven cancel relies on this
        atomicity alone. Assumes the in-memory mutations never raise, which holds while
        ordered tasks carry ``position_in_epoch`` (so the ready-queue helpers never hit
        their guards).

        Within a worker's report, a failed commit is held back with every later write.
        """

        def commit() -> None:
            moves: dict[str, dict[str, list[str]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for task_id in dict.fromkeys(task_ids):
                if (record := self._tasks.get(task_id)) is not None:
                    moves[record.workflow_id][membership(record)].append(task_id)
            for workflow_id, by_status in moves.items():
                self.commit_transition_locked(
                    workflow_id,
                    records=self.records_locked(
                        *chain.from_iterable(by_status.values())
                    ),
                    dispatched=by_status[TaskStatus.DISPATCHED],
                    pending=by_status[TaskStatus.PENDING],
                    done=by_status[TaskStatus.DONE],
                    failed=by_status[TaskStatus.FAILED],
                    cancelled=by_status[TaskStatus.CANCELLED],
                    sched=self._sched_locked(workflow_id) if sched else None,
                )
                if any(by_status[status] for status in TERMINAL_TASK_STATUSES):
                    self.notify_terminal_transition(workflow_id)
            self._reservations.release_ended_dispatches_locked(task_ids)

        self._write_locked(commit, lambda held: held.task_ids.extend(task_ids))

    def _write_locked(
        self, write: Callable[[], None], hold: Callable[[HeldWrites], None]
    ) -> None:
        """Make one durable write, or hold it back within a worker's report once this
        or an earlier write of the report fails."""
        held = self.report_writes.held
        if held is not None and held.error is not None:
            hold(held)
            return
        try:
            write()
        except Exception as exc:
            if held is None:
                raise
            held.error = exc
            hold(held)

    def writes_held(self) -> bool:
        """Whether a durable write of the report being handled failed."""
        held = self.report_writes.held
        return held is not None and held.error is not None

    def recommit_locked(self, held: HeldWrites) -> None:
        """Make the durable writes a report held back: its tasks, then its spawned
        children, then each touched workflow's ledger and credential reclaim.

        The children are committed again after they are added, so each one's
        membership follows its current status.
        """
        self.commit_locked(*held.task_ids)
        for workflow_id, child_task_ids, retire in held.children:
            if (engine := self._engines.get(workflow_id)) is not None:
                self.commit_new_children_locked(
                    workflow_id, engine, child_task_ids, retire
                )
        self.commit_locked(*chain.from_iterable(ids for _, ids, _ in held.children))
        workflow_ids = [
            record.workflow_id
            for task_id in held.task_ids
            if (record := self._tasks.get(task_id)) is not None
        ]
        workflow_ids += [workflow_id for workflow_id, _, _ in held.children]
        for workflow_id in dict.fromkeys(workflow_ids + held.workflow_ids):
            self.save_ledger_locked(workflow_id)
            self.reclaim_vault_if_settled_locked(workflow_id)

    def notify_terminal_transition(self, workflow_id: str) -> None:
        """Tell the completion finalizer a workflow may have reached its end.

        Every terminal commit funnels through `_commit_locked`, whether a worker
        reported it or the control plane settled it alone, so one notification covers
        both.
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
        if terminal_ids:
            self.commit_locked(*terminal_ids)
        else:
            self.commit_transition_locked(
                workflow_id, sched=self._sched_locked(workflow_id)
            )

    def reclaim_vault_if_settled_locked(self, workflow_id: str) -> None:
        """Purge a workflow's vaulted credentials once its last task has settled.

        Called after an event's advance materializes any new children, so a producer
        that fans out is not reclaimed while its children are still pending.
        """
        if self.writes_held() or self.workflow_settlement_locked(workflow_id).settled:
            self._write_locked(
                lambda: self._credential_vault.purge(workflow_id),
                lambda held: held.workflow_ids.append(workflow_id),
            )

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
        if child_task_ids or retire:
            self._persist_declared_failures_locked(engine, child_task_ids)
            self._write_locked(
                lambda: self._workflow_registry.commit_dynamic_tasks(
                    workflow_id,
                    self.records_locked(*child_task_ids),
                    engine.to_snapshot(),
                    retire=retire,
                ),
                lambda held: held.children.append(
                    (workflow_id, list(child_task_ids), list(retire))
                ),
            )
            if retire:
                self.retired_region_templates.setdefault(workflow_id, set()).update(
                    retire
                )
                # A retire drains the remaining set as a terminal does, and can drain
                # its last entry: a spawn that seals with no children leaves the
                # workflow complete with no task terminal behind it. The two drains --
                # a terminal commit and a retire -- each notify, and they are the only
                # two, so no completion escapes the finalizer.
                if not self.writes_held():
                    self.notify_terminal_transition(workflow_id)

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

    def save_ledger_locked(self, workflow_id: str) -> None:
        """Save a workflow's ledger, and queue each termination waiting on it for
        release once the save succeeds. A workflow with no ledger has its terminal in
        its task records, committed before."""
        engine = self._engines.get(workflow_id)
        if engine is not None:
            self._persist_declared_failures_locked(engine)

        def save() -> None:
            if engine is not None:
                with self._control.ledger_snapshot(workflow_id):
                    self._workflow_registry.save_ledger_snapshot(
                        workflow_id, engine.to_snapshot()
                    )
            self._terminations.ledger_saved(workflow_id)

        if engine is None and not self._terminations.has_undurable(workflow_id):
            return
        self._write_locked(save, lambda held: held.workflow_ids.append(workflow_id))

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

    def reported[O: (SettleOutcome, FailureOutcome)](
        self,
        report: str,
        task_id: str,
        worker_id: str | None,
        dispatch_id: str | None,
        transition: Callable[[], O],
    ) -> O:
        """Apply a worker's report to its task through ``transition``.

        A transition whose durable write fails completes in memory, holds back its
        later writes, and raises. The report handled again makes what was held back,
        heals as a replay does, and returns what the first handling did; it keeps
        doing so until one handling completes.
        """
        # A report naming no worker has nothing to replay against, and a nested one
        # runs under the outer report's hold.
        if worker_id is None or self.report_writes.held is not None:
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
            if pending is not None:
                # Make the writes the first handling held back, before the replay
                # writes anything after them.
                self.recommit_locked(pending.held)
        # A fresh hold collects every write that fails from here on, so the transition
        # completes in memory and its writes stay ordered.
        held = self.report_writes.held = HeldWrites()
        try:
            outcome = transition()
        except Exception:
            # A replay that raises keeps its stash, now holding the replay's writes.
            if pending is not None and held.error is not None:
                with self._lock:
                    pending.held = held
            raise
        finally:
            self.report_writes.held = None
        # A replay sees its own event as stale, so it answers with what the first
        # handling did.
        if pending is not None:
            outcome = cast(O, pending.outcome)
        with self._lock:
            # A failed write stashes the report for its next handling; a clean replay
            # clears the stash unless a newer handling replaced it.
            if held.error is not None:
                # A stash of another report is replaced, but what it held stays held.
                replaced = self.unacknowledged.get(task_id)
                if replaced is not None and replaced is not pending:
                    held.follow(replaced.held)
                self.unacknowledged[task_id] = _Unacknowledged(
                    report, worker_id, dispatch_id, held, outcome
                )
            elif pending is not None and self.unacknowledged.get(task_id) is pending:
                del self.unacknowledged[task_id]
        # The caller sees the failure so the report is delivered again; a stashed hold
        # never raises it twice.
        if (error := held.error) is not None:
            held.error = None
            raise error
        return outcome

    def commit_cancelled_locked(
        self, workflow_id: str, touched: list[str], returned: list[str]
    ) -> None:
        """Commit the tasks a cancel moved, then the merged children it returned to the
        queue."""

        def commit() -> None:
            self.commit_transition_locked(
                workflow_id,
                records=self.records_locked(*touched),
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

        self._write_locked(commit, lambda held: held.task_ids.extend(touched))
        self.commit_locked(*(task_id for task_id in returned if task_id not in touched))

    def set_completion_notifier(self, notify: Callable[[str], None]) -> None:
        """Install the callback that a terminal transition notifies."""
        self.on_workflow_settled = notify

    def drop_unacknowledged(self, task_id: str) -> None:
        del self.unacknowledged[task_id]
