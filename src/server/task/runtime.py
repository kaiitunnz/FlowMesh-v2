import copy
import heapq
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from shared.harness import (
    AgentEpisodeDispatch,
    BoundaryEventKind,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    InputBinding,
    InputBindingMember,
)
from shared.schemas.command import InterruptMessage
from shared.schemas.result import ResultEnvelope, result_file_path
from shared.tasks import TaskEnvelopeTemplate
from shared.utils import new_workflow_id
from shared.utils.ids import new_model_secret_ref

from ..config import AgentBindingConfig, OrchestrationConfig
from ..hooks import SUPPLIER_RESOLVERS
from ..orchestration import (
    AcceptedInput,
    AcceptedInputMember,
    Advance,
    LedgerSnapshot,
    OrchestrationEngine,
    PublicationOutcome,
    RecoveryDisposition,
    RegionError,
    ResultPublication,
    ScopeBudget,
    ValueRef,
    WorkItemStatus,
)
from ..orchestration.episode import BoundaryEvent
from ..orchestration.harness import to_boundary_event
from ..orchestration.tool_dispatch import (
    MODEL_INTERFACE,
    SEARCH_INTERFACE,
    FacadeTurnGroup,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from ..registries.worker import Worker, WorkerRegistry
from ..registries.workflow import PersistedTask, WorkflowRegistry, WorkflowSched
from ..services.model_secret_vault import ModelSecretVault
from ..utils.time import parse_iso_ts
from .models import (
    TERMINAL_TASK_STATUSES,
    TaskInfo,
    TaskParsingResult,
    TaskRecord,
    TaskStatus,
    TaskUsage,
    categorize_task_type,
)
from .parser import ParsedWorkflow, parse_workflow
from .v2 import (
    ExecutionMode,
    FrontendWorkflowSource,
    InspectionReport,
    LoweringStrategy,
    PersistedV2Workflow,
    build_inspection,
    compile_bundle,
)
from .v2.compiler.agent_binding import AgentBindingDefaults
from .v2.credentials import pop_inline_model_secrets, redact_source_text
from .v2.representations.operators import AgentModelGatewayBinding, FacadeDescriptor
from .v2.representations.plan import EpisodeSpec

# A live-feasibility check: whether a lowered episode's declared alternative can be
# placed now.
EpisodeFeasibility = Callable[[EpisodeSpec], bool]


def _stringify(value: Any) -> str:
    """A stable string rendering of a resolved input value, for delivery."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _binding_defaults(cfg: AgentBindingConfig) -> AgentBindingDefaults:
    """Convert the deployment binding config into the compiler's injected defaults."""
    return AgentBindingDefaults(
        default_backend=cfg.default_backend,
        default_version=cfg.default_version,
        default_mode=cfg.default_mode,
        default_url=cfg.default_url,
        default_model=cfg.default_model,
    )


def _sanitize_merge_spec(spec: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(spec)
    if isinstance(clone.get("inference"), dict):
        inference_cfg = clone["inference"]
        inference_cfg.pop("system_prompt", None)
    clone.pop("data", None)
    return clone


def _compute_merge_key(task: TaskEnvelopeTemplate) -> str | None:
    task_type = str(task.spec.taskType or "").strip().lower()
    if task_type not in {"inference", "rag", "diffusion"}:
        return None
    try:
        spec = task.spec.model_dump(mode="python", exclude_none=True)
        sanitized = _sanitize_merge_spec(spec)
        return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    except Exception:
        return None


class TaskRuntime:
    """In-memory task registry with FIFO-ready queue and dependency tracking."""

    def __init__(
        self,
        workflow_registry: WorkflowRegistry,
        worker_registry: WorkerRegistry,
        orchestration: OrchestrationConfig,
        results_dir: Path,
        logger: logging.Logger,
        secret_vault: ModelSecretVault,
        feasibility_check: EpisodeFeasibility | None = None,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._worker_registry = worker_registry
        self._logger = logger
        self._results_dir = results_dir
        self._feasibility_check = feasibility_check
        self._secret_vault = secret_vault
        self._scope_budget = ScopeBudget.from_config(orchestration)
        self._web_search = orchestration.web_search
        self._input_budget_bytes = orchestration.agent_input_budget_bytes
        self._agent_binding_defaults = _binding_defaults(orchestration.agent_binding)
        self._lowering_strategy = (
            LoweringStrategy.EPISODE_CUT
            if orchestration.episode_lowering
            else LoweringStrategy.TRANSPARENT
        )
        self._tasks: dict[str, TaskRecord] = {}
        self._original_deps: dict[str, set[str]] = {}
        self._pending_deps: dict[str, set[str]] = {}
        self._dependents: dict[str, set[str]] = defaultdict(set)
        self._ready_by_workflow: dict[str, list[tuple[int, str]]] = {}
        self._ready_queue: deque[tuple[str, bool]] = (
            deque()
        )  # task_id | workflow_id, is_workflow
        self._ready_index: set[str] = set()
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._merge_key_by_task: dict[str, tuple[str | None, str | None]] = {}
        self._merge_buckets: dict[tuple[str, str | None], list[str]] = defaultdict(list)
        self._merge_children_map: dict[str, list[str]] = defaultdict(list)
        self._merge_parent_map: dict[str, str] = {}
        self._workflow_epoch_tasks: dict[str, deque[set[str]]] = {}
        self._workflow_epoch_frontier: dict[str, int] = {}
        self._workflow_in_epoch_order: dict[str, bool] = {}
        self._task_epoch_index: dict[str, int] = {}
        self._rehydrated_dispatched: dict[str, float] = {}
        self._engines: dict[str, OrchestrationEngine] = {}
        self._retired_region_templates: dict[str, set[str]] = {}
        # Interface-keyed handlers for a mediated boundary, dispatched off the caller's
        # lane. The model gateway settles the "model" interface; the fabric tool broker
        # executes a fabric-served tool ("search/v1"). Both take the durable envelope.
        self._model_settler: Callable[[ToolInvocationEnvelope], None] | None = None
        self._tool_broker: Callable[[ToolInvocationEnvelope], None] | None = None
        self._resident_terminal_hook: Callable[[str, bool], None] | None = None
        # Facade boundaries the agent-model gateway captured server-side during an
        # episode's model turn, keyed by task; the completion path reroutes the clean
        # turn-completion into the pending boundary rather than settling it.
        self._pending_facade_groups: dict[str, FacadeTurnGroup] = {}

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

    # ------------------------------------------------------------------ #
    # Registration & submission
    # ------------------------------------------------------------------ #

    def validate(self, payload: str, format: str = "native") -> list[TaskParsingResult]:
        parsed_workflow = parse_workflow(payload, format)
        specs = parsed_workflow.tasks
        results: list[TaskParsingResult] = []
        for entry in specs:
            task_id = entry.task_id
            depends_on = entry.depends_on.copy()
            results.append(
                TaskParsingResult(
                    task_id=task_id,
                    graph_node_name=entry.graph_node_name,
                    depends_on=depends_on,
                )
            )
        return results

    def inspect_v2(
        self, payload: str, format: str = "native"
    ) -> InspectionReport | None:
        """Compile a v2 submission into an inspection report without executing.

        Returns ``None`` for a non-v2 submission. Structural frontend errors raise
        ``CompileError``; semantic findings ride on the report's diagnostics.
        """
        parsed_workflow = parse_workflow(payload, format)
        if not ExecutionMode.is_v2(parsed_workflow.api_version):
            return None
        # A dry run never vaults; drop any inline credential and redact the source so
        # the inspection echoes no raw key back to the caller.
        pop_inline_model_secrets(parsed_workflow)
        source = FrontendWorkflowSource.capture(
            redact_source_text(payload, format), format
        )
        return build_inspection(
            new_workflow_id(),
            parsed_workflow,
            source,
            bindings=self._agent_binding_defaults,
        )

    async def _vault_inline_secrets(
        self, workflow_id: str, parsed: ParsedWorkflow
    ) -> dict[str, str]:
        """Vault each agent's inline model credential, returning its generated ref.

        The credential is stripped from the parsed spec and stored under the workflow,
        so only the opaque ref reaches the compiled binding, the persisted record, and
        every downstream surface.
        """
        secrets = pop_inline_model_secrets(parsed)
        if not secrets:
            return {}
        secret_refs: dict[str, str] = {}
        for task_id, secret in secrets.items():
            ref = new_model_secret_ref()
            await self._secret_vault.store(workflow_id, ref, secret)
            secret_refs[task_id] = ref
        return secret_refs

    async def register(
        self, owner_id: str, org_id: str, payload: str, format: str = "native"
    ) -> tuple[str, list[TaskParsingResult]]:
        parsed_workflow = parse_workflow(payload, format)
        specs = parsed_workflow.tasks
        yaml_text = redact_source_text(payload, format)
        results: list[TaskParsingResult] = []
        workflow_id = new_workflow_id()
        task_records: list[TaskRecord] = []
        candidate_ready: list[str] = []
        graph_task_ids: dict[str, str] = {}

        v2_bundle: PersistedV2Workflow | None = None
        v2_engine: OrchestrationEngine | None = None
        if ExecutionMode.is_v2(parsed_workflow.api_version):
            secret_refs = await self._vault_inline_secrets(workflow_id, parsed_workflow)
            source = FrontendWorkflowSource.capture(yaml_text, format)
            v2_bundle = compile_bundle(
                workflow_id,
                parsed_workflow,
                source,
                strategy=self._lowering_strategy,
                bindings=self._agent_binding_defaults,
                secret_refs=secret_refs,
            )
            v2_engine = OrchestrationEngine.build(
                workflow_id, owner_id, org_id, v2_bundle, budget=self._scope_budget
            )

        with self._cv:
            if (
                parsed_workflow.schedule_in_epoch_order
                and parsed_workflow.epoch_groups is not None
            ):
                self._ready_by_workflow[workflow_id] = []
                self._workflow_in_epoch_order[workflow_id] = True
            for entry in specs:
                task_id = entry.task_id
                task = entry.task.model_copy(deep=True)
                depends_on = entry.depends_on.copy()
                original = set(depends_on)
                pending = {dep for dep in depends_on if dep not in self._completed}

                task_type = task.spec.taskType
                category = categorize_task_type(task_type)

                selected_worker_raw = entry.selected_worker
                selected_worker: list[str] | None
                if isinstance(selected_worker_raw, list):
                    normalized_workers = [
                        str(worker_id).strip()
                        for worker_id in selected_worker_raw
                        if str(worker_id).strip()
                    ]
                    selected_worker = list(dict.fromkeys(normalized_workers)) or None
                elif isinstance(selected_worker_raw, str):
                    selected_worker = (
                        [selected_worker_raw.strip()]
                        if selected_worker_raw.strip()
                        else None
                    )
                else:
                    selected_worker = None

                record = TaskRecord(
                    task_id=task_id,
                    workflow_id=workflow_id,
                    owner_id=owner_id,
                    raw_yaml=yaml_text,
                    task=task,
                    local_name=entry.local_name,
                    graph_node_name=entry.graph_node_name,
                    load=entry.load,
                    position_in_epoch=entry.position_in_epoch,
                    selected_worker=selected_worker,
                    task_type=task_type,
                    category=category,
                )
                task_records.append(record)
                record.last_queue_ts = record.submitted_ts
                if v2_engine is None:
                    merge_key = _compute_merge_key(task)
                    record.merge_key = merge_key
                    selected_worker_hint = (
                        record.selected_worker[0]
                        if record.selected_worker and len(record.selected_worker) == 1
                        else None
                    )
                    self._merge_key_by_task[task_id] = (merge_key, selected_worker_hint)

                self._tasks[task_id] = record
                if record.graph_node_name:
                    graph_task_ids[record.graph_node_name] = task_id
                self._original_deps[task_id] = original
                self._failed.discard(task_id)
                if record.status == TaskStatus.DONE:
                    self._completed.add(task_id)
                else:
                    self._completed.discard(task_id)

                # v2 readiness is owned by the orchestration engine; the legacy
                # dependency machinery stays unwired so it cannot admit v2 work.
                if v2_engine is None:
                    self._pending_deps[task_id] = pending
                    for dep in original:
                        self._dependents[dep].add(task_id)
                    if not pending and record.status == TaskStatus.PENDING:
                        candidate_ready.append(task_id)

                results.append(
                    TaskParsingResult(
                        task_id=task_id,
                        graph_node_name=entry.graph_node_name,
                        depends_on=depends_on,
                    )
                )
            epoch_groups = parsed_workflow.epoch_groups
            if epoch_groups and v2_engine is None:
                epoch_queue: deque[set[str]] = deque()
                has_epoch_tasks = False
                for epoch_idx, epoch_nodes in enumerate(epoch_groups):
                    epoch_task_ids: set[str] = set()
                    for node_name in epoch_nodes:
                        mapped_task_id = graph_task_ids.get(node_name)
                        if mapped_task_id is None:
                            continue
                        epoch_task_ids.add(mapped_task_id)
                        self._task_epoch_index[mapped_task_id] = epoch_idx
                        has_epoch_tasks = True
                    epoch_queue.append(epoch_task_ids)
                if has_epoch_tasks:
                    self._workflow_epoch_tasks[workflow_id] = epoch_queue
                    self._workflow_epoch_frontier[workflow_id] = 0

        await self._workflow_registry.register_workflow_async(
            workflow_id, task_records, v2=v2_bundle
        )

        with self._cv:
            persisted = [
                item
                for record in task_records
                if (item := self._persisted_task_locked(record.task_id))
            ]
            in_epoch_order = self._workflow_in_epoch_order.get(workflow_id, False)
            frontier = self._workflow_epoch_frontier.get(workflow_id, 0)
        await self._workflow_registry.save_task_states_async(persisted)
        await self._workflow_registry.save_workflow_sched_async(
            workflow_id, in_epoch_order, frontier
        )

        new_ready = False
        with self._cv:
            if v2_engine is not None:
                self._engines[workflow_id] = v2_engine
                if self._apply_advance_locked(workflow_id, v2_engine.initial_advance()):
                    new_ready = True
            for task_id in candidate_ready:
                maybe_record = self._tasks.get(task_id)
                if not maybe_record or maybe_record.status != TaskStatus.PENDING:
                    continue
                if self._pending_deps.get(task_id):
                    continue
                if self._enqueue_ready_locked(task_id):
                    new_ready = True
            if new_ready:
                self._cv.notify_all()

        # Snapshot last, after the initial advance has persisted any authority-denied
        # roots, so the ledger never leads durable task state.
        if v2_engine is not None:
            await self._workflow_registry.save_ledger_snapshot_async(
                workflow_id, v2_engine.to_snapshot()
            )

        return workflow_id, results

    # ------------------------------------------------------------------ #
    # Rehydration
    # ------------------------------------------------------------------ #

    async def rehydrate(self) -> int:
        """Rebuild in-memory scheduler state from durable Redis records.

        Reconstructs every live workflow's DAG, ready queue, and epoch state from the
        persisted per-task snapshots. In-flight (DISPATCHED / CANCELLING) tasks are left
        assigned to their worker: completions that landed during the restart arrive via
        the replayed task-event stream, and genuinely departed workers are recovered by
        the watchdog. Returns the number of workflows restored.
        """
        workflow_ids = await self._workflow_registry.get_workflow_ids_async()
        rehydrated_at = time.time()
        restored = 0
        for workflow_id in sorted(workflow_ids):
            wf_record = await self._workflow_registry.get_workflow_record_async(
                workflow_id
            )
            if wf_record is None:
                continue
            dynamic_ids = await self._workflow_registry.get_dynamic_task_ids_async(
                workflow_id
            )
            task_ids = list(dict.fromkeys([*wf_record.task_ids, *sorted(dynamic_ids)]))
            tasks: list[PersistedTask] = [
                state
                for state in await self._workflow_registry.load_task_states_async(
                    *task_ids
                )
                if state
            ]
            if not tasks:
                continue
            sched = await self._workflow_registry.load_workflow_sched_async(workflow_id)
            snapshot = await self._workflow_registry.load_ledger_snapshot_async(
                workflow_id
            )
            bundle = (
                await self._workflow_registry.get_v2_workflow_async(workflow_id)
                if snapshot is not None
                else None
            )
            with self._cv:
                if snapshot is not None and bundle is not None:
                    self._install_rehydrated_v2_workflow_locked(
                        workflow_id, tasks, snapshot, bundle, rehydrated_at
                    )
                else:
                    self._install_rehydrated_workflow_locked(
                        workflow_id, tasks, sched, rehydrated_at
                    )
                self._cv.notify_all()
            restored += 1
        if restored:
            self._logger.info("Rehydrated %d workflow(s) from durable state", restored)
        return restored

    def _install_rehydrated_workflow_locked(
        self,
        workflow_id: str,
        tasks: list[PersistedTask],
        sched: WorkflowSched | None,
        rehydrated_at: float,
    ) -> None:
        terminal = TERMINAL_TASK_STATUSES
        in_epoch_order = sched.in_epoch_order if sched else False
        frontier = sched.epoch_frontier if sched else 0
        epoch_members: dict[int, set[str]] = defaultdict(set)

        for persisted in tasks:
            record = persisted.record
            task_id = record.task_id
            self._tasks[task_id] = record
            self._original_deps[task_id] = set(persisted.depends_on)
            selected_worker_hint = (
                record.selected_worker[0]
                if record.selected_worker and len(record.selected_worker) == 1
                else None
            )
            self._merge_key_by_task[task_id] = (record.merge_key, selected_worker_hint)
            if persisted.epoch_index is not None:
                self._task_epoch_index[task_id] = persisted.epoch_index
                epoch_members[persisted.epoch_index].add(task_id)
            if record.status == TaskStatus.DONE:
                self._completed.add(task_id)
            elif record.status == TaskStatus.FAILED:
                self._failed.add(task_id)
            elif record.status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                self._rehydrated_dispatched[task_id] = rehydrated_at

        for persisted in tasks:
            record = persisted.record
            task_id = record.task_id
            original = self._original_deps.get(task_id) or set()
            for dep in original:
                self._dependents[dep].add(task_id)
            if record.status in terminal:
                continue
            # Only completed deps are subtracted, not failed ones: a failure
            # cascade-fails its dependents and persists them FAILED atomically,
            # so a non-terminal task here never has a FAILED dep to clear.
            self._pending_deps[task_id] = {
                dep for dep in original if dep not in self._completed
            }

        if in_epoch_order:
            self._workflow_in_epoch_order[workflow_id] = True
            self._ready_by_workflow.setdefault(workflow_id, [])
        if epoch_members:
            epoch_queue: deque[set[str]] = deque(
                epoch_members[idx] for idx in sorted(epoch_members) if idx >= frontier
            )
            if epoch_queue:
                self._workflow_epoch_tasks[workflow_id] = epoch_queue
                self._workflow_epoch_frontier[workflow_id] = frontier

        for persisted in tasks:
            record = persisted.record
            if record.status != TaskStatus.PENDING:
                continue
            if self._pending_deps.get(record.task_id):
                continue
            self._enqueue_ready_locked(record.task_id)

    def _install_rehydrated_v2_workflow_locked(
        self,
        workflow_id: str,
        tasks: list[PersistedTask],
        snapshot: LedgerSnapshot,
        bundle: PersistedV2Workflow,
        rehydrated_at: float,
    ) -> None:
        """Rebuild a v2 workflow: restore the engine and re-admit ready work items.

        The legacy dependency machinery stays unwired; the orchestration engine is the
        readiness authority. Terminal task facts reconcile the engine idempotently, so a
        crash between a task's terminal write and its ledger snapshot never loses a
        settlement and never duplicates a publication or effect receipt.
        """
        for persisted in tasks:
            record = persisted.record
            task_id = record.task_id
            self._tasks[task_id] = record
            self._original_deps[task_id] = set(persisted.depends_on)
            if record.status == TaskStatus.DONE:
                self._completed.add(task_id)
            elif record.status == TaskStatus.FAILED:
                self._failed.add(task_id)
            elif record.status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                self._rehydrated_dispatched[task_id] = rehydrated_at

        engine = OrchestrationEngine(snapshot, bundle, budget=self._scope_budget)
        self._engines[workflow_id] = engine
        cancelled = False
        for persisted in tasks:
            record = persisted.record
            if record.status == TaskStatus.DONE:
                engine.on_succeeded(record.task_id)
            elif record.status == TaskStatus.FAILED:
                engine.on_failed(
                    record.task_id, record.error or "task failed", retryable=False
                )
            elif record.status == TaskStatus.CANCELLED:
                cancelled = True
        # Replay cancellation after the settled facts so a settled outcome is never
        # overwritten; a cancelled workflow is then never re-admitted below.
        if cancelled:
            engine.cancel_instance()

        # Re-derive readiness for every PENDING task from the engine rather than
        # trusting the cached work-item status: a crash mid-retry can leave a task
        # PENDING while the snapshot still shows its work item in flight, and only a
        # re-derivation re-admits it instead of orphaning the workflow.
        for persisted in tasks:
            record = persisted.record
            if record.status == TaskStatus.PENDING and engine.reconcile_pending(
                record.task_id
            ):
                self._enqueue_ready_locked(record.task_id)

        # Re-drive fan-out for any DONE producer whose spawn never sealed before a
        # crash: its terminal event will not replay, so nothing else materializes the
        # children. Re-driving is idempotent once the spawn has sealed.
        for persisted in tasks:
            if persisted.record.status != TaskStatus.DONE:
                continue
            advance = self._fan_out_children_locked(
                workflow_id, engine, persisted.record.task_id
            )
            self._apply_advance_locked(workflow_id, advance)

        # Re-issue the off-lane dispatch for any mediated boundary suspended with no
        # durable outcome: the handler ran on an in-memory executor a crash discarded,
        # so nothing else resumes the agent. The durable envelope routes it to the same
        # handler (a search to the broker, a model to the gateway) it was recorded for.
        for envelope in engine.pending_tool_dispatches():
            self._dispatch_boundary(envelope)
        self._save_ledger_locked(workflow_id)

    # ------------------------------------------------------------------ #
    # Durable state persistence
    # ------------------------------------------------------------------ #

    def _persisted_task_locked(self, task_id: str) -> PersistedTask | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        return PersistedTask(
            record=record,
            depends_on=self._original_deps.get(task_id) or set(),
            epoch_index=self._task_epoch_index.get(task_id),
        )

    def _records_locked(self, *task_ids: str) -> list[PersistedTask]:
        return [
            persisted
            for task_id in dict.fromkeys(task_ids)
            if (persisted := self._persisted_task_locked(task_id))
        ]

    def _sched_locked(self, workflow_id: str) -> WorkflowSched:
        return WorkflowSched(
            in_epoch_order=self._workflow_in_epoch_order.get(workflow_id, False),
            epoch_frontier=self._workflow_epoch_frontier.get(workflow_id, 0),
        )

    def _persist_locked(self, *task_ids: str) -> None:
        """Commit task records (no membership change) atomically, per workflow."""
        by_workflow: dict[str, list[str]] = defaultdict(list)
        for task_id in dict.fromkeys(task_ids):
            if record := self._tasks.get(task_id):
                by_workflow[record.workflow_id].append(task_id)
        for workflow_id, ids in by_workflow.items():
            self._workflow_registry.commit_transition(
                workflow_id, records=self._records_locked(*ids)
            )

    def _persist_terminal_locked(self, *task_ids: str, sched: bool = True) -> None:
        """Commit each task's final state — its record and its done/failed/cancelled
        set membership (by current status) — and the workflow schedule, as one atomic
        transaction per workflow and the single last step of a transition.

        Committing only after all in-memory mutations means a failed or crashed write
        can't leave durable state half-applied: the transaction commits in full or not
        at all. Event-driven callers additionally heal via the at-least-once replay
        (``_repersist_terminal_workflow_locked``); the API-driven cancel relies on this
        atomicity alone. Assumes the in-memory mutations never raise, which holds while
        ordered tasks carry ``position_in_epoch`` (so the ready-queue helpers never hit
        their guards).
        """
        moves: dict[str, tuple[list[str], list[str], list[str]]] = defaultdict(
            lambda: ([], [], [])
        )
        for task_id in dict.fromkeys(task_ids):
            record = self._tasks.get(task_id)
            if record is None:
                continue
            match record.status:
                case TaskStatus.DONE:
                    moves[record.workflow_id][0].append(task_id)
                case TaskStatus.FAILED:
                    moves[record.workflow_id][1].append(task_id)
                case TaskStatus.CANCELLED:
                    moves[record.workflow_id][2].append(task_id)
                case _:
                    self._logger.warning(
                        "Non-terminal task %s (%s) skipped in terminal persist",
                        task_id,
                        record.status,
                    )
        for workflow_id, (done, failed, cancelled) in moves.items():
            if ids := done + failed + cancelled:
                self._workflow_registry.commit_transition(
                    workflow_id,
                    records=self._records_locked(*ids),
                    done=done,
                    failed=failed,
                    cancelled=cancelled,
                    sched=self._sched_locked(workflow_id) if sched else None,
                )

    def _repersist_terminal_workflow_locked(self, workflow_id: str) -> None:
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
            self._persist_terminal_locked(*terminal_ids)
        else:
            self._workflow_registry.commit_transition(
                workflow_id, sched=self._sched_locked(workflow_id)
            )

    def _reclaim_vault_if_settled_locked(self, workflow_id: str) -> None:
        """Purge a workflow's vaulted credentials once its last task has settled.

        The primary reclaim on the terminal transition, for a workflow that completes or
        fails; the vault's sliding TTL only backstops a submission that never settles.
        Called after an event's advance materializes any new children, so a producer
        that fans out is not reclaimed while its children are still pending.
        """
        records = [r for r in self._tasks.values() if r.workflow_id == workflow_id]
        if records and all(r.status in TERMINAL_TASK_STATUSES for r in records):
            self._secret_vault.purge(workflow_id)

    # ------------------------------------------------------------------ #
    # Ready queue helpers
    # ------------------------------------------------------------------ #

    def _enqueue_ready_locked(self, task_id: str, *, front: bool = False) -> bool:
        """Add a task to the ready queue if it is pending and not already queued."""
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return False
        if task_id in self._ready_index:
            return False
        if not self._is_epoch_ready_locked(record):
            return False
        workflow_id = record.workflow_id
        if (
            workflow_id in self._workflow_in_epoch_order
            and task_id in self._task_epoch_index
        ):
            queue = self._ready_by_workflow[workflow_id]
            position_in_epoch = record.position_in_epoch
            if position_in_epoch is None:
                raise ValueError(
                    "Ordered workflow task is missing position_in_epoch "
                    f"(task_id={task_id})"
                )
            heapq.heappush(queue, (position_in_epoch, task_id))
            ready_entry = (workflow_id, True)
        else:
            ready_entry = (task_id, False)
        if front:
            self._ready_queue.appendleft(ready_entry)
        else:
            self._ready_queue.append(ready_entry)
        self._ready_index.add(task_id)
        record.last_queue_ts = time.time()
        self._merge_bucket_add(task_id)
        return True

    def _pop_ready_locked(self) -> str | None:
        while self._ready_queue:
            task_or_workflow_id, is_workflow = self._ready_queue.popleft()
            if is_workflow:
                _, task_id = heapq.heappop(self._ready_by_workflow[task_or_workflow_id])
            else:
                task_id = task_or_workflow_id
            self._ready_index.discard(task_id)
            record = self._tasks.get(task_id)
            if not record or record.status != TaskStatus.PENDING:
                continue
            return task_id
        return None

    def _remove_from_ready_locked(self, task_id: str) -> None:
        if task_id not in self._ready_index:
            return
        record = self._tasks.get(task_id)
        if not record:
            return
        workflow_id = record.workflow_id
        if (
            workflow_id in self._workflow_in_epoch_order
            and task_id in self._task_epoch_index
        ):
            queue = self._ready_by_workflow[workflow_id]
            position_in_epoch = record.position_in_epoch
            if position_in_epoch is None:
                raise ValueError(
                    "Ordered workflow task is missing position_in_epoch "
                    f"(task_id={task_id})"
                )
            queue.remove((position_in_epoch, task_id))
            heapq.heapify(queue)
            ready_entry = (workflow_id, True)
        else:
            ready_entry = (task_id, False)
        self._ready_queue.remove(ready_entry)
        self._ready_index.discard(task_id)

    def _merge_bucket_add(self, task_id: str) -> None:
        key = self._merge_key_by_task.get(task_id)
        if not key:
            return
        merge_key, selected_worker = key
        if not merge_key:
            return
        bucket = self._merge_buckets.setdefault((merge_key, selected_worker), [])
        if task_id not in bucket:
            bucket.append(task_id)

    def _merge_bucket_remove(self, task_id: str) -> None:
        key = self._merge_key_by_task.get(task_id)
        if not key:
            return
        merge_key, selected_worker = key
        if not merge_key:
            return
        bucket = self._merge_buckets.get((merge_key, selected_worker))
        if not bucket:
            return
        try:
            bucket.remove(task_id)
        except ValueError:
            pass
        if not bucket:
            self._merge_buckets.pop((merge_key, selected_worker), None)

    def _is_epoch_ready_locked(self, record: TaskRecord) -> bool:
        epoch_index = self._task_epoch_index.get(record.task_id)
        if epoch_index is None:
            return True
        frontier = self._workflow_epoch_frontier.get(record.workflow_id)
        if frontier is None:
            return True
        return epoch_index == frontier

    def _try_advance_epoch_frontier_locked(self, workflow_id: str) -> list[str]:
        epoch_tasks = self._workflow_epoch_tasks.get(workflow_id)
        if not epoch_tasks:
            return []
        frontier = self._workflow_epoch_frontier[workflow_id]

        ready: list[str] = []
        while True:
            self._workflow_epoch_frontier[workflow_id] = frontier
            current_tasks = epoch_tasks[0] if epoch_tasks else set()
            if current_tasks and not all(
                (task := self._tasks.get(task_id)) is not None
                and task.status == TaskStatus.DONE
                for task_id in current_tasks
            ):
                break

            if epoch_tasks:
                epoch_tasks.popleft()
            frontier += 1
            self._workflow_epoch_frontier[workflow_id] = frontier
            if not epoch_tasks:
                self._workflow_epoch_frontier.pop(workflow_id, None)
                self._workflow_epoch_tasks.pop(workflow_id, None)
                break

            for task_id in epoch_tasks[0]:
                record = self._tasks.get(task_id)
                if not record or record.status != TaskStatus.PENDING:
                    continue
                if self._pending_deps.get(task_id):
                    continue
                if self._enqueue_ready_locked(task_id):
                    ready.append(task_id)

        return ready

    def _fail_later_epochs_locked(
        self,
        workflow_id: str,
        failed_epoch: int,
        reason: str,
    ) -> list[tuple[str, str]]:
        epoch_tasks = self._workflow_epoch_tasks.get(workflow_id)
        if not epoch_tasks:
            return []
        frontier = self._workflow_epoch_frontier[workflow_id]

        impacted: list[tuple[str, str]] = []
        for offset, epoch_task_ids in enumerate(epoch_tasks):
            epoch = frontier + offset
            if epoch <= failed_epoch:
                continue
            for task_id in epoch_task_ids:
                record = self._tasks.get(task_id)
                if not record or record.status != TaskStatus.PENDING:
                    continue
                record.status = TaskStatus.FAILED
                record.error = reason
                record.assigned_worker = None
                record.finished_ts = time.time()
                self._failed.add(task_id)
                self._completed.discard(task_id)
                self._pending_deps.pop(task_id, None)
                self._remove_from_ready_locked(task_id)
                self._merge_bucket_remove(task_id)
                self._merge_key_by_task.pop(task_id, None)
                self._merge_parent_map.pop(task_id, None)
                self._merge_children_map.pop(task_id, None)
                impacted.append((task_id, reason))

        return impacted

    def next_ready(
        self, stop_event: threading.Event, timeout: float = 1.0
    ) -> str | None:
        """
        Block until a task is ready or stop_event is set. Returns a task_id or None when
        stopping.
        """
        with self._cv:
            while not stop_event.is_set():
                task_id = self._pop_ready_locked()
                if task_id:
                    return task_id
                self._cv.wait(timeout)
            return None

    def mark_pending(self, task_id: str, *, increment_retry: bool = False) -> None:
        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            if record.status == TaskStatus.CANCELLED:
                return
            record.status = TaskStatus.PENDING
            record.assigned_worker = None
            record.topic = None
            record.dispatched_ts = None
            record.started_ts = None
            record.finished_ts = None
            record.error = None
            if increment_retry:
                try:
                    if record.max_attempts is not None and record.max_attempts >= 0:
                        record.attempts = min(record.attempts + 1, record.max_attempts)
                    else:
                        record.attempts = record.attempts + 1
                except Exception:
                    record.attempts = (record.attempts or 0) + 1
            self._workflow_registry.commit_transition(
                record.workflow_id,
                records=self._records_locked(task_id),
                pending=[task_id],
            )
            if increment_retry and (engine := self._engines.get(record.workflow_id)):
                # A retry reuses the work item and its invocation; the engine records
                # the failed attempt and readies the work item for a fresh one.
                engine.on_failed(
                    task_id, record.last_error or "task failed", retryable=True
                )
                self._save_ledger_locked(record.workflow_id)

    def requeue(self, task_id: str, *, front: bool = False) -> bool:
        """Reinsert a task into the ready queue."""
        with self._cv:
            added = self._enqueue_ready_locked(task_id, front=front)
            if added:
                self._cv.notify_all()
            return added

    # ------------------------------------------------------------------ #
    # v2 orchestration ledger (`DS`)
    # ------------------------------------------------------------------ #

    def is_v2_workflow(self, workflow_id: str) -> bool:
        with self._lock:
            return workflow_id in self._engines

    def orchestration_engine(self, workflow_id: str) -> OrchestrationEngine | None:
        with self._lock:
            return self._engines.get(workflow_id)

    def apply_boundary_event(self, task_id: str, event: BoundaryEvent) -> bool:
        """Carry an episode's boundary event into the ledger and dispatch its effect.

        Routes the event into the engine, synthesizes a task record for any dispatchable
        child it materializes, applies the advance, and writes the ledger snapshot after
        the task records so the ledger never leads durable state.
        """
        with self._cv:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return False
            advance = engine.route_boundary_event(task_id, event)
            self._synthesize_ready_children_locked(record.workflow_id, engine, advance)
            changed = self._apply_advance_locked(record.workflow_id, advance)
            self._save_ledger_locked(record.workflow_id)
            if changed:
                self._cv.notify_all()
            return changed

    def _apply_episode_step_locked(self, task_id: str, hr: HarnessResult) -> None:
        """Route one non-terminal agent-episode step and re-dispatch or suspend.

        A failure/cancellation settles the agent terminally. A boundary routes into the
        ledger (validated, recorded, capsule persisted); a spawn/seal/state-access/yield
        continues immediately with the ledger-recorded outcome, a denial re-readies with
        its typed outcome, and a model or effect boundary suspends until its durable
        outcome is settled. The record never goes DONE for a non-completion step.
        """
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if record is None or engine is None:
            return
        if hr.kind in (HarnessResultKind.FAILURE, HarnessResultKind.CANCELLATION):
            self._pending_facade_groups.pop(task_id, None)
            record.pending_facade_group = None
            reason = hr.error or (
                "agent episode cancelled"
                if hr.kind is HarnessResultKind.CANCELLATION
                else "agent episode failed"
            )
            if self._apply_advance_locked(
                record.workflow_id, engine.on_failed(task_id, reason, retryable=False)
            ):
                self._cv.notify_all()
            self._save_ledger_locked(record.workflow_id)
            return
        request = hr.request
        if request is None:
            return
        capsule = hr.capsule.blob if hr.capsule is not None else None
        wi = engine.work_item(task_id)
        if wi is not None and capsule is not None:
            wi.continuation_ref = capsule
        # The outcome that drove this step was consumed by its dispatch; clear it so a
        # later step never re-injects it.
        engine.mark_pending_outcome(task_id, None)
        event = to_boundary_event(request, continuation=capsule)
        advance = engine.route_boundary_event(task_id, event)
        self._synthesize_ready_children_locked(record.workflow_id, engine, advance)
        changed = self._apply_advance_locked(record.workflow_id, advance)
        corr = request.call_correlation
        env = (
            engine.boundary_envelope(wi.activation_id, corr)
            if wi is not None and corr is not None
            else None
        )
        handled = False
        if corr is not None and env is not None and env.denial is not None:
            engine.mark_pending_outcome(task_id, corr)
            engine.deliver_boundary_outcome(task_id, corr)
            self._reenqueue_episode_locked(task_id)
            changed = handled = True
        elif request.kind in (BoundaryEventKind.SPAWN, BoundaryEventKind.SPAWN_SEAL):
            if corr is not None:
                engine.mark_pending_outcome(task_id, corr)
            self._reenqueue_episode_locked(task_id)
            changed = handled = True
        elif request.kind in (BoundaryEventKind.STATE_ACCESS, BoundaryEventKind.YIELD):
            self._reenqueue_episode_locked(task_id)
            changed = handled = True
        elif corr is not None and request.kind in (
            BoundaryEventKind.INVOCATION,
            BoundaryEventKind.EXTERNAL_EFFECT,
        ):
            # A model or fabric-tool boundary suspends until its handler settles it
            # off-lane; the durable envelope routes it by exact (kind, interface).
            envelope = engine.tool_dispatch_envelope(task_id, corr)
            if envelope is not None:
                self._dispatch_boundary(envelope)
                handled = True
        if not handled:
            # A blocked boundary no branch re-readied or handed off (a denial lacking a
            # correlation) must not hang the episode; re-ready it so it can continue.
            stuck = engine.work_item(task_id)
            if stuck is not None and stuck.status is WorkItemStatus.BLOCKED:
                self._reenqueue_episode_locked(task_id)
                changed = True
        self._save_ledger_locked(record.workflow_id)
        if changed:
            self._cv.notify_all()

    def _route_and_dispatch_facade_group_locked(
        self,
        task_id: str,
        group: FacadeTurnGroup,
        capsule: HarnessCapsule | None,
    ) -> None:
        """Route a facade group into the ledger and dispatch its search members.

        The group's single continuation is the clean turn's capsule. Its spawn members
        materialize one child each (whose dispatchable records are synthesized here) and
        settle at admission; its search members up to the per-turn parallel cap dispatch
        concurrently to the broker, any beyond it settling as typed quota outcomes in
        source order — never a 500, never truncation. A spawn-only group re-readies the
        lead at once; a group with a search holds the lane until every await member is
        durably resolved.
        """
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if record is None or engine is None:
            return
        wi = engine.work_item(task_id)
        if wi is not None and capsule is not None:
            wi.continuation_ref = capsule.blob
        engine.mark_pending_outcome(task_id, None)
        advance = engine.route_facade_turn_group(task_id, group)
        self._synthesize_ready_children_locked(record.workflow_id, engine, advance)
        self._apply_advance_locked(record.workflow_id, advance)
        cap = self._web_search.max_parallel
        for index, envelope in enumerate(
            engine.group_dispatch_envelopes(task_id, group.group_id)
        ):
            if index < cap:
                self._dispatch_boundary(envelope)
            else:
                overflow = ToolOutcome(
                    status=ToolOutcomeStatus.QUOTA,
                    value=f"web search parallel cap ({cap}) exceeded this turn",
                )
                self.settle_episode_invocation(
                    task_id, envelope.call_correlation, overflow.model_dump_json()
                )
        # A spawn-only group settled at admission (the lane never suspended): re-enqueue
        # the lead now, closing this turn's attempt, so its next step injects the ack
        # vector. A group holding a search stays suspended until its members settle.
        stuck = engine.work_item(task_id)
        if stuck is not None and stuck.status is not WorkItemStatus.BLOCKED:
            self._reenqueue_episode_locked(task_id)
        else:
            # The lane stays suspended on a search: still persist the record so the
            # cleared pending_facade_group is durable. The ledger already holds this
            # group, so a crash before the first member settles must not leave the stale
            # capture on disk to hijack the lead's next completion on restart.
            self._persist_locked(task_id)
        self._save_ledger_locked(record.workflow_id)
        self._cv.notify_all()

    def _reenqueue_episode_locked(self, task_id: str) -> None:
        """Re-ready a still-running agent episode for its next run-to-yield step."""
        record = self._tasks.get(task_id)
        if record is None or record.status in TERMINAL_TASK_STATUSES:
            return
        # Settle the finished attempt so a long continuing episode's history stays
        # bounded (a no-op when a suspend already closed it).
        if engine := self._engines.get(record.workflow_id):
            engine.close_latest_attempt(task_id)
        record.status = TaskStatus.PENDING
        record.assigned_worker = None
        record.dispatched_ts = None
        record.started_ts = None
        self._enqueue_ready_locked(task_id, front=True)
        self._persist_locked(task_id)

    def settle_episode_invocation(
        self,
        task_id: str,
        call_correlation: str,
        value: str | None,
        *,
        error: str | None = None,
    ) -> bool:
        """Settle a suspended model/effect boundary and re-ready its episode.

        A value lands durably on the boundary envelope and the work item returns to
        READY for a re-dispatch that injects it at its originating call. An upstream
        ``error`` instead fails the boundary terminally, so a gateway failure never
        resumes the agent as a phantom empty success.
        """
        with self._cv:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return False
            if error is not None:
                advance = engine.on_failed(
                    task_id,
                    f"agent-model gateway upstream failed: {error}",
                    retryable=False,
                )
                invocation_id = engine.terminalize_boundary_invocation(
                    task_id, call_correlation
                )
                changed = self._apply_advance_locked(record.workflow_id, advance)
                self._save_ledger_locked(record.workflow_id)
                # A fenced failure terminal releases the resident credit just as a
                # completion does; nothing else may release an accepted credit.
                self._release_resident_credit(invocation_id, failed=True)
                if changed:
                    self._cv.notify_all()
                return changed
            advance = engine.settle_boundary_outcome(
                task_id, call_correlation, value=value
            )
            invocation_id = engine.terminalize_boundary_invocation(
                task_id, call_correlation
            )
            # The work item re-readied; the record must return to PENDING too, or the
            # dispatcher skips it (a ready task dispatches only from a pending record).
            if advance.ready:
                self._reenqueue_episode_locked(task_id)
            else:
                self._apply_advance_locked(record.workflow_id, advance)
            self._save_ledger_locked(record.workflow_id)
            self._release_resident_credit(invocation_id, failed=False)
            self._cv.notify_all()
            return True

    def _dispatch_boundary(self, env: ToolInvocationEnvelope) -> None:
        """Route a recorded mediated boundary to its handler by exact (kind, interface).

        Only ``(INVOCATION, "model")`` reaches the model gateway and only ``(INVOCATION,
        "search/v1")`` the fabric tool broker. An unrecognized interface is a fabric
        misconfiguration terminalized as a typed unavailable outcome — never a silent
        fall-through to the model settler.
        """
        if (
            env.kind is BoundaryEventKind.INVOCATION
            and env.interface == MODEL_INTERFACE
        ):
            if self._model_settler is not None:
                self._model_settler(env)
            return
        if (
            env.kind is BoundaryEventKind.INVOCATION
            and env.interface == SEARCH_INTERFACE
        ):
            if self._tool_broker is not None:
                self._tool_broker(env)
            return
        outcome = ToolOutcome(
            status=ToolOutcomeStatus.UNAVAILABLE,
            value=f"no fabric handler for interface {env.interface!r}",
        )
        self.settle_episode_invocation(
            env.task_id, env.call_correlation, outcome.model_dump_json()
        )

    def set_model_settler(
        self, settler: Callable[[ToolInvocationEnvelope], None]
    ) -> None:
        """Install the off-lane handler for the mediated ``model`` interface."""
        self._model_settler = settler

    def set_tool_broker(self, broker: Callable[[ToolInvocationEnvelope], None]) -> None:
        """Install the off-lane handler for fabric-served tool interfaces."""
        self._tool_broker = broker

    def set_resident_terminal_hook(self, hook: Callable[[str, bool], None]) -> None:
        """Install the consumer that releases a resident admission credit on DS
        terminal.

        The hook receives the settled boundary's ``invocation_id`` and whether the
        outcome was a failure, so the Admission controller advances the linked claim to
        terminal on any fenced outcome — the sole normal credit release.
        """
        self._resident_terminal_hook = hook

    def _release_resident_credit(
        self, invocation_id: str | None, *, failed: bool
    ) -> None:
        if invocation_id is not None and self._resident_terminal_hook is not None:
            self._resident_terminal_hook(invocation_id, failed)

    def originate_facade_turn_group(self, task_id: str, group: FacadeTurnGroup) -> None:
        """Record a turn-scoped facade group the gateway captured, before it acks.

        The agent-model gateway sees a model turn's native facade calls before the
        harness does and clean-completes the turn; the whole ordered membership and its
        single continuation are persisted on the task record before the gateway acks the
        clean turn, so the episode's next completion routes the group rather than
        settling the episode DONE, and a restart-replayed completion still routes it. At
        most one group is open per episode; a second capture while one holds the gate is
        refused by the gateway fence, not stored here.
        """
        with self._lock:
            self._pending_facade_groups[task_id] = group
            if (record := self._tasks.get(task_id)) is not None:
                record.pending_facade_group = group
                self._persist_locked(task_id)

    def has_pending_facade(self, task_id: str) -> bool:
        """Whether a facade group is already captured or still open for this episode.

        The gateway fence reads this to refuse a distinct second group before the open
        one's await-outcome members settle, so a group is never overwritten mid-flight.
        A spawn-only group holds nothing here once routed, so a later turn may issue it.
        """
        with self._lock:
            if task_id in self._pending_facade_groups:
                return True
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is not None and record.pending_facade_group is not None:
                return True
            if engine is None:
                return False
            return engine.has_open_facade_group(task_id)

    def resolve_model_binding(self, task_id: str) -> AgentModelGatewayBinding | None:
        """The pinned managed-model binding for a task's agent, for the gateway.

        Returns the effective binding frozen at submission so a mediated invocation
        resolves its upstream from the activation, never from the request body or a
        later environment change.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None:
                return None
            op = engine.agent_operator(task_id)
            return op.model_binding if op is not None else None

    def agent_facade_descriptors(self, task_id: str) -> list[FacadeDescriptor]:
        """The fabric facades pinned on a task's agent, for the gateway to inject.

        Only an agent's compile-pinned facades are injectable, so the model can never be
        offered a fabric tool the agent did not declare.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None:
                return []
            op = engine.agent_operator(task_id)
            return list(op.facades) if op is not None else []

    def gateway_binding_for(
        self, task_id: str
    ) -> tuple[str, AgentModelGatewayBinding] | None:
        """The task's owning workflow and its pinned model binding, for the gateway.

        The workflow id scopes the credential resolution so a vaulted ref yields a
        secret only within the workflow that minted it.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return None
            op = engine.agent_operator(task_id)
            if op is None or op.model_binding is None:
                return None
            return record.workflow_id, op.model_binding

    def agent_episode_dispatch(self, task_id: str) -> AgentEpisodeDispatch | None:
        """The agent-episode context to ship with a dispatch, or None for a non-agent.

        The backend key comes from the operator's pinned harness binding, so a later
        deployment-default change cannot move a live activation.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None:
                return None
            op = engine.agent_operator(task_id)
            harness = op.harness_binding if op is not None else None
            if harness is None:
                return None
            capsule_blob, outcomes = engine.episode_context(task_id)
            # First-turn dataflow inputs are delivered only on the first dispatch; a
            # resume injects only the harness's own delivered outcomes.
            input_bindings = (
                self._agent_input_bindings(engine, task_id)
                if capsule_blob is None
                else ()
            )
            return AgentEpisodeDispatch(
                backend=HarnessBackendKey(
                    backend=harness.backend, version=harness.version
                ),
                capsule_blob=capsule_blob,
                delivered_outcomes=outcomes,
                input_bindings=input_bindings,
            )

    def _synthesize_ready_children_locked(
        self, workflow_id: str, engine: OrchestrationEngine, advance: Advance
    ) -> None:
        """Give any newly ready child a dispatchable, durably persisted task record."""
        new_children: list[str] = []
        for child_task_id in advance.ready:
            if child_task_id in self._tasks:
                continue
            wi = engine.work_item(child_task_id)
            template = self._tasks.get(wi.operator_id) if wi else None
            if template is not None:
                self._register_child_locked(child_task_id, template, None)
                new_children.append(child_task_id)
        self._commit_new_children_locked(workflow_id, engine, new_children)

    def _register_child_locked(
        self, child_task_id: str, template: TaskRecord, element: Any
    ) -> None:
        """Install a self-contained task record for one materialized child."""
        self._tasks[child_task_id] = self._synthesize_child_record(
            template, child_task_id, element
        )
        self._original_deps[child_task_id] = set()

    def _commit_new_children_locked(
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
            self._workflow_registry.commit_dynamic_tasks(
                workflow_id,
                self._records_locked(*child_task_ids),
                engine.to_snapshot(),
                retire=retire,
            )

    def _retire_sealed_region_templates_locked(
        self, workflow_id: str, engine: OrchestrationEngine
    ) -> None:
        """Retire an agent-region child template once its spawn region has sealed.

        A dynamic spawn region's child body is a template, never dispatched as a task;
        once the region seals it no longer holds the workflow open, so it is dropped
        from the remaining set (idempotently, tracked per workflow) with the ledger.
        """
        already = self._retired_region_templates.setdefault(workflow_id, set())
        pending = engine.sealed_region_child_templates() - already
        if pending:
            already.update(pending)
            self._commit_new_children_locked(
                workflow_id, engine, [], retire=sorted(pending)
            )

    def episode_feasible(self, task_id: str) -> bool:
        """Whether a ready episode's declared alternative can be placed now.

        The generic live-feasibility handoff from the lowerer's episode annotation to
        the scheduler: an infeasible alternative is deferred by the dispatcher, holding
        no worker. Absent a configured check, or for a task the plan did not cut into an
        episode, placement is always feasible.
        """
        if self._feasibility_check is None:
            return True
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            spec = engine.episode_spec(task_id) if engine else None
        return True if spec is None else self._feasibility_check(spec)

    def retry_deferred_fanout(self, producer_task_id: str) -> None:
        """Re-drive a producer's deferred fan-out once its result lands out-of-band.

        A producer's terminal event is processed once; in a multi-node deployment its
        result reaches this node through result ingest, unordered against that event, so
        a fan-out deferred for a not-yet-readable result has no later event to re-drive
        it. Re-driving is idempotent: a sealed spawn admits no new child.
        """
        with self._cv:
            record = self._tasks.get(producer_task_id)
            if record is None or record.status != TaskStatus.DONE:
                return
            engine = self._engines.get(record.workflow_id)
            if engine is None:
                return
            advance = self._fan_out_children_locked(
                record.workflow_id, engine, producer_task_id
            )
            if self._apply_advance_locked(record.workflow_id, advance):
                self._cv.notify_all()

    def resolve_v2_output(
        self, workflow_id: str, output_id: str
    ) -> ResultPublication | None:
        """Resolve a declared logical output to its terminal publication."""
        with self._lock:
            engine = self._engines.get(workflow_id)
            return engine.resolve_output(output_id) if engine else None

    def resolve_v2_legacy_result(
        self, workflow_id: str, task_id: str
    ) -> ResultPublication | None:
        """Resolve a legacy task's induced output slot (compatibility adapter)."""
        with self._lock:
            engine = self._engines.get(workflow_id)
            return engine.resolve_legacy_task(task_id) if engine else None

    def recovery_disposition(
        self, workflow_id: str, task_id: str
    ) -> RecoveryDisposition | None:
        """Whether a settled v2 operation may be recomputed or must be restored."""
        with self._lock:
            engine = self._engines.get(workflow_id)
            return engine.recovery_disposition(task_id) if engine else None

    def mark_v2_uncertain(self, task_id: str) -> Advance:
        """Resolve a lost acknowledgement or route loss for an in-flight v2 work item.

        A replayable invocation is reissued through its stable identity as a fresh
        attempt; a non-replayable one becomes ambiguity-terminal and never silently
        retries or reports success.
        """
        with self._cv:
            return self._resolve_uncertain_locked(task_id)

    def _resolve_uncertain_locked(self, task_id: str) -> Advance:
        record = self._tasks.get(task_id)
        if record is None or (engine := self._engines.get(record.workflow_id)) is None:
            return Advance()
        advance = engine.on_uncertain(task_id)
        if advance.retry:
            self.mark_pending(task_id, increment_retry=False)
            self.requeue(task_id, front=True)
        elif advance.failed:
            self._fail_v2_records_locked(
                advance.failed, "ambiguity-terminal effect", persist=True
            )
        self._save_ledger_locked(record.workflow_id)
        return advance

    def _save_ledger_locked(self, workflow_id: str) -> None:
        if (engine := self._engines.get(workflow_id)) is not None:
            self._workflow_registry.save_ledger_snapshot(
                workflow_id, engine.to_snapshot()
            )

    def _apply_advance_locked(self, workflow_id: str, advance: Advance) -> bool:
        # A ready/settle advance never carries a retry; the failure path drives those.
        assert not advance.retry, "retry is applied by the failure path"
        engine = self._engines.get(workflow_id)
        if engine is not None:
            self._resolve_agent_inputs_locked(engine, advance)
            self._retire_sealed_region_templates_locked(workflow_id, engine)
        changed = False
        for task_id in advance.ready:
            if self._enqueue_ready_locked(task_id):
                changed = True
        for task_id in advance.failed:
            reason = (
                engine and engine.failure_reason(task_id)
            ) or "declared-failure obligation"
            self._fail_v2_records_locked([task_id], reason, persist=True)
            changed = True
        return changed

    def _fan_out_children_locked(
        self, workflow_id: str, engine: OrchestrationEngine, producer_task_id: str
    ) -> Advance:
        """Materialize one dispatchable child per element of a producer's fan-out.

        When the settled producer feeds a spawn whose child template is a dispatchable
        leaf, its result collection drives the child cardinality: each element mints a
        child work item with a synthesized task record, the child-init authority is then
        sealed, and the new records persist durably ahead of the ledger snapshot. A
        producer that feeds no spawn, or a spawn whose child body is not a leaf task,
        yields no children.
        """
        advance = Advance()
        spawn_op = engine.spawn_successor(producer_task_id)
        if spawn_op is None:
            return advance
        if not engine.spawn_is_open(spawn_op):
            return advance  # already sealed: a re-driven fan-out is a no-op
        child_template_id = engine.child_template_of(spawn_op)
        template_record = (
            self._tasks.get(child_template_id) if child_template_id else None
        )
        if child_template_id is None or template_record is None:
            # Compile-time validation rejects an unresolved or non-leaf child template,
            # so reaching here is an internal inconsistency; fail the workflow rather
            # than defer a join that could never close.
            self._logger.error(
                "Spawn %s child template %r is unresolvable; failing the workflow",
                spawn_op,
                child_template_id,
            )
            self._fail_workflow_locked(
                workflow_id,
                f"spawn child template {child_template_id!r} is not a "
                "dispatchable leaf",
            )
            return advance
        elements = self._load_fanout_elements(producer_task_id)
        if elements is None:
            # The producer's result is not yet on this node: defer rather than seal a
            # spurious zero-child spawn. A later replay re-drives the fan-out once the
            # result lands, and the spawn stays open until then.
            self._logger.warning(
                "Deferring fan-out for %s: its result is not available yet",
                producer_task_id,
            )
            return advance
        # An agent child receives its element through the typed accepted-input channel
        # its declared entry port; a leaf child keeps the legacy spec.data injection.
        child_is_agent = engine.agent_entry_port(child_template_id) is not None
        new_children: list[str] = []
        for index, element in enumerate(elements):
            try:
                if child_is_agent:
                    child_task_id = engine.create_fanout_child(
                        spawn_op, producer_task_id, index
                    )
                    self._register_child_locked(child_task_id, template_record, None)
                    self._mint_fanout_facet_locked(
                        engine, child_task_id, producer_task_id, index
                    )
                    advance.extend(engine.reconsider_admission(child_task_id))
                    new_children.append(child_task_id)
                    continue
                value_ref = ValueRef(
                    kind="legacy_task_result", legacy_task_id=producer_task_id
                )
                child_advance = engine.materialize_child(spawn_op, value_ref=value_ref)
            except RegionError:
                break  # a budget, seal, or denial stops further children
            for child_task_id in child_advance.ready:
                self._register_child_locked(child_task_id, template_record, element)
                new_children.append(child_task_id)
            advance.extend(child_advance)
        advance.extend(engine.seal_spawn(spawn_op))
        self._commit_new_children_locked(
            workflow_id, engine, new_children, retire=[child_template_id]
        )
        return advance

    def _load_fanout_elements(self, producer_task_id: str) -> list[Any] | None:
        """The producer result's fan-out collection: a ``fanout`` or ``items`` list.

        Returns ``None`` when the producer result is not yet readable on this node — a
        distinct signal from an empty collection, so a not-yet-delivered result is
        deferred rather than mistaken for a zero-child fan-out.
        """
        path = result_file_path(self._results_dir, producer_task_id)
        if not path.exists():
            return None
        try:
            envelope = ResultEnvelope.model_validate_json(path.read_text("utf-8"))
        except (ValueError, OSError):
            # A present-but-unreadable result is an anomaly, not a not-yet-delivered
            # one; surface it and defer rather than seal a wrong empty fan-out.
            self._logger.error(
                "Fan-out producer %s has an unreadable result at %s",
                producer_task_id,
                path,
            )
            return None
        payload = envelope.result.model_dump()
        collection = payload.get("fanout")
        if not isinstance(collection, list):
            collection = payload.get("items")
        if not isinstance(collection, list):
            return []
        # Unwrap a producer element's carried value (an echo item is ``{output: v}``)
        # so it is a valid child input rather than the producer's own result shape.
        return [
            item["output"] if isinstance(item, dict) and "output" in item else item
            for item in collection
        ]

    def _mint_fanout_facet_locked(
        self,
        engine: OrchestrationEngine,
        child_task_id: str,
        producer_task_id: str,
        index: int,
    ) -> None:
        """Record a fan-out child's typed entry-port input from its producer element."""
        wi = engine.work_item(child_task_id)
        if wi is None:
            return
        entry_port = engine.agent_entry_port(wi.operator_id)
        if entry_port is None:
            return
        value_ref = ValueRef(
            kind="legacy_task_result",
            legacy_task_id=producer_task_id,
            collection_key=str(index),
        )
        engine.record_accepted_input(
            AcceptedInput(
                activation_id=wi.activation_id,
                target_port=entry_port,
                occurrence_key=str(index),
                provenance="spawn_element",
                members=(
                    AcceptedInputMember(
                        source_operator_id=producer_task_id,
                        source_activation_id=wi.activation_id,
                        child_index=index,
                        outcome=PublicationOutcome.SUCCESS,
                        value_ref=value_ref,
                    ),
                ),
            )
        )

    def _resolve_agent_inputs_locked(
        self, engine: OrchestrationEngine, advance: Advance
    ) -> None:
        """Resolve and record each edge-bound agent's accepted inputs, then re-admit.

        The engine owns membership and ordering; the runtime resolves every member's
        frozen value and records the durable accepted input. A member whose value is not
        yet readable defers the whole port until a later advance.
        """
        for task_id in engine.blocked_input_agents():
            plan = engine.agent_input_plan(task_id)
            if plan is None:
                continue
            resolved: list[tuple[str, AcceptedInput]] = []
            total_bytes = 0
            for port in plan.ports:
                members: list[AcceptedInputMember] = []
                for member in port.members:
                    value_ref = ValueRef(
                        kind=member.value_ref_kind,
                        legacy_task_id=member.legacy_task_id,
                        collection_key=member.collection_key,
                        literal=member.literal,
                    )
                    value = self._resolve_value_ref(value_ref)
                    if value is None:
                        break
                    total_bytes += len(value.encode("utf-8"))
                    members.append(
                        AcceptedInputMember(
                            source_operator_id=member.source_operator_id,
                            source_activation_id=member.source_activation_id,
                            child_index=member.child_index,
                            outcome=PublicationOutcome(member.outcome),
                            value_ref=value_ref,
                            ordinal=member.ordinal,
                        )
                    )
                if len(members) != len(port.members):
                    continue
                resolved.append(
                    (
                        port.target_port,
                        AcceptedInput(
                            activation_id=plan.activation_id,
                            target_port=port.target_port,
                            provenance=port.provenance,
                            members=tuple(members),
                        ),
                    )
                )
            # Overflow is a typed declared failure, never silent truncation: a resolved
            # input cone exceeding the budget fails the agent instead of dispatching it.
            if total_bytes > self._input_budget_bytes:
                advance.extend(
                    engine.on_failed(
                        task_id,
                        f"input_too_large: resolved input is {total_bytes} bytes, "
                        f"over the {self._input_budget_bytes}-byte budget",
                        retryable=False,
                    )
                )
                continue
            for _, accepted in resolved:
                engine.record_accepted_input(accepted)
            advance.extend(engine.reconsider_admission(task_id))

    def _resolve_value_ref(self, value_ref: ValueRef | None) -> str | None:
        """Resolve a frozen value reference to its immutable string value, or None.

        An inline value is its literal; a producer reference reads the settled result
        selects one collection element (a fan-out element) or the whole ``value``. None
        signals a not-yet-readable result, deferring the input.
        """
        if value_ref is None:
            return None
        if value_ref.kind == "inline":
            return value_ref.literal or ""
        if value_ref.kind == "empty":
            return ""
        if value_ref.kind != "legacy_task_result" or not value_ref.legacy_task_id:
            return None
        path = result_file_path(self._results_dir, value_ref.legacy_task_id)
        if not path.exists():
            return None
        try:
            envelope = ResultEnvelope.model_validate_json(path.read_text("utf-8"))
        except (ValueError, OSError):
            return None
        payload = envelope.result.model_dump()
        if value_ref.collection_key is not None:
            collection = payload.get("fanout")
            if not isinstance(collection, list):
                collection = payload.get("items")
            if not isinstance(collection, list):
                return None
            index = int(value_ref.collection_key)
            if index < 0 or index >= len(collection):
                return None
            item = collection[index]
            item = (
                item["output"] if isinstance(item, dict) and "output" in item else item
            )
            return _stringify(item)
        value = payload.get("value")
        return _stringify(value if value is not None else payload)

    def _agent_input_bindings(
        self, engine: OrchestrationEngine, task_id: str
    ) -> tuple[InputBinding, ...]:
        """The resolved first-turn input bindings for an agent's input ports."""
        bindings: list[InputBinding] = []
        for ordinal, accepted in enumerate(engine.accepted_inputs_for_task(task_id)):
            members = tuple(
                InputBindingMember(
                    source_operator_id=member.source_operator_id,
                    source_activation_id=member.source_activation_id,
                    child_index=member.child_index,
                    outcome=member.outcome.value,
                    value=self._resolve_value_ref(member.value_ref),
                    ordinal=member.ordinal,
                )
                for member in accepted.members
            )
            bindings.append(
                InputBinding(
                    port=accepted.target_port,
                    provenance=accepted.provenance,
                    ordinal=accepted.ordinal or ordinal,
                    members=members,
                )
            )
        return tuple(bindings)

    def _synthesize_child_record(
        self, template: TaskRecord, child_task_id: str, element: Any
    ) -> TaskRecord:
        """Clone a child-template record for one materialized child, injecting input."""
        task = template.task
        spec = task.spec
        if element is not None and "data" in type(spec).model_fields:
            spec = spec.model_copy(
                update={"data": {"type": "list", "items": [element]}}
            )
            task = task.model_copy(update={"spec": spec})
        return template.model_copy(
            deep=True,
            update={
                "task_id": child_task_id,
                "task": task,
                "status": TaskStatus.PENDING,
                "assigned_worker": None,
                "started_ts": None,
                "finished_ts": None,
                "error": None,
                "usages": [],
                "position_in_epoch": None,
                "graph_node_name": None,
                "local_name": None,
                "merge_key": None,
                "merged_children": None,
                "selected_worker": None,
                "submitted_ts": time.time(),
            },
        )

    def _fail_v2_records_locked(
        self, task_ids: list[str], reason: str, *, persist: bool
    ) -> list[str]:
        """Fail the non-terminal task records terminally; returns the ones changed.

        Persists them here when ``persist`` is set; the failure cascade instead lets the
        caller's single terminal-persist cover them.
        """
        failed_now: list[str] = []
        for task_id in task_ids:
            record = self._tasks.get(task_id)
            if not record or record.status in TERMINAL_TASK_STATUSES:
                continue
            record.status = TaskStatus.FAILED
            record.error = reason
            record.assigned_worker = None
            record.finished_ts = time.time()
            self._failed.add(task_id)
            self._remove_from_ready_locked(task_id)
            failed_now.append(task_id)
        if persist and failed_now:
            self._persist_terminal_locked(*failed_now)
        return failed_now

    def _fail_workflow_locked(self, workflow_id: str, reason: str) -> None:
        """Fail every non-terminal task of a workflow and persist the terminal facts."""
        non_terminal = [
            task_id
            for task_id, record in self._tasks.items()
            if record.workflow_id == workflow_id
            and record.status not in TERMINAL_TASK_STATUSES
        ]
        if self._fail_v2_records_locked(non_terminal, reason, persist=True):
            self._save_ledger_locked(workflow_id)
            self._cv.notify_all()

    def _fail_v2_cascade_locked(
        self, primary: str, cascade: list[str]
    ) -> list[tuple[str, str]]:
        reason = f"Dependency {primary} failed"
        downstream = [task_id for task_id in cascade if task_id != primary]
        failed = self._fail_v2_records_locked(downstream, reason, persist=False)
        return [(task_id, reason) for task_id in failed]

    def plan_merge(
        self, task_id: str, max_batch_size: int, assigned_worker: str
    ) -> list[str]:
        if max_batch_size <= 1:
            return []
        with self._cv:
            return self._plan_merge_locked(task_id, max_batch_size, assigned_worker)

    def _plan_merge_locked(
        self, task_id: str, max_batch_size: int, assigned_worker: str
    ) -> list[str]:
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return []
        if record.merge_key is None:
            return []
        if self._merge_children_map.get(task_id):
            return []
        if record.selected_worker and assigned_worker not in record.selected_worker:
            raise ValueError(
                f"The worker assigned for task {task_id} ({assigned_worker}) "
                f"is not in selected workers {record.selected_worker}."
            )
        bucket = (
            self._merge_buckets[(record.merge_key, assigned_worker)]
            + self._merge_buckets[(record.merge_key, None)]
        )
        if not bucket or len(bucket) <= 1:
            return []
        siblings: list[str] = []
        for candidate in bucket:
            if candidate == task_id:
                continue
            if len(siblings) >= max_batch_size - 1:
                break
            candidate_record = self._tasks.get(candidate)
            if not candidate_record or candidate_record.status != TaskStatus.PENDING:
                continue
            if (
                candidate_record.selected_worker
                and assigned_worker not in candidate_record.selected_worker
            ):
                continue
            if candidate not in self._ready_index:
                continue
            siblings.append(candidate)
        if not siblings:
            return []

        record.merged_children = siblings
        self._merge_children_map[task_id] = siblings.copy()
        for sibling in siblings:
            self._merge_parent_map[sibling] = task_id
            self._remove_from_ready_locked(sibling)
            self._merge_bucket_remove(sibling)
            sibling_record = self._tasks.get(sibling)
            if sibling_record:
                sibling_record.status = TaskStatus.DISPATCHED
                sibling_record.merged_parent_id = task_id
                sibling_record.assigned_worker = None
                sibling_record.merge_slice = None
        self._workflow_registry.commit_transition(
            record.workflow_id,
            records=self._records_locked(task_id, *siblings),
            dispatched=siblings,
        )

        return siblings

    def release_merge(self, task_id: str) -> None:
        with self._cv:
            self._release_merge_locked(task_id)

    def _release_merge_locked(self, task_id: str) -> None:
        children = self._merge_children_map.pop(task_id, [])
        if not children:
            parent = self._tasks.get(task_id)
            if parent:
                parent.merged_children = None
            self._persist_locked(task_id)
            return
        parent = self._tasks.get(task_id)
        if parent:
            parent.merged_children = None
        for child_id in children:
            self._merge_parent_map.pop(child_id, None)
            child_record = self._tasks.get(child_id)
            if not child_record:
                continue
            if child_record.status == TaskStatus.DONE:
                continue
            child_record.status = TaskStatus.PENDING
            child_record.merged_parent_id = None
            child_record.merge_slice = None
            if child_id not in self._ready_index:
                self._enqueue_ready_locked(child_id, front=True)
            else:
                self._remove_from_ready_locked(child_id)
                self._enqueue_ready_locked(child_id, front=True)
        self._persist_locked(task_id, *children)
        self._cv.notify_all()

    def _finalize_merged_child_success(
        self,
        child_id: str,
        worker_id: str | None,
        finished_ts: float,
        started_ts: float | None,
        usage: TaskUsage | None,
    ) -> list[str]:
        ready_children: list[str] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return ready_children
        child_record.status = TaskStatus.DONE
        child_record.error = None
        child_record.finished_ts = finished_ts
        if started_ts is not None and child_record.started_ts is None:
            child_record.started_ts = started_ts
        if worker_id:
            child_record.assigned_worker = worker_id
        child_record.merged_parent_id = None
        child_record.merge_slice = None
        if usage is not None:
            child_record.usages.append(usage)
        self._completed.add(child_id)
        self._failed.discard(child_id)
        self._pending_deps.pop(child_id, None)
        self._merge_parent_map.pop(child_id, None)
        self._merge_key_by_task.pop(child_id, None)
        self._remove_from_ready_locked(child_id)
        self._merge_bucket_remove(child_id)
        dependents = list(self._dependents.pop(child_id, set()))
        for dep_id in dependents:
            pending = self._pending_deps.get(dep_id)
            if pending is None:
                continue
            pending.discard(child_id)
            if not pending:
                dep_record = self._tasks.get(dep_id)
                if dep_record and dep_record.status == TaskStatus.PENDING:
                    if self._enqueue_ready_locked(dep_id):
                        ready_children.append(dep_id)
        return ready_children

    def _finalize_merged_child_failure(
        self,
        child_id: str,
        reason: str,
        finished_ts: float,
        started_ts: float | None,
        usage: TaskUsage | None,
    ) -> list[tuple[str, str]]:
        impacted: list[tuple[str, str]] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return impacted
        child_record.status = TaskStatus.FAILED
        child_record.error = reason
        child_record.finished_ts = finished_ts
        if started_ts is not None and child_record.started_ts is None:
            child_record.started_ts = started_ts
        child_record.assigned_worker = None
        child_record.merged_parent_id = None
        child_record.merge_slice = None
        if usage is not None:
            child_record.usages.append(usage)
        self._failed.add(child_id)
        self._completed.discard(child_id)
        self._pending_deps.pop(child_id, None)
        self._merge_parent_map.pop(child_id, None)
        self._merge_key_by_task.pop(child_id, None)
        self._remove_from_ready_locked(child_id)
        self._merge_bucket_remove(child_id)

        dependents = list(self._dependents.pop(child_id, set()))
        for dep_id in dependents:
            pending = self._pending_deps.get(dep_id)
            if pending is not None:
                pending.discard(child_id)
            dep_record = self._tasks.get(dep_id)
            if not dep_record or dep_record.status != TaskStatus.PENDING:
                continue
            fail_reason = f"Dependency {child_id} failed"
            dep_record.status = TaskStatus.FAILED
            dep_record.error = fail_reason
            dep_record.assigned_worker = None
            dep_record.finished_ts = time.time()
            self._pending_deps.pop(dep_id, None)
            self._remove_from_ready_locked(dep_id)
            impacted.append((dep_id, fail_reason))
        return impacted

    # ------------------------------------------------------------------ #
    # State updates (dispatch & events)
    # ------------------------------------------------------------------ #

    def mark_dispatched(self, task_id: str, worker: Worker) -> None:
        supplier_id = ""
        for resolver in SUPPLIER_RESOLVERS:
            if (resolved := resolver.resolve(worker)) is not None:
                supplier_id = resolved
                break

        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            if record.status in TERMINAL_TASK_STATUSES:
                # A replayed or late dispatch must not regress a terminal task.
                return
            record.status = TaskStatus.DISPATCHED
            record.assigned_worker = worker.id
            record.topic = "tasks"
            record.dispatched_ts = time.time()
            record.next_retry_at = None
            record.supplier_id = supplier_id
            self._remove_from_ready_locked(task_id)
            self._merge_bucket_remove(task_id)
            self._workflow_registry.commit_transition(
                record.workflow_id,
                records=self._records_locked(task_id),
                dispatched=[task_id],
            )
            if engine := self._engines.get(record.workflow_id):
                engine.on_dispatched(task_id, worker.id)
                self._save_ledger_locked(record.workflow_id)

    def mark_started(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
    ) -> None:
        started_ts = parse_iso_ts(str(payload.get("started_at") or ts))
        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            if record.status in TERMINAL_TASK_STATUSES:
                # A replayed or late start must not regress a terminal task.
                return
            record.status = TaskStatus.DISPATCHED
            record.started_ts = started_ts
            if worker_id:
                record.assigned_worker = worker_id
            self._workflow_registry.commit_transition(
                record.workflow_id,
                records=self._records_locked(task_id),
                dispatched=[task_id],
            )
            if engine := self._engines.get(record.workflow_id):
                engine.on_started(task_id)
                self._save_ledger_locked(record.workflow_id)

    def mark_updated(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record.status in TERMINAL_TASK_STATUSES:
                # A replayed or late progress update must not touch a terminal task.
                return
            record.latest_update = payload
            self._persist_locked(task_id)

    def mark_succeeded(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        *,
        empty: bool = False,
    ) -> list[tuple[str, TaskUsage]]:
        """
        Mark a task as completed and enqueue any dependents that have become ready.
        Returns the per-task usage rows produced by the completion. ``empty`` records a
        conditional-skip settlement, which resolves a v2 output to an explicit-empty
        publication.
        """
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        # TODO(kaiitunnz): Make usage task-specific
        usage = TaskUsage.from_payload(payload, TaskStatus.DONE)
        usages: list[tuple[str, TaskUsage]] = []
        if usage is not None:
            usages.append((task_id, usage))

        with self._cv:
            record = self._tasks.get(task_id)
            episode_step = payload.get("agent_episode")
            if episode_step is not None and record is not None:
                harness_result = HarnessResult.model_validate(episode_step)
                # The durable record is the source of truth: a restart drops the
                # in-memory stash, but a replayed completion still finds its captured
                # boundary and reroute rather than settling the episode DONE.
                group = self._pending_facade_groups.pop(task_id, None)
                if group is None:
                    group = record.pending_facade_group
                if harness_result.kind is not HarnessResultKind.COMPLETION:
                    # A non-terminal episode step routes its boundary and re-dispatches;
                    # a completion falls through to the terminal path below.
                    self._apply_episode_step_locked(task_id, harness_result)
                    return usages
                if group is not None:
                    # The gateway captured a turn-scoped facade group: the clean
                    # turn-completion is a yield on that group, not the episode's
                    # terminal result, so it routes the whole ordered membership
                    # kind-specifically, consumed durably in the same reroute save.
                    record.pending_facade_group = None
                    self._route_and_dispatch_facade_group_locked(
                        task_id, group, harness_result.capsule
                    )
                    return usages
                engine = self._engines.get(record.workflow_id)
                if (
                    engine is not None
                    and record.status not in TERMINAL_TASK_STATUSES
                    and not engine.latest_attempt_open(task_id)
                ):
                    # A completion whose attempt the reroute already closed is a
                    # superseded replay; settling it would preempt the live turn and
                    # publish the episode with a stale intermediate result. A post-DONE
                    # terminal replay is excluded so it still reaches the idempotent
                    # done-branch below (its fan-out / re-persist heal must survive).
                    return usages
            if record:
                if record.status == TaskStatus.CANCELLED:
                    return usages
                if record.status == TaskStatus.DONE:
                    # Idempotent: a replayed TASK_SUCCEEDED must not re-apply, but
                    # re-persist in case the original completion's write failed
                    # after its in-memory commit.
                    self._repersist_terminal_workflow_locked(record.workflow_id)
                    # Recover a fan-out lost to a crash between the producer's terminal
                    # persist and its children: re-driving is a no-op once the spawn has
                    # sealed, so replaying an already-materialized fan-out does nothing.
                    if engine := self._engines.get(record.workflow_id):
                        advance = self._fan_out_children_locked(
                            record.workflow_id, engine, task_id
                        )
                        if self._apply_advance_locked(record.workflow_id, advance):
                            self._cv.notify_all()
                    return []
                if record.status == TaskStatus.FAILED:
                    self._logger.warning(
                        "Ignoring TASK_SUCCEEDED for task %s in terminal status FAILED",
                        task_id,
                    )
                    return []
                record.status = TaskStatus.DONE
                record.error = None
                record.finished_ts = finished_ts
                if started_ts:
                    record.started_ts = started_ts
                if worker_id:
                    record.assigned_worker = worker_id
                record.merged_children = None
                if usage is not None:
                    record.usages.append(usage)

            self._completed.add(task_id)
            self._failed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            ready_children: list[str] = []
            merged_children_ids: list[str] = self._merge_children_map.pop(task_id, [])
            self._merge_key_by_task.pop(task_id, None)

            dependents = list(self._dependents.pop(task_id, set()))
            for child in dependents:
                pending = self._pending_deps.get(child)
                if pending is None:
                    continue
                pending.discard(task_id)
                if not pending:
                    child_record = self._tasks.get(child)
                    if child_record and child_record.status == TaskStatus.PENDING:
                        if self._enqueue_ready_locked(child):
                            ready_children.append(child)

            for merged_child in merged_children_ids:
                ready_children.extend(
                    self._finalize_merged_child_success(
                        merged_child,
                        worker_id,
                        finished_ts,
                        started_ts,
                        usage,
                    )
                )

            if record is not None:
                ready_children.extend(
                    self._try_advance_epoch_frontier_locked(record.workflow_id)
                )

            self._persist_terminal_locked(task_id, *merged_children_ids)

            notify = bool(ready_children)
            if record is not None and (engine := self._engines.get(record.workflow_id)):
                advance = engine.on_succeeded(task_id, empty=empty)
                advance.extend(
                    self._fan_out_children_locked(record.workflow_id, engine, task_id)
                )
                if self._apply_advance_locked(record.workflow_id, advance):
                    notify = True
                self._save_ledger_locked(record.workflow_id)

            if record is not None:
                self._reclaim_vault_if_settled_locked(record.workflow_id)
            if notify:
                self._cv.notify_all()

            return usages

    def mark_failed(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        *,
        error: str | None = None,
    ) -> tuple[list[tuple[str, str]], list[str], list[tuple[str, TaskUsage]]]:
        """
        Mark a task as failed. Dependent tasks still waiting on this task are
        automatically failed to avoid running without prerequisites.

        Returns (impacted_dependents, merged_children_ids, usages).
        """
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        message = error or str(payload.get("error") or "task failed")
        # TODO(kaiitunnz): Make usage task-specific
        usage = TaskUsage.from_payload(payload, TaskStatus.FAILED)
        usages: list[tuple[str, TaskUsage]] = []
        if usage is not None:
            usages.append((task_id, usage))

        with self._cv:
            record = self._tasks.get(task_id)
            if record:
                if record.status == TaskStatus.CANCELLED:
                    return [], [], usages
                if record.status == TaskStatus.FAILED:
                    # Idempotent: a replayed TASK_FAILED must not re-apply, but
                    # re-persist in case the original failure's write (including
                    # its cascade) failed after the in-memory commit.
                    self._repersist_terminal_workflow_locked(record.workflow_id)
                    return [], [], []
                if record.status == TaskStatus.DONE:
                    self._logger.warning(
                        "Ignoring TASK_FAILED for task %s in terminal status DONE",
                        task_id,
                    )
                    return [], [], []
                record.status = TaskStatus.FAILED
                record.error = message
                record.finished_ts = finished_ts
                # A facade captured on a turn that then failed is never rerouted; drop
                # it so a replayed completion can't resurrect it against a failed task.
                self._pending_facade_groups.pop(task_id, None)
                record.pending_facade_group = None
                if started_ts:
                    record.started_ts = started_ts
                if worker_id:
                    record.assigned_worker = worker_id
                record.merged_children = None
                if usage is not None:
                    record.usages.append(usage)

            self._failed.add(task_id)
            self._completed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            self._remove_from_ready_locked(task_id)
            merged_children_ids = self._merge_children_map.pop(task_id, [])
            self._merge_key_by_task.pop(task_id, None)

            impacted: list[tuple[str, str]] = []
            dependents = list(self._dependents.pop(task_id, set()))
            for child in dependents:
                pending = self._pending_deps.get(child)
                if pending is not None:
                    pending.discard(task_id)
                child_record = self._tasks.get(child)
                if not child_record or child_record.status != TaskStatus.PENDING:
                    continue
                reason = f"Dependency {task_id} failed"
                child_record.status = TaskStatus.FAILED
                child_record.error = reason
                child_record.assigned_worker = None
                child_record.finished_ts = time.time()
                self._pending_deps.pop(child, None)
                self._remove_from_ready_locked(child)
                impacted.append((child, reason))

            if record is not None and (engine := self._engines.get(record.workflow_id)):
                advance = engine.on_failed(task_id, message, retryable=False)
                impacted.extend(self._fail_v2_cascade_locked(task_id, advance.failed))

            for merged_child in merged_children_ids:
                impacted.extend(
                    self._finalize_merged_child_failure(
                        merged_child,
                        f"Parent {task_id} failed",
                        finished_ts,
                        started_ts,
                        usage,
                    )
                )

            failed_epoch = self._task_epoch_index.get(task_id)
            if record and failed_epoch is not None:
                impacted.extend(
                    self._fail_later_epochs_locked(
                        record.workflow_id,
                        failed_epoch,
                        f"Blocked by failed task {task_id} in earlier epoch",
                    )
                )

            self._persist_terminal_locked(
                task_id, *merged_children_ids, *(dep_id for dep_id, _ in impacted)
            )
            # The ledger snapshot writes last, after the task terminal records, so a
            # crash can only leave the ledger behind — never ahead — of durable task
            # state, which rehydration then reconciles.
            if record is not None and record.workflow_id in self._engines:
                self._save_ledger_locked(record.workflow_id)

            if record is not None:
                self._reclaim_vault_if_settled_locked(record.workflow_id)
            return impacted, merged_children_ids, usages

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def cancel_workflow(self, workflow_id: str, reason: str = "cancelled") -> list[str]:
        cancelled: list[str] = []
        cancelling: list[str] = []
        touched: list[str] = []
        interrupts: list[InterruptMessage] = []
        resident_invocation_ids: list[str] = []
        with self._cv:
            workflow_tasks = [
                item
                for item in self._tasks.items()
                if item[1].workflow_id == workflow_id
            ]
            if not workflow_tasks:
                return touched  # Unknown workflow: no records to move
            for task_id, record in workflow_tasks:
                match record.status:
                    case TaskStatus.PENDING if not self._parent_is_active(task_id):
                        record.status = TaskStatus.CANCELLED
                        record.error = reason
                        record.finished_ts = time.time()
                        record.assigned_worker = None
                        record.merged_children = None
                        # TODO(kaiitunnz): Handle usages for cancelled tasks
                        self._completed.discard(task_id)
                        self._failed.discard(task_id)
                        self._pending_deps.pop(task_id, None)
                        self._remove_from_ready_locked(task_id)
                        self._merge_bucket_remove(task_id)
                        self._merge_key_by_task.pop(task_id, None)
                        self._merge_parent_map.pop(task_id, None)
                        self._merge_children_map.pop(task_id, None)
                        cancelled.append(task_id)
                        touched.append(task_id)
                    case TaskStatus.DISPATCHED if (
                        not self._parent_is_active(task_id)
                        and not record.merged_children
                        and record.assigned_worker
                    ):
                        record.status = TaskStatus.CANCELLING
                        record.error = reason
                        interrupts.append(
                            InterruptMessage(
                                task_id=task_id,
                                worker_id=record.assigned_worker,
                                reason=reason,
                            )
                        )
                        cancelling.append(task_id)
                        touched.append(task_id)
                    case _:
                        continue

            self._workflow_epoch_tasks.pop(workflow_id, None)
            self._workflow_epoch_frontier.pop(workflow_id, None)
            self._workflow_in_epoch_order.pop(workflow_id, None)
            for task_id, _ in workflow_tasks:
                self._task_epoch_index.pop(task_id, None)
            self._workflow_registry.commit_transition(
                workflow_id,
                records=self._records_locked(*touched),
                cancelled=cancelled,
                sched=self._sched_locked(workflow_id),
            )
            # Mirror the cancellation into the orchestration ledger so a v2 workflow's
            # work items settle CANCELLED in lockstep with its task records; the
            # snapshot follows the committed task state so the ledger never leads it.
            if workflow_id in self._engines:
                engine = self._engines[workflow_id]
                engine.cancel_instance()
                resident_invocation_ids = (
                    engine.cancel_outstanding_boundary_invocations()
                )
                self._save_ledger_locked(workflow_id)

        # A cancelled in-flight resident invocation releases its credit from this fenced
        # cancellation terminal, so a lost or draining replica is not held forever.
        for invocation_id in resident_invocation_ids:
            self._release_resident_credit(invocation_id, failed=True)

        for interrupt in interrupts:
            worker = self._worker_registry.get_worker(interrupt.worker_id)
            if worker is None:
                self._logger.warning(
                    "Cannot publish interrupt for %s; worker %s missing",
                    interrupt.task_id,
                    interrupt.worker_id,
                )
            else:
                self._worker_registry.publish_interrupt(worker, interrupt)
        self._secret_vault.purge(workflow_id)
        return touched

    def mark_cancelled(
        self, task_id: str, worker_id: str | None, payload: dict[str, Any], ts: str
    ) -> list[tuple[str, TaskUsage]]:
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        usage = TaskUsage.from_payload(payload, TaskStatus.CANCELLED)
        usages: list[tuple[str, TaskUsage]] = []
        if usage is not None:
            usages.append((task_id, usage))

        with self._cv:
            record = self._tasks.get(task_id)
            if record is None:
                return usages
            if record.status == TaskStatus.CANCELLED:
                # Idempotent: a replayed cancellation must not re-apply, but
                # re-persist in case the original cancellation's write failed
                # after its in-memory commit.
                self._repersist_terminal_workflow_locked(record.workflow_id)
                return usages
            if record.status in (TaskStatus.DONE, TaskStatus.FAILED):
                self._logger.warning(
                    "Ignoring cancellation for task %s in terminal status %s",
                    task_id,
                    record.status,
                )
                return usages
            record.status = TaskStatus.CANCELLED
            record.finished_ts = finished_ts
            if started_ts:
                record.started_ts = started_ts
            record.merged_children = None
            if usage is not None:
                record.usages.append(usage)
            # TODO(kaiitunnz): Handle usages for cancelled tasks
            self._completed.discard(task_id)
            self._failed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            self._remove_from_ready_locked(task_id)
            self._merge_bucket_remove(task_id)
            self._merge_key_by_task.pop(task_id, None)
            self._merge_children_map.pop(task_id, None)
            record.assigned_worker = None
            # Persist the task terminal record first, then mirror the cancellation
            # into the ledger and snapshot last, so the ledger never leads task state.
            self._persist_terminal_locked(task_id, sched=False)
            if (engine := self._engines.get(record.workflow_id)) is not None:
                advance = engine.on_cancelled(task_id)
                assert not (
                    advance.ready or advance.retry
                ), "a whole-instance cancel readies no work"
                self._save_ledger_locked(record.workflow_id)
            self._reclaim_vault_if_settled_locked(record.workflow_id)
            return usages

    def get_record(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_merged_children(self, task_id: str) -> list[str]:
        """Read the merged-children list without consuming it."""
        with self._cv:
            return self._merge_children_map.get(task_id, [])

    def describe_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None
            return self._build_task_info_locked(task_id, record)

    def list_tasks(self) -> list[TaskInfo]:
        with self._lock:
            return [
                self._build_task_info_locked(task_id, record)
                for task_id, record in self._tasks.items()
            ]

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    @property
    def tasks(self) -> dict[str, TaskRecord]:
        return self._tasks

    def recover_tasks_for_worker(self, worker_id: str) -> list[str]:
        """
        Move DISPATCHED tasks assigned to a departed worker back to the ready queue.
        Returns affected task_ids.
        """
        recovered: list[str] = []
        with self._cv:
            for task_id, record in list(self._tasks.items()):
                if record.assigned_worker != worker_id:
                    continue
                if record.status not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                    continue
                self._rehydrated_dispatched.pop(task_id, None)
                # v2 route/worker loss resolves through the uncertainty FSM: a
                # replayable invocation reissues under its stable id; the caller does
                # not also fail or requeue it.
                if (
                    record.status == TaskStatus.DISPATCHED
                    and record.workflow_id in self._engines
                ):
                    self._resolve_uncertain_locked(task_id)
                    continue
                recovered.append(task_id)
        return recovered

    def has_rehydrated_in_flight(self, worker_id: str, within_sec: float) -> bool:
        """
        Whether ``worker_id`` still owns an in-flight task that was rehydrated within
        the last ``within_sec`` seconds.

        Worker heartbeats are dropped while the root is down, so a surviving worker
        looks briefly stale right after a restart. The watchdog uses this to extend a
        worker's death grace until its rehydrated tasks' window has elapsed, giving the
        worker time to re-register before its tasks are reclaimed.
        """
        now = time.time()
        with self._cv:
            for task_id, rehydrated_at in list(self._rehydrated_dispatched.items()):
                record = self._tasks.get(task_id)
                if record is None or record.status not in (
                    TaskStatus.DISPATCHED,
                    TaskStatus.CANCELLING,
                ):
                    self._rehydrated_dispatched.pop(task_id, None)
                    continue
                if now - rehydrated_at >= within_sec:
                    continue
                if record.assigned_worker == worker_id:
                    return True
        return False

    def shutdown(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def ready_queue_length(self) -> int:
        with self._cv:
            return len(self._ready_queue)

    def queued_gpu_counts(self) -> set[int]:
        """Return the set of distinct GPU counts requested by tasks in the ready queue.

        0 represents a CPU-only task.  Used to match each candidate server to the best
        worker it can create for the current queue.
        """
        counts: set[int] = set()
        with self._cv:
            for task_id, _ in self._ready_queue:
                record = self._tasks.get(task_id)
                if record is None:
                    continue
                resources = record.task.spec.resources
                if resources is None or resources.hardware is None:
                    counts.add(0)
                    continue
                gpu = resources.hardware.gpu
                if gpu:
                    # Default to 1 if a GPU is required but count is unspecified
                    counts.add(int(gpu.count) if gpu.count else 1)
                else:
                    counts.add(0)
        return counts

    def task_status_counts(self) -> tuple[int, int, int, int, int]:
        with self._cv:
            queueing = len(self._ready_queue)
            dispatched = 0
            pending = 0
            done = 0
            for task_id, record in self._tasks.items():
                status = record.status
                if status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                    dispatched += 1
                elif status == TaskStatus.DONE:
                    done += 1
                elif status == TaskStatus.PENDING and task_id not in self._ready_index:
                    pending += 1
            total = len(self._tasks)
            return queueing, dispatched, pending, done, total

    def _build_task_info_locked(self, task_id: str, record: TaskRecord) -> TaskInfo:
        return TaskInfo(
            **dict(record),
            depends_on=sorted(self._original_deps.get(task_id, set())),
            pending_dependencies=sorted(self._pending_deps.get(task_id, set())),
            dependents=sorted(self._dependents.get(task_id, set())),
            completed=task_id in self._completed,
            failed=task_id in self._failed,
        )

    def _parent_is_active(self, task_id: str) -> bool:
        parent_id = self._merge_parent_map.get(task_id)
        if not parent_id:
            return False
        parent_record = self._tasks.get(parent_id)
        if not parent_record:
            return False
        return parent_record.status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING)
