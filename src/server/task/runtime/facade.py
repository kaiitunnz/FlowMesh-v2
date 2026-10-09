"""The task runtime: every workflow's task records, scheduling and dispatch."""

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Final

from opentelemetry.trace import Tracer
from pydantic import ValidationError

from server.telemetry.tracing import (
    NULL_CONTROL_TRACER,
    ControlPlaneTracer,
    format_traceparent,
)
from shared.content import ContentReference
from shared.harness import (
    AgentEpisodeDispatch,
    BoundaryEventKind,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    ServiceLeafEpisodeDispatch,
)
from shared.inference import (
    CanonicalInferenceContract,
    InputResolutionBinding,
    ResolvedInputMaterialization,
)
from shared.outcome import OutcomeManifest
from shared.private_state import (
    OwnerFence,
    PrivateStateSealReport,
    PrivateStateUnavailable,
)
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentOpOutcome,
    ResidentRouteObservation,
)
from shared.schemas.command import MediatedOpMessage
from shared.schemas.event import TaskEvent, TaskFailureKind
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.schemas.result.binding import upstream_value
from shared.tasks import TaskEnvelopeTemplate
from shared.tasks.credentials import set_spec_values
from shared.tasks.result_binding import ResultBinding, ResultElementRef
from shared.tasks.specs import ModelBindingMode, TaskSpecBase
from shared.telemetry.config import TelemetryConfig
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.telemetry.semconv import ControlPlaneStage, ControlPlaneWindow
from shared.tools.contract import AgentModelTurnProposal, MediatedOperationOutcome
from shared.utils import new_workflow_id
from shared.utils.redact import credential_scrubber

from ...config import AgentBindingConfig, N8nConfig, OrchestrationConfig
from ...orchestration import (
    Advance,
    OrchestrationEngine,
    RecoveryDisposition,
    RegionError,
    ResultPublication,
    ScopeBudget,
    ValueRef,
    WorkItemStatus,
)
from ...orchestration.engine.advance import legacy_control_unsupported
from ...orchestration.engine.topology import (
    blueprint_operators,
    materialized_operators,
)
from ...orchestration.episode import BoundaryEvent
from ...orchestration.harness import to_boundary_event
from ...orchestration.state import TERMINAL_WORK_ITEM_STATUSES
from ...orchestration.telemetry import build_span_emitter
from ...orchestration.tool_dispatch import (
    MODEL_INTERFACE,
    SEARCH_INTERFACE,
    FacadeTurnGroup,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from ...registries.worker import Worker, WorkerRegistry
from ...registries.workflow import (
    PersistedTask,
    WorkflowControl,
    WorkflowRegistry,
    WorkflowSched,
)
from ...services.credential_vault import CredentialVault
from ...utils.cursors import page_slice
from ...utils.query import QueryFilter
from ...utils.time import now_iso, parse_iso_ts
from ..credentials import (
    CredentialRefs,
    InlineCredentials,
    TaskCredentials,
    credential_merge_key,
    mask_inline_credentials,
    redact_source_text,
    redact_stored_source,
    take_inline_credentials,
    take_spec_credentials,
)
from ..models import (
    SERVE_TASK_TYPES,
    SETTLING_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    DispatchEnd,
    EventEffect,
    FailureOutcome,
    LossOutcome,
    PublishGate,
    SettleOutcome,
    TaskInfo,
    TaskInputElement,
    TaskLoopTime,
    TaskOccurrence,
    TaskOrder,
    TaskParsingResult,
    TaskRecord,
    TaskStatus,
    TaskUsage,
    WorkerRecovery,
    WorkflowSettlement,
    categorize_task_type,
    task_order,
)
from ..outputs import (
    OutputMember,
    PublishedOutput,
    PublishedOutputs,
    published_member,
    published_members,
)
from ..parser import ParsedWorkflow, parse_workflow
from ..redrive import StoreRedriveScheduler
from ..results import ResultReader, ResultUnreadable
from ..v2 import (
    ExecutionMode,
    FrontendWorkflowSource,
    InspectionReport,
    LoweringStrategy,
    PersistedV2Workflow,
    build_inspection,
    compile_bundle,
)
from ..v2.compiler.agent_binding import AgentBindingDefaults
from ..v2.policy import PolicySurface
from ..v2.representations.admission import ResidentAdmissionBinding
from ..v2.representations.operators import AgentModelGatewayBinding
from ..v2.representations.plan import InferenceEmbodimentMenu
from ..workflow_retry import WorkflowRetryScheduler
from . import (
    agent_inputs,
    control_reads,
    fanout,
)
from .after_commit import (
    AfterCommit,
    AfterCommitActions,
    AuthorizeTurn,
    CreditRelease,
    Interrupt,
    Issue,
    Purge,
    Reap,
    Revoke,
    Settled,
)
from .agent_inputs import AgentInputs
from .commits import TransitionCommitter, TransitionNotDurable, store_unavailable
from .content_bindings import ContentBindings, ScopedInput, UnreadableInput
from .dispatch_fence import DispatchFence, Publish, supplier_id
from .episode_dispatch import EpisodeDispatch, EpisodeFeasibility
from .fanout import FanoutRead
from .input_checks import INPUT_VERDICT_REPORT, InputChecks
from .mediated_ops import (
    MediatedOperations,
    PendingOp,
    ResidentTerminalHook,
    deny_model_turn_payload,
)
from .merges import TaskMerges
from .occurrences import OccurrenceMaterializer, materialized_work
from .record_failures import RecordFailures
from .reports import (
    LOSS_EFFECTS,
    failed_task_can_retry,
    in_flight_usage,
    reported_child_references,
    reported_reference,
    reset_to_pending,
    settle_outcome,
)
from .reservations import WorkerReservations
from .resident_tasks import ResidentServeTasks
from .scheduling import EpochFrontier, ReadyQueue
from .static_dag import StaticDag
from .task_table import TaskTable

_RESIDUAL_CANCEL_REASON = "cancelled by its region's residual policy"
# The empty result a task no route reaches settles with.
ROUTE_NOT_TAKEN: Final = {"skipped": True, "reason": "route_not_taken"}


def blueprint_missing(operator_id: str) -> str:
    """The reason a workflow fails for when work of ``operator_id`` has no blueprint to
    make its task from."""
    return f"BlueprintMissing: operator {operator_id} has no task blueprint"


class _Drain(threading.local):
    active = False


def _issue(env: ToolInvocationEnvelope) -> Issue:
    return Issue(env.task_id, env.call_correlation, env.invocation_id)


def _failed_credits(invocation_ids: Iterable[str]) -> list[AfterCommit]:
    return [
        CreditRelease(invocation_id, failed=True) for invocation_id in invocation_ids
    ]


def _durability_retry(
    fire: Callable[[str], None], logger: logging.Logger
) -> WorkflowRetryScheduler:
    return WorkflowRetryScheduler(fire, logger, thread_name="durability-retry")


def _element(value: ValueRef, index: int) -> ValueRef:
    """One element of the collection a fan-out value names: a member of a whole
    result's collection or of a part of one, or an index into a part of an element."""
    if value.collection_key is None:
        return value.model_copy(
            update={
                "collection": value.projection,
                "projection": (),
                "collection_key": str(index),
            }
        )
    return value.model_copy(update={"projection": (*value.projection, index)})


@dataclass(frozen=True)
class _StagedRegistration:
    """A submission's records and plan, built in memory before its durable write."""

    results: list[TaskParsingResult]
    task_records: list[TaskRecord]
    depends_on: dict[str, set[str]]
    merge_keys: dict[str, tuple[str | None, str | None]]
    task_epoch_index: dict[str, int]
    epoch_queue: deque[set[str]] | None
    in_epoch_order: bool
    v2_bundle: PersistedV2Workflow | None
    v2_engine: OrchestrationEngine | None
    # The tasks region definitions and spawns materialize their work from.
    blueprints: list[TaskRecord]

    def persisted(self) -> list[PersistedTask]:
        return [
            PersistedTask(
                record=record,
                depends_on=self.depends_on[record.task_id],
                epoch_index=self.task_epoch_index.get(record.task_id),
            )
            for record in self.task_records
        ]


def _binding_defaults(
    cfg: AgentBindingConfig,
    sandbox_enabled: bool = False,
    sandbox_egress_enabled: bool = False,
) -> AgentBindingDefaults:
    """Convert the deployment binding config into the compiler's injected defaults."""
    return AgentBindingDefaults(
        default_backend=cfg.default_backend,
        default_version=cfg.default_version,
        default_mode=cfg.default_mode,
        default_url=cfg.default_url,
        default_model=cfg.default_model,
        sandbox_enabled=sandbox_enabled,
        sandbox_egress_enabled=sandbox_enabled and sandbox_egress_enabled,
    )


@dataclass(frozen=True)
class _OpCredential:
    """The provider credential authority one mediated-operation permit carries."""

    credential: str | None = field(default=None, repr=False)
    deployment_credential: bool = False


@dataclass(frozen=True)
class _MissingCredential:
    """A pinned model credential missing from the vault: the operation is denied."""

    reason: str = "model credential unavailable"


def _is_default_url(url: str | None, default_url: str | None) -> bool:
    if not url or not default_url:
        return False
    return url.rstrip("/") == default_url.rstrip("/")


# Extra lifetime a worker-originated operation permit gets beyond the request timeout,
# to cover dispatch and queue latency before the origin worker validates it.
_OP_PERMIT_SLACK_SEC = 60.0
# A pending operation whose outcome has not arrived is re-driven at its permit's
# deadline plus this long times 2^n - 1 after the n-th re-drive.
_OP_REDRIVE_BACKOFF_SEC = 30.0


def _captured_calls(
    step: HarnessResult | None, group: FacadeTurnGroup | None
) -> list[tuple[str, str | None]]:
    """The calls whose request a step's worker holds: the step's digested boundary and
    the digested members of the facade group its turn captured."""
    captures: list[tuple[str, str | None]] = []
    if (
        step is not None
        and (request := step.request) is not None
        and request.request_digest is not None
        and request.call_correlation is not None
    ):
        captures.append((request.call_correlation, request.interface))
    if group is not None:
        captures.extend(
            (member.call_correlation, SEARCH_INTERFACE)
            for member in group.members
            if member.request_digest is not None
        )
    return captures


def _unscrubbed(text: str) -> str:
    return text


def _listing_fields(
    task_id: str,
    record: TaskRecord,
    query: QueryFilter,
    computed: Mapping[str, Callable[[str], Any]],
) -> dict[str, Any]:
    """Read the fields ``query`` names from a record, computing those ``TaskInfo``
    derives from runtime state."""
    return {
        key: computed[key](task_id) if key in computed else getattr(record, key)
        for key in query.terms
    }


class TaskRuntime:
    """In-memory task registry with FIFO-ready queue and dependency tracking."""

    def __init__(
        self,
        workflow_registry: WorkflowRegistry,
        worker_registry: WorkerRegistry,
        orchestration: OrchestrationConfig,
        results: ResultReader,
        logger: logging.Logger,
        credential_vault: CredentialVault,
        feasibility_check: EpisodeFeasibility | None = None,
        surface: PolicySurface | None = None,
        control: ControlPlaneTracer | None = None,
        tracer: Tracer | None = None,
        telemetry: TelemetryConfig | None = None,
        content_scope_authority: Callable[[str, str], None] | None = None,
        n8n: N8nConfig | None = None,
        redrive: Callable[
            [Callable[[str], None], logging.Logger], StoreRedriveScheduler
        ] = StoreRedriveScheduler,
        durability_retry: Callable[
            [Callable[[str], None], logging.Logger], WorkflowRetryScheduler
        ] = _durability_retry,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._worker_registry = worker_registry
        self._logger = logger
        self._results = results
        self._redrive = redrive(self._drive_workflow, logger)
        self._durability = durability_retry(self._retry_durability, logger)
        self._draining = _Drain()
        self._feasibility_check = feasibility_check
        self._policy_surface = surface if surface is not None else PolicySurface()
        self._credential_vault = credential_vault
        self._n8n_credential_password = (n8n or N8nConfig()).credential_password
        self._control = control if control is not None else NULL_CONTROL_TRACER
        self._tracer = tracer
        self._telemetry = telemetry
        self._scope_budget = ScopeBudget.from_config(orchestration)
        self._web_search = orchestration.web_search
        self._max_prepared_input_bytes = orchestration.max_prepared_input_bytes
        self._agent_binding_defaults = _binding_defaults(
            orchestration.agent_binding,
            orchestration.agent_sandbox_enabled,
            orchestration.agent_sandbox_egress_enabled,
        )
        self._lowering_strategy = (
            LoweringStrategy.EPISODE_CUT
            if orchestration.episode_lowering
            else LoweringStrategy.TRANSPARENT
        )
        self._tasks = TaskTable()
        self._original_deps: dict[str, set[str]] = {}
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._rehydrated_dispatched: dict[str, float] = {}
        self._engines: dict[str, OrchestrationEngine] = {}
        # Interface-keyed handlers for a mediated boundary, dispatched off the caller's
        # lane. The model gateway settles the "model" interface; the fabric tool broker
        # executes a fabric-served tool ("search/v1"). Both take the durable envelope.
        self._model_settler: Callable[[ToolInvocationEnvelope], None] | None = None
        self._tool_broker: Callable[[ToolInvocationEnvelope], None] | None = None
        self._failure_reporter: Callable[[TaskEvent], None] | None = None
        # The worker-originated resident path: originate admits and relays the handoff
        # to the origin worker; the ack and outcome handlers consume the worker's fenced
        # transition reports. Set when resident-capacity control is enabled.
        self._resident_originate: Callable[[ToolInvocationEnvelope], bool] | None = None
        self._resident_ack: Callable[[ResidentBootstrapAck], None] | None = None
        self._resident_outcome: Callable[[ResidentOpOutcome], None] | None = None
        self._resident_route_observation: (
            Callable[[ResidentRouteObservation], None] | None
        ) = None
        # Facade boundaries the agent-model gateway captured server-side during an
        # episode's model turn, keyed by task; the completion path reroutes the clean
        # turn-completion into the pending boundary rather than settling it.
        self._pending_facade_groups: dict[str, FacadeTurnGroup] = {}
        # Each held model turn's latest authorization waiting on its workflow's writes,
        # by agent task and call correlation.
        self._parked_turns: dict[tuple[str, str], AuthorizeTurn] = {}
        # Tasks taken off the queue while a report of theirs is being handled.
        self._awaiting_reports: set[str] = set()
        # The tasks whose publication waits for a write of their faulted workflow.
        self._write_faulted: dict[str, set[str]] = {}

        self._dag = StaticDag()
        self._epochs = EpochFrontier()
        self._ready = ReadyQueue(self._epochs, self._dag, self._tasks)
        self._record_failures = RecordFailures(
            self._dag, self._ready, self._tasks, self._failed
        )
        self._mediated_ops = MediatedOperations(
            self._tasks,
            self._engines,
            self._worker_registry,
            self._web_search,
            orchestration.gateway.timeout_sec,
            content_scope_authority,
            self._control,
        )
        self._actions = AfterCommitActions(self._tasks)
        self._resident_tasks = ResidentServeTasks(self._tasks)
        self._content_bindings = ContentBindings(
            self._tasks, self._engines, self._original_deps, self._logger
        )
        self._agent_inputs = AgentInputs(
            self._content_bindings,
            self._redrive,
            orchestration.agent_input_budget_bytes,
        )
        self._episode_dispatch = EpisodeDispatch(
            self._tasks,
            self._engines,
            self._logger,
            self._content_bindings,
            self._agent_inputs,
        )
        self._reservations = WorkerReservations(
            self._tasks, self._episode_dispatch, self._worker_registry, self._logger
        )

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._committer = TransitionCommitter(
            self._epochs,
            self._resident_tasks,
            self._reservations,
            self._actions,
            self._record_failures,
            self._tasks,
            self._original_deps,
            self._engines,
            self._workflow_registry,
            self._control,
            self._logger,
            self._lock,
        )
        self._committer.on_debt = self._durability.schedule
        self._committer.on_fault_cleared = self._release_write_faulted_locked
        self._occurrences = OccurrenceMaterializer(
            self._tasks, self._original_deps, self._committer
        )
        self._merges = TaskMerges(
            self._ready,
            self._dag,
            self._committer,
            self._content_bindings,
            self._tasks,
            self._completed,
            self._failed,
            self._cv,
        )
        self._fence = DispatchFence(
            self._merges,
            self._committer,
            self._ready,
            self._reservations,
            self._episode_dispatch,
            self._tasks,
            self._engines,
        )
        self._inputs = InputChecks(
            self._ready,
            self._committer,
            self._tasks,
            self._results,
            self._redrive,
            self._logger,
            self._cv,
        )

    # ------------------------------------------------------------------ #
    # Registration & submission
    # ------------------------------------------------------------------ #

    def _parse(self, payload: str, format: str) -> ParsedWorkflow:
        return parse_workflow(payload, format, self._n8n_credential_password)

    def validate(
        self, payload: str, format: str = "native"
    ) -> tuple[list[TaskParsingResult], InspectionReport | None]:
        """Parse a submission without executing it into the tasks it registers, and
        for a v2 submission compile its inspection report.

        A v2 workflow's tasks exclude the blueprints its regions materialize work
        from, as a submission does. Structural frontend errors raise
        ``CompileError``; semantic findings ride on the report's diagnostics.
        """
        parsed_workflow = self._parse(payload, format)
        results = [
            TaskParsingResult(
                task_id=entry.task_id,
                graph_node_name=entry.graph_node_name,
                depends_on=entry.depends_on.copy(),
            )
            for entry in parsed_workflow.tasks
        ]
        inspection = self._inspect_parsed(parsed_workflow, payload, format)
        if inspection is not None:
            blueprints = materialized_operators(inspection.template)
            results = [entry for entry in results if entry.task_id not in blueprints]
        return results, inspection

    def _inspect_parsed(
        self, parsed_workflow: ParsedWorkflow, payload: str, format: str
    ) -> InspectionReport | None:
        if not ExecutionMode.is_v2(parsed_workflow.api_version):
            return None
        # A dry run never vaults; drop any inline credential and redact the source so
        # the inspection echoes no credential back.
        mask_inline_credentials(parsed_workflow)
        source = FrontendWorkflowSource.capture(
            redact_source_text(payload, format), format
        )
        return build_inspection(
            new_workflow_id(),
            parsed_workflow,
            source,
            bindings=self._agent_binding_defaults,
            strategy=self._lowering_strategy,
            surface=self._policy_surface,
        )

    async def register(
        self,
        owner_id: str,
        org_id: str,
        payload: str,
        format: str = "native",
        *,
        resident: bool = False,
    ) -> tuple[str, list[TaskParsingResult]]:
        submitted_at = now_iso()
        parsed_workflow = self._parse(payload, format)
        workflow_id = new_workflow_id()
        # Credentials leave the parsed specs before anything persists or compiles, so
        # only an opaque ref reaches the template, the plan and the records.
        credentials = take_inline_credentials(parsed_workflow)
        await self._credential_vault.store_values(workflow_id, credentials.values)
        try:
            staged = self._stage_registration(
                owner_id,
                org_id,
                payload,
                format,
                resident,
                workflow_id,
                parsed_workflow,
                credentials,
            )
        except BaseException:
            self._discard_credentials(workflow_id)
            raise
        try:
            await self._workflow_registry.register_workflow_async(
                workflow_id,
                staged.persisted(),
                WorkflowSched(in_epoch_order=staged.in_epoch_order),
                v2=staged.v2_bundle,
                ledger=(
                    staged.v2_engine.to_snapshot()
                    if staged.v2_engine is not None
                    else None
                ),
                submitted_at=submitted_at,
                blueprints=[PersistedTask(record=r) for r in staged.blueprints],
            )
        except BaseException:
            await self._discard_registration(workflow_id)
            raise
        self._install_registration(workflow_id, staged)
        return workflow_id, staged.results

    async def _discard_registration(self, workflow_id: str) -> None:
        """Remove what a failed submission may have written, and its credentials once
        nothing of the workflow remains; otherwise the startup sweep reclaims them."""
        try:
            await self._workflow_registry.unregister_workflows_async(workflow_id)
        except Exception:
            self._logger.exception(
                "Failed to remove the failed registration of workflow %s",
                workflow_id,
            )
            return
        self._discard_credentials(workflow_id)

    def _discard_credentials(self, workflow_id: str) -> None:
        try:
            self._credential_vault.purge(workflow_id)
        except Exception:
            self._logger.exception(
                "Failed to purge the credentials of unregistered workflow %s",
                workflow_id,
            )

    def _stage_registration(
        self,
        owner_id: str,
        org_id: str,
        payload: str,
        format: str,
        resident: bool,
        workflow_id: str,
        parsed_workflow: ParsedWorkflow,
        credentials: InlineCredentials,
    ) -> _StagedRegistration:
        specs = parsed_workflow.tasks
        yaml_text = redact_source_text(payload, format)
        results: list[TaskParsingResult] = []
        task_records: list[TaskRecord] = []
        graph_task_ids: dict[str, str] = {}

        v2_bundle: PersistedV2Workflow | None = None
        v2_engine: OrchestrationEngine | None = None
        blueprint_ops: frozenset[str] = frozenset()
        materialized: frozenset[str] = frozenset()
        blueprints: list[TaskRecord] = []
        if ExecutionMode.is_v2(parsed_workflow.api_version):
            source = FrontendWorkflowSource.capture(yaml_text, format)
            v2_bundle = compile_bundle(
                workflow_id,
                parsed_workflow,
                source,
                strategy=self._lowering_strategy,
                bindings=self._agent_binding_defaults,
                secret_refs=credentials.model_keys,
                surface=self._policy_surface,
                control=self._control,
            )
            with self._control.workflow_stage(
                ControlPlaneStage.ENGINE_BUILD,
                ControlPlaneWindow.SUBMIT,
                workflow_id,
            ):
                v2_engine = OrchestrationEngine.build(
                    workflow_id,
                    owner_id,
                    org_id,
                    v2_bundle,
                    budget=self._scope_budget,
                    control=self._control,
                    emitter=build_span_emitter(
                        self._tracer, self._telemetry, workflow_id
                    ),
                )
            blueprint_ops = blueprint_operators(v2_bundle.template)
            materialized = materialized_operators(v2_bundle.template)

        in_epoch_order = bool(
            parsed_workflow.schedule_in_epoch_order
            and parsed_workflow.epoch_groups is not None
        )
        depends: dict[str, set[str]] = {}
        merge_keys: dict[str, tuple[str | None, str | None]] = {}
        task_epoch_index: dict[str, int] = {}
        epoch_queue: deque[set[str]] | None = None
        for entry in specs:
            task_id = entry.task_id
            task = entry.task.model_copy(deep=True)
            task_credentials = credentials.tasks.get(task_id, TaskCredentials())
            depends_on = entry.depends_on.copy()

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
                org_id=org_id,
                raw_yaml=yaml_text,
                task=task,
                local_name=entry.local_name,
                graph_node_name=entry.graph_node_name,
                load=entry.load,
                position_in_epoch=entry.position_in_epoch,
                selected_worker=selected_worker,
                task_type=task_type,
                category=category,
                resident=resident,
                credential_refs=task_credentials.refs,
            )
            if task_id in blueprint_ops:
                blueprints.append(record.model_copy(deep=True))
            if task_id in materialized:
                continue
            depends[task_id] = set(depends_on)
            task_records.append(record)
            record.last_queue_ts = record.submitted_ts
            if v2_engine is None:
                # A merged dispatch stores every result under its parent's
                # authorization scope, so only tasks of one scope merge.
                merge_key = task_credentials.merge_key(task.spec, scope=org_id)
                record.merge_key = merge_key
                selected_worker_hint = (
                    record.selected_worker[0]
                    if record.selected_worker and len(record.selected_worker) == 1
                    else None
                )
                merge_keys[task_id] = (merge_key, selected_worker_hint)

            if record.graph_node_name:
                graph_task_ids[record.graph_node_name] = task_id

            results.append(
                TaskParsingResult(
                    task_id=task_id,
                    graph_node_name=entry.graph_node_name,
                    depends_on=depends_on,
                )
            )
        epoch_groups = parsed_workflow.epoch_groups
        if epoch_groups and v2_engine is None:
            queue: deque[set[str]] = deque()
            has_epoch_tasks = False
            for epoch_idx, epoch_nodes in enumerate(epoch_groups):
                epoch_task_ids: set[str] = set()
                for node_name in epoch_nodes:
                    mapped_task_id = graph_task_ids.get(node_name)
                    if mapped_task_id is None:
                        continue
                    epoch_task_ids.add(mapped_task_id)
                    task_epoch_index[mapped_task_id] = epoch_idx
                    has_epoch_tasks = True
                queue.append(epoch_task_ids)
            if has_epoch_tasks:
                epoch_queue = queue

        return _StagedRegistration(
            results=results,
            task_records=task_records,
            depends_on=depends,
            merge_keys=merge_keys,
            task_epoch_index=task_epoch_index,
            epoch_queue=epoch_queue,
            in_epoch_order=in_epoch_order,
            v2_bundle=v2_bundle,
            v2_engine=v2_engine,
            blueprints=blueprints,
        )

    def _install_registration(
        self, workflow_id: str, staged: _StagedRegistration
    ) -> None:
        """Make a durably registered workflow live: install its records and schedule,
        then apply its initial advance."""
        v2_engine = staged.v2_engine
        with self._transition():
            if staged.in_epoch_order:
                self._ready.ready_by_workflow[workflow_id] = []
                self._epochs.workflow_in_epoch_order[workflow_id] = True
            if staged.epoch_queue is not None:
                self._epochs.task_epoch_index.update(staged.task_epoch_index)
                self._epochs.workflow_epoch_tasks[workflow_id] = staged.epoch_queue
                self._epochs.workflow_epoch_frontier[workflow_id] = 0
            self._ready.merge_key_by_task.update(staged.merge_keys)
            candidate_ready: list[str] = []
            for record in staged.task_records:
                task_id = record.task_id
                original = staged.depends_on[task_id]
                self._tasks[task_id] = record
                self._original_deps[task_id] = original
                self._failed.discard(task_id)
                self._completed.discard(task_id)
                # v2 readiness is owned by the orchestration engine; the legacy
                # dependency machinery stays unwired so it cannot admit v2 work.
                if v2_engine is None:
                    pending = {dep for dep in original if dep not in self._completed}
                    self._dag.pending_deps[task_id] = pending
                    for dep in original:
                        self._dag.dependents[dep].add(task_id)
                    if not pending:
                        candidate_ready.append(task_id)

            new_ready = False
            if v2_engine is not None:
                self._engines[workflow_id] = v2_engine
                self._occurrences.install_locked(workflow_id, staged.blueprints)
                with self._control.workflow_stage(
                    ControlPlaneStage.DS_INITIAL_ADVANCE,
                    ControlPlaneWindow.SUBMIT,
                    workflow_id,
                ):
                    if self._apply_advance_locked(
                        workflow_id, v2_engine.initial_advance()
                    ):
                        new_ready = True
                self._committer.save_ledger_locked(workflow_id)
            for task_id in candidate_ready:
                if self._ready.enqueue_ready_locked(task_id):
                    new_ready = True
            if new_ready:
                self._cv.notify_all()

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
        restored: list[str] = []
        for workflow_id in sorted(workflow_ids):
            try:
                stored = await self._load_stored_workflow(workflow_id)
            except Exception as exc:
                if store_unavailable(exc):
                    raise
                tasks, revokes = await self._fail_unrestorable_workflow(
                    workflow_id, exc
                )
                if tasks:
                    with self._transition():
                        self._install_rehydrated_workflow_locked(
                            workflow_id, tasks, None, rehydrated_at
                        )
                        self._actions.file_locked(
                            workflow_id, *revokes, Settled(workflow_id)
                        )
                continue
            if stored is None:
                continue
            tasks, remaining, sched, engine, blueprints = stored
            with self._transition():
                if engine is not None:
                    self._install_rehydrated_v2_workflow_locked(
                        workflow_id,
                        tasks,
                        engine,
                        rehydrated_at,
                        [persisted.record for persisted in blueprints],
                        remaining,
                    )
                else:
                    self._install_rehydrated_workflow_locked(
                        workflow_id, tasks, sched, rehydrated_at
                    )
                # A task a restart found still being cancelled may never have had its
                # interrupt delivered. A workflow whose last task settled just before
                # the crash has no event left to close it: replay the completion
                # notification for every restored workflow, and let the finalizer
                # reject the ones still running or already closed.
                self._actions.file_locked(
                    workflow_id,
                    *(
                        interrupt
                        for _, interrupt in self._actions.cancelling_interrupts_locked(
                            lambda record: record.workflow_id == workflow_id
                        )
                    ),
                    Settled(workflow_id),
                )
                self._cv.notify_all()
            restored.append(workflow_id)
        with self._transition():
            self._merges.restore_merges_locked()
            self._resident_tasks.seed_dispatched_resident_locked()
            self._reservations.seed_held_dispatches_locked()
            live = [
                workflow_id
                for workflow_id in restored
                if not self._committer.workflow_settlement_locked(workflow_id).settled
            ]
        # Rehydrate completes before the API accepts a submission, so no workflow is
        # between vaulting its credentials and registering.
        await self._credential_vault.retain_only(live)
        if restored:
            self._logger.info(
                "Rehydrated %d workflow(s) from durable state", len(restored)
            )
        return len(restored)

    async def _load_stored_workflow(self, workflow_id: str) -> (
        tuple[
            list[PersistedTask],
            set[str],
            WorkflowSched | None,
            OrchestrationEngine | None,
            list[PersistedTask],
        ]
        | None
    ):
        """Read a stored workflow and restore its orchestration engine, without
        installing either: its tasks, remaining set, schedule, engine and task
        blueprints, or None when it holds no task."""
        tasks = await self._stored_tasks(workflow_id)
        if not tasks:
            return None
        await self._vault_stored_credentials(workflow_id, tasks)
        remaining = await self._workflow_registry.get_remaining_tasks_async(workflow_id)
        sched = await self._workflow_registry.load_workflow_sched_async(workflow_id)
        snapshot = await self._workflow_registry.load_ledger_snapshot_async(workflow_id)
        bundle = (
            await self._workflow_registry.get_v2_workflow_async(workflow_id)
            if snapshot is not None
            else None
        )
        if snapshot is None or bundle is None:
            return tasks, remaining, sched, None, []
        blueprints = await self._workflow_registry.load_blueprints_async(workflow_id)
        engine = OrchestrationEngine(
            snapshot,
            bundle,
            budget=self._scope_budget,
            control=self._control,
            emitter=build_span_emitter(self._tracer, self._telemetry, workflow_id),
        )
        return tasks, remaining, sched, engine, blueprints

    async def _fail_unrestorable_workflow(
        self, workflow_id: str, error: Exception
    ) -> tuple[list[PersistedTask], list[Revoke]]:
        """Fail a stored workflow this server cannot read or restore, leaving every
        other workflow to restore.

        Every task of it still open fails with the typed reason, written with that
        reason in one transition; returns the workflow's task records, when they read,
        and the revocation of each dispatch that failing ended.
        """
        reason = f"UnsupportedWorkflowVersion: {type(error).__name__}: {error}"[:500]
        stored = await self._workflow_registry.get_workflow_record_async(workflow_id)
        if stored is not None and stored.control_failure == reason:
            self._logger.warning("Workflow %s stays failed: %s", workflow_id, reason)
        else:
            self._logger.exception(
                "Workflow %s cannot be restored; failing it", workflow_id
            )
        try:
            tasks = await self._stored_tasks(workflow_id)
        except Exception as exc:
            if store_unavailable(exc):
                raise
            tasks = []
        failed: list[PersistedTask] = []
        revokes: list[Revoke] = []
        for persisted in tasks:
            record = persisted.record
            if record.status in TERMINAL_TASK_STATUSES:
                continue
            if record.assigned_worker is not None and (
                revoke := self._actions.revoke_for(
                    record.task_id, record.assigned_worker, record.dispatch_id, None
                )
            ):
                revokes.append(revoke)
            record.status = TaskStatus.FAILED
            record.error = reason
            record.assigned_worker = None
            record.finished_ts = time.time()
            failed.append(persisted)
        await self._workflow_registry.commit_transition_async(
            workflow_id,
            records=failed,
            failed=[persisted.record.task_id for persisted in failed],
            control=WorkflowControl(failure=reason),
        )
        return tasks, revokes

    async def _stored_tasks(self, workflow_id: str) -> list[PersistedTask]:
        """The task records a stored workflow holds, its dynamic ones included."""
        wf_record = await self._workflow_registry.get_workflow_record_async(workflow_id)
        if wf_record is None:
            return []
        dynamic_ids = await self._workflow_registry.get_dynamic_task_ids_async(
            workflow_id
        )
        task_ids = list(dict.fromkeys([*wf_record.task_ids, *sorted(dynamic_ids)]))
        return [
            state
            for state in await self._workflow_registry.load_task_states_async(*task_ids)
            if state
        ]

    async def _vault_stored_credentials(
        self, workflow_id: str, tasks: list[PersistedTask]
    ) -> None:
        """Take the inline credentials out of records stored before they were vaulted.

        A record that can still dispatch has its credentials vaulted and the refs
        recorded, as a submission does; a settled one is masked. Every one gets its
        source redacted again. The vault is written before the records, so a crash in
        between leaves the records to be taken again at the next start. A record that
        cannot be taken is left as stored, and runs with its stored credentials.
        """
        refs = CredentialRefs()
        taken: list[PersistedTask] = []
        for persisted in tasks:
            record = persisted.record
            if record.credential_refs is not None:
                continue
            task = record.task.model_copy(deep=True)
            try:
                live = record.status not in TERMINAL_TASK_STATUSES
                credentials = take_spec_credentials(
                    task.spec, refs if live else CredentialRefs()
                )
                source = redact_stored_source(record.raw_yaml)
            except Exception:
                self._logger.exception(
                    "Task %s keeps its stored credentials; they could not be vaulted",
                    record.task_id,
                )
                continue
            record.task = task
            record.raw_yaml = source
            record.credential_refs = credentials.refs if live else {}
            if record.merge_key is not None:
                record.merge_key = credentials.merge_key(task.spec, scope=record.org_id)
            taken.append(persisted)
        if not taken:
            return
        await self._credential_vault.store_values(workflow_id, refs.values)
        await self._workflow_registry.save_task_states_async(taken)

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
            if record.merge_key is not None:
                # A key persisted under an earlier rule may omit what now keeps two
                # tasks apart, so a restored task's key comes from the task itself.
                record.merge_key = credential_merge_key(
                    record.task.spec, record.credential_refs or {}, scope=record.org_id
                )
            self._ready.merge_key_by_task[task_id] = (
                record.merge_key,
                selected_worker_hint,
            )
            if persisted.epoch_index is not None:
                self._epochs.task_epoch_index[task_id] = persisted.epoch_index
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
                self._dag.dependents[dep].add(task_id)
            if record.status in terminal:
                continue
            # Only completed deps are subtracted, not failed ones: a failure
            # cascade-fails its dependents and persists them FAILED atomically,
            # so a non-terminal task here never has a FAILED dep to clear.
            self._dag.pending_deps[task_id] = {
                dep for dep in original if dep not in self._completed
            }

        if in_epoch_order:
            self._epochs.workflow_in_epoch_order[workflow_id] = True
            self._ready.ready_by_workflow.setdefault(workflow_id, [])
        if epoch_members:
            epoch_queue: deque[set[str]] = deque(
                epoch_members[idx] for idx in sorted(epoch_members) if idx >= frontier
            )
            if epoch_queue:
                self._epochs.workflow_epoch_tasks[workflow_id] = epoch_queue
                self._epochs.workflow_epoch_frontier[workflow_id] = frontier

        for persisted in tasks:
            record = persisted.record
            if record.status != TaskStatus.PENDING:
                continue
            if self._dag.pending_deps.get(record.task_id):
                continue
            self._ready.enqueue_ready_locked(record.task_id)

    def _reconcile_failures_locked(
        self, engine: OrchestrationEngine, tasks: list[PersistedTask]
    ) -> None:
        """Fail what each failed task left standing downstream of it.

        Every failure is walked again, so a stored ledger holding a failure whose
        downstream never settled fails that downstream in the ledger, and one whose
        downstream already failed yields nothing. The next ledger write persists the
        tasks it failed.
        """
        for persisted in tasks:
            if persisted.record.status == TaskStatus.FAILED:
                engine.reconcile_failure(persisted.record.task_id)

    def _reconcile_residual_cancels_locked(
        self, engine: OrchestrationEngine, tasks: list[PersistedTask]
    ) -> None:
        """Cancel each task a crash left behind the cancel of its work item: a pending
        one settles CANCELLED, and a dispatched one goes CANCELLING until its worker's
        terminal settles it."""
        moved: list[str] = []
        for persisted in tasks:
            record = persisted.record
            wi = engine.work_item(record.task_id)
            if wi is None or wi.status is not WorkItemStatus.CANCELLED:
                continue
            if self._cancel_record_locked(record, _RESIDUAL_CANCEL_REASON) is None:
                continue
            record.residual_cancel = True
            moved.append(record.task_id)
        if moved:
            self._committer.commit_locked(*moved)

    def _install_rehydrated_v2_workflow_locked(
        self,
        workflow_id: str,
        tasks: list[PersistedTask],
        engine: OrchestrationEngine,
        rehydrated_at: float,
        blueprints: list[TaskRecord],
        remaining: set[str],
    ) -> None:
        """Rebuild a v2 workflow: restore the engine and re-admit ready work items.

        The legacy dependency machinery stays unwired; the orchestration engine is the
        readiness authority. Terminal task facts reconcile the engine idempotently, so a
        crash between a task's terminal write and its ledger snapshot never loses a
        settlement and never duplicates a publication or effect receipt. A workflow
        stored with its child templates as tasks has them read as blueprints and
        withdrawn from its remaining tasks.
        """
        materialized = materialized_operators(engine.template)
        prototypes = [p.record for p in tasks if p.record.task_id in materialized]
        tasks = [p for p in tasks if p.record.task_id not in materialized]
        # A workflow holding no blueprints makes a recursive agent's children from the
        # agent's own root task.
        covered = {record.task_id for record in [*prototypes, *blueprints]}
        roots = [
            p.record
            for p in tasks
            if p.record.task_id in blueprint_operators(engine.template)
            and p.record.task_id not in covered
        ]
        self._occurrences.install_locked(
            workflow_id, [*prototypes, *blueprints, *roots]
        )
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

        self._engines[workflow_id] = engine
        cancelled = False
        for persisted in tasks:
            record = persisted.record
            if record.status == TaskStatus.DONE:
                # A dead route's record settles from its branch decision, restored or
                # made again, and never as a success.
                if record.result_skip == ROUTE_NOT_TAKEN:
                    continue
                engine.on_succeeded(
                    record.task_id,
                    empty=record.result_skip is not None,
                    content=record.result_reference,
                )
            elif record.status == TaskStatus.FAILED:
                engine.on_failed(
                    record.task_id, record.error or "task failed", retryable=False
                )
            elif (
                record.status in (TaskStatus.CANCELLED, TaskStatus.CANCELLING)
                and not record.residual_cancel
            ):
                cancelled = True
        # A dispatch a crash recorded on its task before the ledger saved it: record it
        # now, so its loss or give-up resolves as any other dispatch's does.
        for persisted in tasks:
            record = persisted.record
            if record.status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                self._catch_up_dispatch_locked(engine, record)
        # Replay cancellation after the settled facts so a settled outcome is never
        # overwritten; a cancelled workflow is then never re-admitted below. A record
        # left CANCELLING carries the cancel just as a settled one does: a crash between
        # the cancelled task write and the ledger save would otherwise restore a
        # workflow the cancel never reached. A child its region's residual policy
        # cancelled is no workflow cancel.
        if cancelled:
            engine.cancel_instance()
        else:
            self._reconcile_failures_locked(engine, tasks)
            engine.fail_undeliverable_region_inputs()
            self._reconcile_residual_cancels_locked(engine, tasks)
        # A crash can beat the settle of a cancelled episode suspended on a boundary,
        # which no worker terminal settles.
        self._settle_suspended_cancels_locked(engine)
        # A boundary invocation of a durably settled task is terminal, even when a crash
        # beat the ledger save that recorded it: the save below makes it durable, and
        # the startup reconcile then releases the credit it holds.
        engine.terminalize_unsettled_invocations(
            persisted.record.task_id
            for persisted in tasks
            if persisted.record.status in SETTLING_TASK_STATUSES
        )

        # Re-derive readiness for every PENDING task from the engine rather than
        # trusting the cached work-item status: a crash mid-retry can leave a task
        # PENDING while the snapshot still shows its work item in flight, and only a
        # re-derivation re-admits it instead of orphaning the workflow.
        for persisted in tasks:
            record = persisted.record
            if record.status != TaskStatus.PENDING:
                continue
            if engine.reconcile_pending(record.task_id):
                self._ready.enqueue_ready_locked(record.task_id)
            elif (worker_id := engine.suspending_worker(record.task_id)) is not None:
                # A crash beat the ledger save of a boundary's settle, which had
                # already returned the record to PENDING. The re-issued boundary
                # reaches the worker that captured it through the record.
                record.status = TaskStatus.DISPATCHED
                record.assigned_worker = worker_id
                self._rehydrated_dispatched[record.task_id] = rehydrated_at
                self._committer.commit_locked(record.task_id)

        self._repair_work_records_locked(workflow_id, engine)

        # Re-drive any DONE producer whose spawn never sealed and any agent waiting on
        # its bound inputs: their terminal events do not replay, so nothing else
        # materializes the children or records the inputs.
        if engine.blocked_input_agents() or any(
            persisted.record.status == TaskStatus.DONE
            and (spawn_op := engine.fanout_spawn(persisted.record.task_id)) is not None
            and engine.spawn_awaits_children(spawn_op)
            for persisted in tasks
        ):
            self._redrive.drive_now(workflow_id)

        # Re-issue the off-lane dispatch for any mediated boundary suspended with no
        # durable outcome: the handler ran on an in-memory executor a crash discarded,
        # so nothing else resumes the agent. The durable envelope routes it to the same
        # handler (a search to the broker, a model to the gateway) it was recorded for.
        self._actions.file_locked(
            workflow_id, *map(_issue, engine.pending_tool_dispatches())
        )
        if engine.awaits_control_reads():
            self._redrive.drive_now(workflow_id)
        if withdrawn := sorted(p.task_id for p in prototypes if p.task_id in remaining):
            self._committer.retire_locked(workflow_id, withdrawn)
        self._committer.save_ledger_locked(workflow_id)
        # A branch or loop with no runnable contract may already have run, so nothing
        # re-evaluates it: a workflow still running through one fails.
        if (
            legacy := engine.legacy_control_regions()
        ) and not self._committer.workflow_settlement_locked(workflow_id).settled:
            self._fail_workflow_locked(
                workflow_id,
                legacy_control_unsupported(legacy[0], "a runnable contract"),
            )

    def _repair_work_records_locked(
        self, workflow_id: str, engine: OrchestrationEngine
    ) -> None:
        """Bring the task records in line with the work the restored ledger holds: an
        unsettled work item with no record gets one, ready to run when its work item
        is, and a pending record whose work item no route reaches settles skipped."""
        skipped: list[str] = []
        for wi in engine.task_work_items():
            record = self._tasks.get(wi.legacy_task_id)
            if (
                record is None
                and materialized_work(wi)
                and wi.status not in TERMINAL_WORK_ITEM_STATUSES
            ):
                if not self._occurrences.register_locked(
                    workflow_id, wi.legacy_task_id, wi.operator_id
                ):
                    self._fail_workflow_locked(
                        workflow_id, blueprint_missing(wi.operator_id)
                    )
                    return
                if wi.status is WorkItemStatus.READY:
                    self._ready.enqueue_ready_locked(wi.legacy_task_id)
            elif (
                record is not None
                and record.status == TaskStatus.PENDING
                and wi.status is WorkItemStatus.SKIPPED
            ):
                skipped.append(wi.legacy_task_id)
        self._skip_dead_routes_locked(skipped)

    def _catch_up_dispatch_locked(
        self, engine: OrchestrationEngine, record: TaskRecord
    ) -> None:
        task_id, worker_id = record.task_id, record.assigned_worker
        if self.prepares_inputs(task_id):
            if engine.input_preparation(task_id) is None:
                engine.on_input_preparation_dispatched(task_id, worker_id)
        elif (wi := engine.work_item(task_id)) is not None and (
            wi.status is WorkItemStatus.READY
        ):
            # Only a lost save leaves a dispatched task's work item READY; a BLOCKED
            # one's dispatch ended at a suspension on a boundary.
            engine.on_dispatched(task_id, worker_id)

    # ------------------------------------------------------------------ #
    # Transitions and what they owe once durable
    # ------------------------------------------------------------------ #

    @contextmanager
    def _transition(self, *, locked: bool = True) -> Iterator[None]:
        """Run a transition, then deliver off the lock what it owes once durable.

        The transition runs under the lock, or, when not ``locked``, takes the lock
        itself, so it can read off the lock first. A write the store does not take is
        held for a retry and the transition completes in memory. A runtime entry an
        acknowledging caller makes raises ``TransitionNotDurable`` at its exit while a
        write its handling made is held, before the caller acts on its answer.
        """
        if outermost := self._committer.enter_scope():
            self._actions.open_scope()
        refused: dict[str, BaseException] = {}
        try:
            try:
                with self._cv if locked else nullcontext():
                    try:
                        yield
                    except BaseException as exc:
                        # A refusal alone stopped after its writes, not partway.
                        if outermost and not (
                            isinstance(exc, TransitionNotDurable)
                            and not exc.stopped_partway
                        ):
                            with self._cv:
                                # A transition that stopped partway may have changed
                                # state no write carried; all of it is owed before any
                                # action.
                                for workflow_id in list(self._actions.parked):
                                    self._committer.mark_dirty_locked(workflow_id)
                        raise
                    finally:
                        if outermost:
                            with self._cv:
                                self._actions.close_scope_locked()
                                self._actions.release_locked(self._committer.durable)
            finally:
                held, refused = self._committer.exit_scope()
                if outermost:
                    self._act_after_commit()
        except TransitionNotDurable:
            raise
        except Exception as exc:
            if refused:
                raise TransitionNotDurable(refused, stopped_partway=True) from exc
            raise
        if held:
            self._logger.warning(
                "Writes of workflow(s) %s are held for a retry: %s",
                ", ".join(sorted(held)),
                next(iter(held.values())),
            )
        if refused:
            raise TransitionNotDurable(refused)

    @contextmanager
    def transition(self) -> Iterator[None]:
        """Run several runtime calls as one transition: what any of them owes is
        delivered once all of them have committed, whichever thread delivers it."""
        with self._transition(locked=False):
            yield

    @contextmanager
    def acknowledging(self) -> Iterator[None]:
        """Run the handling of a request its caller acknowledges, as one transition.

        A runtime entry the handling makes raises ``TransitionNotDurable`` at its exit
        while a write the handling made is held, and so does the handling itself when
        it ends with one held, whatever else it raised, so the caller leaves the
        request unacknowledged and handles it again. A task whose report the handling
        took up is not dispatched again until the handling ends.
        """
        outermost = self._committer.open_acknowledging()
        acknowledged = False
        try:
            try:
                with self._transition(locked=False):
                    yield
            except TransitionNotDurable:
                raise
            except Exception as exc:
                if outermost and (held := self._committer.close_acknowledging()):
                    raise TransitionNotDurable(held, stopped_partway=True) from exc
                raise
            if outermost and (held := self._committer.close_acknowledging()):
                raise TransitionNotDurable(held)
            acknowledged = True
        finally:
            if outermost:
                self._committer.close_acknowledging()
                self._end_reports(acknowledged)

    def _end_reports(self, acknowledged: bool) -> None:
        """Release the reports this thread's handling took up, and queue again each
        task whose publication waited for them."""
        with self._transition():
            for task_id in self._committer.end_reports_locked(acknowledged):
                if task_id in self._awaiting_reports:
                    self._awaiting_reports.discard(task_id)
                    if (record := self._tasks.get(task_id)) is not None and (
                        record.status == TaskStatus.PENDING
                    ):
                        self._return_dispatch_locked(
                            record, increment_retry=False, front=True
                        )
                        self._cv.notify_all()

    def _release_write_faulted_locked(self, workflow_id: str) -> None:
        """Queue again each task whose publication waited for a write of the
        workflow."""
        for task_id in sorted(self._write_faulted.pop(workflow_id, ())):
            if (record := self._tasks.get(task_id)) is not None and (
                record.status == TaskStatus.PENDING
            ):
                self._return_dispatch_locked(record, increment_retry=False, front=True)
                self._cv.notify_all()

    def _file_locked(self, task_id: str, *actions: AfterCommit | None) -> None:
        """Hold actions until the transition of the task's workflow commits."""
        present = [action for action in actions if action is not None]
        if (record := self._tasks.get(task_id)) is not None:
            self._actions.file_locked(record.workflow_id, *present)
        else:
            self._actions.queue_locked(*present)

    def _act_after_commit(self) -> None:
        """Deliver every committed action and release each worker whose dispatch
        ended, off the lock.

        A delivery that files actions of its own leaves them to this drain.
        """
        if self._draining.active:
            return
        self._draining.active = True
        try:
            while True:
                with self._lock:
                    ready = self._actions.take_ready()
                    ended = self._reservations.take_ended()
                if failed := self._reservations.release_workers(ended):
                    with self._lock:
                        self._reservations.requeue_ended(failed)
                if not ready:
                    return
                for workflow_id, action in ready:
                    try:
                        self._deliver(workflow_id, action)
                    except Exception:
                        self._logger.exception(
                            "Delivering %s failed", type(action).__name__
                        )
                        self._retain(workflow_id, action)
        finally:
            self._draining.active = False

    def _retain(self, workflow_id: str | None, action: AfterCommit) -> None:
        """Keep an action whose delivery failed for its workflow's retry.

        An action filed for no workflow concerns a task the runtime no longer holds,
        such as the revoke of a dispatch control does not hold, which the worker's next
        report of itself busy on that dispatch revokes again, so a failed one is
        dropped.
        """
        if workflow_id is None:
            return
        with self._lock:
            self._actions.retain_failed_locked(workflow_id, action)
        self._durability.schedule(workflow_id)

    def _deliver(self, workflow_id: str | None, action: AfterCommit) -> None:
        match action:
            case CreditRelease(invocation_id=invocation_id, failed=failed):
                consumed = self._mediated_ops.release_resident_credit(
                    invocation_id, failed=failed
                )
                if consumed is not None:
                    consumed.add_done_callback(
                        lambda done: self._credit_consumed(workflow_id, action, done)
                    )
            case Reap(worker_id=worker_id, task_id=task_id, call_correlation=call):
                if action.resident:
                    self._mediated_ops.relay_resident_reap(worker_id, task_id, call)
                else:
                    self._mediated_ops.reap_mediated_op(worker_id, task_id, call)
            case Interrupt(message=interrupt):
                worker = self._worker_registry.get_worker(interrupt.worker_id)
                if worker is None:
                    self._logger.warning(
                        "Cannot publish interrupt for %s; worker %s missing",
                        interrupt.task_id,
                        interrupt.worker_id,
                    )
                else:
                    self._worker_registry.publish_interrupt(worker, interrupt)
            case Revoke(message=message, node_id=node_id):
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
            case Issue():
                self._issue(action)
            case Purge(workflow_id=purged, settled_only=settled_only):
                with self._lock:
                    if (
                        settled_only
                        and not self._committer.workflow_settlement_locked(
                            purged
                        ).settled
                    ):
                        return
                self._credential_vault.purge(purged)
            case Settled(workflow_id=settled):
                self._committer.notify_terminal_transition(settled)
            case AuthorizeTurn(proposal=proposal, proposer_id=proposer_id):
                self._answer_model_turn(proposal, proposer_id, action)

    def _credit_consumed(
        self, workflow_id: str | None, action: AfterCommit, consumed: Future[Any]
    ) -> None:
        if consumed.cancelled():
            error: BaseException | str | None = "cancelled"
        elif (error := consumed.exception()) is None:
            return
        self._logger.warning(
            "Releasing a resident credit failed; keeping it for a retry: %s", error
        )
        self._retain(workflow_id, action)

    def _retry_durability(self, workflow_id: str) -> None:
        """Make a workflow's held writes durable and deliver what failed to deliver,
        keeping the workflow scheduled while either remains."""
        with self._transition():
            self._committer.close_locked(workflow_id)
            self._actions.retry_failed_locked(workflow_id)
        with self._lock:
            pending = not self._committer.durable(
                workflow_id
            ) or self._actions.has_failed(workflow_id)
        if pending:
            self._durability.schedule(workflow_id)

    # ------------------------------------------------------------------ #
    # Worker reservations and revokes
    # ------------------------------------------------------------------ #

    def release_ended_reservations(self) -> None:
        """Release every worker reserved for a dispatch not in flight, such as one
        whose task settled just before a restart, and revoke the dispatch."""
        ended = [
            reservation
            for reservation in self._worker_registry.reservations()
            if not self.dispatch_in_flight(
                reservation.task_id, reservation.dispatch_id, reservation.worker_id
            )
        ]
        with self._transition():
            self._actions.queue_locked(
                *(
                    revoke
                    for r in ended
                    if (
                        revoke := self._actions.revoke_for(
                            r.task_id, r.worker_id, r.dispatch_id, None
                        )
                    )
                )
            )
        self._reservations.release_workers(
            [(r.worker_id, r.dispatch_id) for r in ended]
        )

    def revoke_dispatch(self, task_id: str, worker_id: str, dispatch_id: str) -> None:
        """Revoke one dispatch on its worker, which control does not hold."""
        with self._transition():
            self._actions.queue_locked(
                *filter(
                    None,
                    [self._actions.revoke_for(task_id, worker_id, dispatch_id, None)],
                )
            )

    # ------------------------------------------------------------------ #
    # Ready queue helpers
    # ------------------------------------------------------------------ #

    def _fail_later_epochs_locked(
        self,
        workflow_id: str,
        failed_epoch: int,
        reason: str,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Fail the pending tasks of every epoch after ``failed_epoch``.

        Returns the failed tasks with their reason, and the merged children of any of
        them returned to the queue.
        """
        epoch_tasks = self._epochs.workflow_epoch_tasks.get(workflow_id)
        if not epoch_tasks:
            return [], []
        frontier = self._epochs.workflow_epoch_frontier[workflow_id]

        impacted: list[tuple[str, str]] = []
        returned: list[str] = []
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
                self._dag.forget_pending(task_id)
                self._ready.remove_from_ready_locked(task_id)
                self._ready.merge_bucket_remove(task_id)
                self._ready.forget_merge_key(task_id)
                returned += self._merges.return_merged_children_locked(
                    self._merges.take_children(task_id), unmerge=True
                )
                impacted.append((task_id, reason))

        return impacted, returned

    def next_ready(
        self, stop_event: threading.Event, timeout: float = 1.0
    ) -> str | None:
        """
        Block until a task is ready or stop_event is set. Returns a task_id or None when
        stopping.
        """
        with self._cv:
            while not stop_event.is_set():
                task_id = self._ready.pop_ready_locked()
                if task_id:
                    return task_id
                self._cv.wait(timeout)
            return None

    # ------------------------------------------------------------------ #
    # v2 orchestration ledger (`DS`)
    # ------------------------------------------------------------------ #

    def is_v2_workflow(self, workflow_id: str) -> bool:
        with self._lock:
            return workflow_id in self._engines

    def orchestration_engine(self, workflow_id: str) -> OrchestrationEngine | None:
        with self._lock:
            return self._engines.get(workflow_id)

    def dispatch_traceparent(self, task_id: str) -> str | None:
        """The ``traceparent`` a dispatched task's worker-side span should parent on.

        Read-only: names the task's episode (work item) span without touching the
        ledger. Names nothing (``None``) rather than a synthesized-anyway value when
        telemetry is off, the task has no v2 work item, or the workflow isn't a v2
        one, so the dispatcher can omit the wire field entirely instead of carrying a
        placeholder.
        """
        if not self._control.enabled:
            return None
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            work_item_id = engine.work_item_id_for_task(task_id) if engine else None
        if record is None or work_item_id is None:
            return None
        return format_traceparent(
            workflow_to_trace_id_int(record.workflow_id),
            derived_span_id(SpanIdKind.WORK_ITEM, work_item_id),
        )

    def apply_boundary_event(self, task_id: str, event: BoundaryEvent) -> bool:
        """Carry an episode's boundary event into the ledger and dispatch its effect.

        Routes the event into the engine, synthesizes a task record for any dispatchable
        child it materializes, applies the advance, and writes the ledger snapshot after
        the task records so the ledger never leads durable state.
        """
        with self._transition():
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return False
            advance = engine.route_boundary_event(task_id, event)
            changed = self._apply_advance_locked(record.workflow_id, advance)
            self._committer.save_ledger_locked(record.workflow_id)
            self._committer.settle_if_done_locked(record.workflow_id)
            if changed:
                self._cv.notify_all()
            return changed

    def _apply_episode_step_locked(
        self, task_id: str, hr: HarnessResult, group: FacadeTurnGroup | None = None
    ) -> None:
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
        worker_id = record.assigned_worker
        captures = self._mediated_ops.reap_captures_locked(
            worker_id, task_id, _captured_calls(hr, group)
        )
        if record.status == TaskStatus.CANCELLING:
            self._file_locked(task_id, *captures)
            self._settle_cancelled_locked(record, time.time())
            return
        if hr.kind in (HarnessResultKind.FAILURE, HarnessResultKind.CANCELLATION):
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
            self._file_locked(task_id, *captures)
            self._committer.save_ledger_locked(record.workflow_id)
            self._committer.settle_if_done_locked(record.workflow_id)
            return
        request = hr.request
        if request is None:
            return
        capsule = hr.capsule.blob if hr.capsule is not None else None
        wi = engine.work_item(task_id)
        if capsule is not None:
            engine.record_continuation(task_id, capsule)
        # The outcome that drove this step was consumed by its dispatch; clear it so a
        # later step never re-injects it.
        engine.mark_pending_outcome(task_id, None)
        event = to_boundary_event(request, continuation=capsule)
        advance = engine.route_boundary_event(task_id, event)
        changed = self._apply_advance_locked(record.workflow_id, advance)
        corr = request.call_correlation
        env = (
            engine.boundary_envelope(wi.activation_id, corr)
            if wi is not None and corr is not None
            else None
        )
        handled = False
        if corr is not None and env is not None and env.denial is not None:
            if request.request_digest is not None:
                self._file_locked(
                    task_id,
                    self._mediated_ops.reap_captured_request_locked(
                        worker_id, task_id, corr, request.interface
                    ),
                )
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
                self._file_locked(task_id, _issue(envelope))
                handled = True
        if not handled:
            # A blocked boundary no branch re-readied or handed off (a denial lacking a
            # correlation) must not hang the episode; re-ready it so it can continue.
            stuck = engine.work_item(task_id)
            if stuck is not None and stuck.status is WorkItemStatus.BLOCKED:
                self._reenqueue_episode_locked(task_id)
                changed = True
        self._committer.save_ledger_locked(record.workflow_id)
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
        if capsule is not None:
            engine.record_continuation(task_id, capsule.blob)
        engine.mark_pending_outcome(task_id, None)
        advance = engine.route_facade_turn_group(task_id, group)
        self._apply_advance_locked(record.workflow_id, advance)
        cap = self._web_search.max_parallel
        for index, envelope in enumerate(
            engine.group_dispatch_envelopes(task_id, group.group_id)
        ):
            if index < cap:
                self._file_locked(task_id, _issue(envelope))
            else:
                overflow = ToolOutcome(
                    status=ToolOutcomeStatus.QUOTA,
                    value=f"web search parallel cap ({cap}) exceeded this turn",
                )
                self._settle_episode_invocation(
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
            self._committer.persist_locked(task_id)
        self._committer.save_ledger_locked(record.workflow_id)
        self._cv.notify_all()

    def _reenqueue_episode_locked(self, task_id: str) -> None:
        """Re-ready a still-running agent episode for its next run-to-yield step."""
        record = self._tasks.get(task_id)
        if record is None or record.status in TERMINAL_TASK_STATUSES:
            return
        if record.status == TaskStatus.CANCELLING:
            self._settle_cancelled_locked(record, time.time())
            return
        # Settle the finished attempt so a long continuing episode's history stays
        # bounded (a no-op when a suspend already closed it).
        if engine := self._engines.get(record.workflow_id):
            engine.close_latest_attempt(task_id)
        self._release_dispatch_locked(record, [task_id], front=True)

    def settle_episode_invocation(
        self,
        task_id: str,
        call_correlation: str,
        value: str | None = None,
        *,
        error: str | None = None,
        ref: OutcomeManifest | None = None,
    ) -> bool:
        """Settle a suspended model/effect boundary and re-ready its episode.

        An inline ``value`` or a reference-backed ``ref`` manifest lands durably on the
        boundary envelope and the work item returns to READY for a re-dispatch that
        injects it at its originating call. An upstream ``error`` instead fails the
        boundary terminally, so a gateway failure never resumes the agent as a phantom
        empty success.
        """
        with self._transition():
            return self._settle_episode_invocation(
                task_id, call_correlation, value, error=error, ref=ref
            )

    def _settle_episode_invocation(
        self,
        task_id: str,
        call_correlation: str,
        value: str | None = None,
        *,
        error: str | None = None,
        ref: OutcomeManifest | None = None,
    ) -> bool:
        with self._cv:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return False
            if not engine.boundary_settleable(task_id, call_correlation):
                # Absorbing: the exact boundary is already resolved, or its work item is
                # no longer BLOCKED because a durable cancellation made it CANCELLED. A
                # late or duplicate tool/model/resident delivery is audit evidence, so
                # it must not settle or fail the boundary, terminalize its invocation,
                # re-ready the episode, or release the resident credit a second time
                # (cancellation already terminalized and released it). This precedes the
                # credit hook so a late success cannot win the claim's terminal reason.
                # A retry absorbed here still makes the first settle durable.
                self._committer.close_locked(record.workflow_id)
                return False
            # A resident request's reap follows its credit release; see the resident
            # terminal hook.
            env = engine.pending_tool_dispatch(task_id, call_correlation)
            captured_on = (
                record.assigned_worker
                if env is not None
                and env.request_digest is not None
                and not (
                    env.interface == MODEL_INTERFACE
                    and engine.service_dependency(task_id) is not None
                )
                else None
            )
            if error is not None:
                advance = engine.on_failed(
                    task_id, f"agent boundary failed: {error}", retryable=False
                )
                invocation_id = engine.terminalize_boundary_invocation(
                    task_id, call_correlation
                )
                changed = self._apply_advance_locked(record.workflow_id, advance)
                # A fenced failure terminal releases the resident credit just as a
                # completion does; nothing else may release an accepted credit.
                self._file_settled_boundary_locked(
                    task_id, call_correlation, invocation_id, captured_on, failed=True
                )
                self._committer.save_ledger_locked(record.workflow_id)
                self._committer.settle_if_done_locked(record.workflow_id)
                if changed:
                    self._cv.notify_all()
                return changed
            advance = engine.settle_boundary_outcome(
                task_id, call_correlation, value=value, ref=ref
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
            self._file_settled_boundary_locked(
                task_id, call_correlation, invocation_id, captured_on, failed=False
            )
            self._committer.save_ledger_locked(record.workflow_id)
            self._cv.notify_all()
            return True

    def _file_settled_boundary_locked(
        self,
        task_id: str,
        call_correlation: str,
        invocation_id: str | None,
        captured_on: str | None,
        *,
        failed: bool,
    ) -> None:
        """Owe a settled boundary's credit release and the reap of its request."""
        self._file_locked(
            task_id,
            (
                CreditRelease(invocation_id, failed)
                if invocation_id is not None
                else None
            ),
            (
                Reap(captured_on, task_id, call_correlation)
                if captured_on is not None
                else None
            ),
        )

    def redispatch_episode_invocation(
        self, task_id: str, call_correlation: str
    ) -> bool:
        """Re-issue a still-pending mediated boundary off-lane without settling it.

        A transient or uncertain delivery loss holds the boundary pending and any linked
        credit uncertain, then re-drives the same durable envelope to its handler — the
        path a restart uses — so the invocation resumes under its held claim rather than
        terminalizing. A boundary that already settled, terminalized, or cancelled is a
        no-op, so a late re-drive neither re-runs the handler nor releases a credit.
        """
        with self._transition():
            handoff = self._prepare_issue_locked(task_id, call_correlation, None)
        if handoff is None:
            return False
        handoff()
        return True

    def _issue(self, issue: Issue) -> None:
        """Hand a recorded boundary to its handler, if it still awaits the invocation
        it was recorded with."""
        with self._transition():
            handoff = self._prepare_issue_locked(
                issue.task_id, issue.call_correlation, issue.invocation_id
            )
        if handoff is not None:
            handoff()

    def _prepare_issue_locked(
        self, task_id: str, call_correlation: str, invocation_id: str | None
    ) -> Callable[[], Any] | None:
        """Prepare the handoff of a still-pending boundary to its handler; with an
        ``invocation_id``, only under that invocation. For a boundary that is not
        pending, make its workflow's held writes durable."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if record is None or engine is None:
            return None
        envelope = (
            engine.pending_tool_dispatch(task_id, call_correlation)
            if record.status not in TERMINAL_TASK_STATUSES
            else None
        )
        if envelope is None or (
            invocation_id is not None and envelope.invocation_id != invocation_id
        ):
            self._committer.close_locked(record.workflow_id)
            return None
        return self._boundary_handoff_locked(envelope)

    def redeliver_to_worker(self, worker_id: str) -> None:
        """Re-relay what control holds for a worker whose task stream attached.

        A frame relayed while the worker had no stream attached may be lost. Each
        pending mediated operation the worker originated is re-minted, and the worker
        drops one it already runs; each task being cancelled there is interrupted
        again, keyed to its dispatch. A lost task dispatch resolves as lost.
        """
        with self._transition():
            pending = self._mediated_ops.pending_for_worker(worker_id)
            for workflow_id, interrupt in self._actions.cancelling_interrupts_locked(
                lambda record: record.assigned_worker == worker_id
            ):
                self._actions.file_locked(workflow_id, interrupt)
        for task_id, call in sorted(pending):
            self.redispatch_episode_invocation(task_id, call)

    def redrive_overdue_ops(self, worker_id: str) -> None:
        """Re-drive each operation a live worker originated whose outcome is overdue.

        A first delivery can be lost or left ambiguous, as when the egress's outcome
        could not be finalized while the root restarted, so an overdue operation is
        re-minted under the same idempotency key, and a worker that already produced
        its outcome returns it. A boundary re-driven _OP_REDRIVE_LIMIT times fails.
        """
        now = time.time()
        with self._cv:
            exhausted, redrive = self._mediated_ops.take_overdue(worker_id, now)
        with self._transition():
            for op in exhausted:
                self._logger.warning(
                    "No outcome arrived for tool operation %s of %s after %d "
                    "re-drives; failing its boundary",
                    op.call_correlation,
                    op.agent_task_id,
                    op.redrives,
                )
                self._settle_episode_invocation(
                    op.agent_task_id,
                    op.call_correlation,
                    error="tool operation outcome never arrived",
                )
        for permit_id, op in redrive:
            if not self.redispatch_episode_invocation(
                op.agent_task_id, op.call_correlation
            ):
                # Its boundary has settled or ended, so no outcome will reap the
                # operation.
                with self._cv:
                    self._mediated_ops.discard_op(permit_id, op)

    def _boundary_handoff_locked(
        self, env: ToolInvocationEnvelope
    ) -> Callable[[], Any] | None:
        """Prepare a recorded mediated boundary's handoff to its handler by exact
        (kind, interface), to run off the lock.

        A recorded request digest marks a boundary the worker originated and holds its
        raw request privately: an external ``model`` runs off-lane on that worker's
        egress sidecar, while a ``resident`` ``model`` originates the claim-gated
        resident path from the worker. A ``model`` boundary without a digest reaches the
        model gateway (a ``canned``/``echo`` binding); a ``search/v1`` boundary with no
        digest is the gateway-captured facade path and reaches the fabric broker. An
        unrecognized interface is a fabric misconfiguration terminalized as a typed
        unavailable outcome — never a silent fall-through to the model settler.
        """
        if (
            env.kind is BoundaryEventKind.INVOCATION
            and env.interface == MODEL_INTERFACE
        ):
            if env.request_digest is not None:
                engine = self._engines.get(self._tasks[env.task_id].workflow_id)
                if engine is not None and engine.service_dependency(env.task_id):
                    return lambda: self._originate_resident(env)
                return self._worker_originated_handoff_locked(env)
            if (settler := self._model_settler) is not None:
                return lambda: settler(env)
            return None
        if (
            env.kind is BoundaryEventKind.INVOCATION
            and env.interface == SEARCH_INTERFACE
        ):
            if env.request_digest is not None:
                return self._worker_originated_handoff_locked(env)
            if (broker := self._tool_broker) is not None:
                return lambda: broker(env)
            return None
        outcome = ToolOutcome(
            status=ToolOutcomeStatus.UNAVAILABLE,
            value=f"no fabric handler for interface {env.interface!r}",
        )
        self._settle_episode_invocation(
            env.task_id, env.call_correlation, outcome.model_dump_json()
        )
        return None

    def _originate_resident(self, env: ToolInvocationEnvelope) -> None:
        """Originate a worker-captured resident boundary through resident admission."""
        if self._resident_originate is None:
            error = "resident-capacity control is not enabled"
        elif not self._resident_originate(env):
            error = "resident-capacity control is not running"
        else:
            return
        with self._transition():
            worker_id = self._assigned_worker_locked(env.task_id)
            self._settle_episode_invocation(
                env.task_id, env.call_correlation, error=error
            )
            if worker_id:
                self._file_locked(
                    env.task_id,
                    Reap(worker_id, env.task_id, env.call_correlation, resident=True),
                )

    def on_resident_bootstrap_ack(self, ack: ResidentBootstrapAck) -> None:
        """Consume an origin worker's resident bootstrap-phase report."""
        if self._resident_ack is not None:
            self._resident_ack(ack)

    def on_resident_outcome(self, outcome: ResidentOpOutcome) -> None:
        """Consume an origin worker's fenced resident terminal outcome report."""
        if self._resident_outcome is not None:
            self._resident_outcome(outcome)

    def on_resident_route_observation(
        self, observation: ResidentRouteObservation
    ) -> None:
        """Consume an origin's classified path evidence for the reachability view."""
        if self._resident_route_observation is not None:
            self._resident_route_observation(observation)

    def _resolve_op_credential(
        self, agent: TaskRecord, interface: str
    ) -> _OpCredential | _MissingCredential:
        """The provider credential authority a permit for this interface carries.

        Other interfaces than the model read their provider key locally.
        """
        if interface != MODEL_INTERFACE:
            return _OpCredential()
        return self._model_credential(agent, self.resolve_model_binding(agent.task_id))

    def _model_credential(
        self, agent: TaskRecord, binding: AgentModelGatewayBinding | None
    ) -> _OpCredential | _MissingCredential:
        """The credential authority a model permit carries for an agent's binding.

        A binding that pins its own key resolves it from the vault so it rides the
        one-use permit down to the egressing worker, and a pinned key missing from the
        vault denies the operation. A binding without one may use the worker's
        deployment key only when its URL is the deployment's current default model URL.
        """
        if binding is None:
            return _OpCredential()
        if binding.secret_ref is not None:
            secret = self._credential_vault.resolve(
                agent.workflow_id, binding.secret_ref
            )
            if secret is None:
                return _MissingCredential()
            return _OpCredential(credential=secret.get_secret_value())
        return _OpCredential(
            deployment_credential=_is_default_url(
                binding.url, self._agent_binding_defaults.default_url
            )
        )

    def _worker_originated_handoff_locked(
        self, env: ToolInvocationEnvelope
    ) -> Callable[[], Any] | None:
        """Mint a permit for a boundary's egress operation and prepare its relay to the
        origin worker.

        The permit is audience-bound to the agent's own worker and relayed there as an
        ordinary control message on the authenticated attachment, never the raw request
        (the origin worker holds it privately) and never a dispatched task. A restart
        re-mints and re-relays it from the still-pending boundary. If the origin worker
        is gone the request cannot be recovered, so the boundary fails clean.
        """
        agent = self._tasks.get(env.task_id)
        engine = self._engines.get(agent.workflow_id) if agent else None
        if agent is None or engine is None:
            return None
        worker_id = agent.assigned_worker
        worker = self._worker_registry.get_worker(worker_id) if worker_id else None
        if worker_id is None or worker is None:
            self._logger.warning(
                "worker-originated tool op for %s has no live origin worker; "
                "failing the boundary clean",
                env.task_id,
            )
            self._settle_episode_invocation(
                env.task_id, env.call_correlation, error="origin worker unavailable"
            )
            return None
        occurrence = (env.task_id, env.call_correlation)
        stale = self._mediated_ops.take_stale_ops(occurrence)
        if any(op.node_alias != worker.node_alias for _, op in stale):
            self._logger.warning(
                "worker-originated tool op for %s: %s now names a worker of another "
                "node; failing the boundary clean",
                env.task_id,
                worker_id,
            )
            self._settle_episode_invocation(
                env.task_id, env.call_correlation, error="origin worker unavailable"
            )
            return None
        op_credential = self._resolve_op_credential(agent, env.interface)
        if isinstance(op_credential, _MissingCredential):
            self._settle_episode_invocation(
                env.task_id, env.call_correlation, error=op_credential.reason
            )
            return None
        max_results, timeout_sec, result_char_cap = self._mediated_ops.op_permit_budget(
            env.interface
        )
        deadline = time.time() + timeout_sec + _OP_PERMIT_SLACK_SEC
        permit = engine.mint_operation_permit(
            env.task_id,
            env.call_correlation,
            target_id=worker_id,
            target_generation=worker.incarnation,
            max_results=max_results,
            timeout_sec=timeout_sec,
            result_char_cap=result_char_cap,
            deadline_epoch=deadline,
            credential=op_credential.credential,
            deployment_credential=op_credential.deployment_credential,
        )
        if permit is None:
            self._settle_episode_invocation(
                env.task_id, env.call_correlation, error="could not mint a permit"
            )
            return None
        # A re-drive re-mints under a fresh permit id, keeping one pending op per
        # occurrence and its re-drive count.
        redrives = max((op.redrives for _, op in stale), default=0)
        self._mediated_ops.record_issued_op(
            permit.permit_id,
            PendingOp(
                env.task_id,
                env.call_correlation,
                worker_id,
                worker.node_alias,
                redrive_at=deadline + _OP_REDRIVE_BACKOFF_SEC * (2**redrives - 1),
                redrives=redrives,
            ),
        )
        message = MediatedOpMessage(
            worker_id=worker_id,
            frame_kind="permit",
            payload=self._mediated_ops.stamped_permit_payload(permit, agent),
        )
        return lambda: self._worker_registry.publish_mediated_op(worker, message)

    def authorize_model_turn(
        self, proposal: AgentModelTurnProposal, proposer_id: str
    ) -> None:
        """Authorize a held agent's in-turn model egress and relay a one-use permit.

        The agent's own worker holds the model request privately and proposes only its
        digest; the control plane validates the activation's model-invoke authority and
        its external binding, mints an audience-bound permit, and relays it over the
        worker's authenticated attachment. A denial — no live worker, a non-external
        binding, or an authority failure — relays a deny frame so the held turn fails
        fast rather than waiting out its deadline. No settle is expected: the held turn
        consumes the outcome in-worker and its durable progress rests on its
        turn-completion boundaries. Only the dispatch holding the agent's task, on the
        worker whose stream relayed the proposal, is authorized; any other proposer is
        denied.
        """
        self._answer_model_turn(proposal, proposer_id, None)

    def _answer_model_turn(
        self,
        proposal: AgentModelTurnProposal,
        proposer_id: str,
        parked: AuthorizeTurn | None,
    ) -> None:
        if (
            frame := self._authorize_model_turn(proposal, proposer_id, parked)
        ) is not None:
            worker, frame_kind, payload = frame
            self._worker_registry.publish_mediated_op(
                worker,
                MediatedOpMessage(
                    worker_id=worker.id, frame_kind=frame_kind, payload=payload
                ),
            )
        if parked is not None:
            # Answered or dropped; parked again, it is a new action.
            with self._lock:
                key = (proposal.agent_task_id, proposal.call_correlation)
                if self._parked_turns.get(key) is parked:
                    del self._parked_turns[key]

    def _authorize_model_turn(
        self,
        proposal: AgentModelTurnProposal,
        proposer_id: str,
        parked: AuthorizeTurn | None,
    ) -> tuple[Worker, str, dict[str, Any]] | None:
        """The permit or deny frame answering a model-turn proposal, and the worker to
        relay it to; None when none is relayed.

        Nothing is authorized from state not yet durable: while the agent's workflow
        holds writes, the authorization is parked until they commit, and the proposal
        gets no frame before then. A parked authorization relays nothing once its
        permit deadline passed, a later proposal of the turn superseded it, its worker
        changed incarnation, or its task no longer runs.
        """
        key = (proposal.agent_task_id, proposal.call_correlation)
        if (worker := self._worker_registry.get_worker(proposer_id)) is None:
            # A gone origin worker cannot receive a relay: the held turn fails on its
            # own permit deadline.
            return None
        with self._transition():
            if parked is not None and (
                self._parked_turns.get(key) != parked
                or time.time() >= parked.deadline_epoch
                or worker.incarnation != parked.incarnation
            ):
                return None
            agent = self._tasks.get(proposal.agent_task_id)
            engine = self._engines.get(agent.workflow_id) if agent else None
            if agent is None or engine is None:
                return None
            if parked is not None and agent.status != TaskStatus.DISPATCHED:
                return None
            if not self._fence.dispatch_live_locked(
                agent, proposer_id, proposal.dispatch_id
            ):
                return (
                    worker,
                    "deny",
                    deny_model_turn_payload(proposal, "model turn not held"),
                )
            _, timeout_sec, result_char_cap = self._mediated_ops.op_permit_budget(
                MODEL_INTERFACE
            )
            if not self._committer.close_locked(agent.workflow_id):
                again = AuthorizeTurn(
                    proposal,
                    proposer_id,
                    worker.incarnation,
                    (
                        parked.deadline_epoch
                        if parked is not None
                        else time.time() + timeout_sec
                    ),
                )
                self._parked_turns[key] = again
                self._actions.file_locked(agent.workflow_id, again)
                return None
            if parked is None:
                # A proposal of the turn answered now supersedes one parked before.
                self._parked_turns.pop(key, None)
            binding = self.resolve_model_binding(proposal.agent_task_id)
            reason = "model turn egress denied"
            if binding is None or binding.mode is not ModelBindingMode.OPENAI:
                return worker, "deny", deny_model_turn_payload(proposal, reason)
            op_credential = self._model_credential(agent, binding)
            if isinstance(op_credential, _MissingCredential):
                return (
                    worker,
                    "deny",
                    deny_model_turn_payload(proposal, op_credential.reason),
                )
            permit = engine.authorize_model_turn(
                proposal.agent_task_id,
                proposal.call_correlation,
                proposal.request_digest,
                target_id=proposer_id,
                target_generation=worker.incarnation,
                timeout_sec=timeout_sec,
                result_char_cap=result_char_cap,
                deadline_epoch=time.time() + timeout_sec + _OP_PERMIT_SLACK_SEC,
                credential=op_credential.credential,
                deployment_credential=op_credential.deployment_credential,
            )
            if permit is None:
                return worker, "deny", deny_model_turn_payload(proposal, reason)
            return (
                worker,
                "permit",
                self._mediated_ops.stamped_permit_payload(permit, agent),
            )

    def settle_mediated_operation(self, outcome: MediatedOperationOutcome) -> None:
        """Settle an agent boundary from its origin worker's fenced outcome report.

        The report carries exactly one of a reference-backed outcome, a bounded inline
        outcome, or a worker-fault error; each settles the originating boundary. The
        terminal fact commits before the worker-private request custody is reaped, so a
        lost report leaves the boundary pending for a same-idempotency-key re-drive. A
        duplicate or late report is absorbing at the boundary.
        """
        with self._transition():
            self._settle_mediated_operation(outcome)

    def _settle_mediated_operation(self, outcome: MediatedOperationOutcome) -> None:
        with self._cv:
            pending = self._mediated_ops.take_settled(outcome)
            worker_id = (
                pending.worker_id
                if pending
                else self._assigned_worker_locked(outcome.agent_task_id)
            )
            agent_task_id = outcome.agent_task_id
            call = outcome.call_correlation
            record = self._tasks.get(agent_task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None or not engine.boundary_settleable(agent_task_id, call):
                # No settle follows a duplicate or late report, so the request its
                # worker holds is reaped here, once the terminal it lost to is durable.
                if record is not None:
                    self._committer.close_locked(record.workflow_id)
                if worker_id:
                    self._file_locked(
                        agent_task_id, Reap(worker_id, agent_task_id, call)
                    )
                return
            if outcome.error is not None:
                self._settle_episode_invocation(
                    agent_task_id, call, error=f"tool operation failed: {outcome.error}"
                )
            elif outcome.outcome_ref is not None:
                self._settle_episode_invocation(
                    agent_task_id, call, ref=outcome.outcome_ref
                )
            elif outcome.outcome is not None:
                self._settle_episode_invocation(
                    agent_task_id, call, value=outcome.outcome.model_dump_json()
                )
            else:
                self._settle_episode_invocation(
                    agent_task_id, call, error="tool operation returned no outcome"
                )

    def _assigned_worker_locked(self, agent_task_id: str) -> str | None:
        record = self._tasks.get(agent_task_id)
        return record.assigned_worker if record else None

    def set_model_settler(
        self, settler: Callable[[ToolInvocationEnvelope], None]
    ) -> None:
        """Install the off-lane handler for the mediated ``model`` interface."""
        self._model_settler = settler

    def set_tool_broker(self, broker: Callable[[ToolInvocationEnvelope], None]) -> None:
        """Install the off-lane handler for fabric-served tool interfaces."""
        self._tool_broker = broker

    def set_resident_handlers(
        self,
        *,
        originate: Callable[[ToolInvocationEnvelope], bool],
        on_ack: Callable[[ResidentBootstrapAck], None],
        on_outcome: Callable[[ResidentOpOutcome], None],
        on_route_observation: Callable[[ResidentRouteObservation], None],
    ) -> None:
        """Install the worker-originated resident origination and report handlers."""
        self._resident_originate = originate
        self._resident_ack = on_ack
        self._resident_outcome = on_outcome
        self._resident_route_observation = on_route_observation

    def set_resident_task_end_hook(self, hook: Callable[[str], None]) -> None:
        """Install the consumer told when a resident serve task stops serving.

        The hook receives the task id once a dispatched resident serve task loses its
        dispatch, including to another dispatch, and once one settles or starts
        cancelling. It runs under the runtime's lock, so it must hand the work
        off and never call back in.
        """
        self._resident_tasks.set_resident_task_end_hook(hook)

    def set_resident_task_update_hook(self, hook: Callable[[str], None]) -> None:
        """Install the consumer told when a dispatched resident serve task reports an
        update under its current dispatch, such as its engine endpoint.

        It runs under the runtime's lock, so it must hand the work off and never call
        back in.
        """
        self._resident_tasks.set_resident_task_update_hook(hook)

    def set_resident_yield_hook(self, hook: Callable[[str], None]) -> None:
        """Install the consumer asked to free a worker a resident serve task occupies.

        The hook receives the serve task's id. It may run on the dispatcher's thread, so
        it must hand the work off and never call back in.
        """
        self._resident_tasks.set_resident_yield_hook(hook)

    def set_resident_terminal_hook(self, hook: ResidentTerminalHook) -> None:
        """Install the consumer that releases a resident admission credit on DS
        terminal.

        The hook receives the settled boundary's ``invocation_id`` and whether the
        outcome was a failure, so the Admission controller advances the linked claim to
        terminal on any fenced outcome — the sole normal credit release.
        """
        self._mediated_ops.set_resident_terminal_hook(hook)

    def resident_invocation_completed(
        self, workflow_id: str, invocation_id: str
    ) -> bool | None:
        """Whether a workflow's terminal boundary invocation completed; None while its
        ledger holds no terminal for it."""
        with self._lock:
            engine = self._engines.get(workflow_id)
            return (
                engine.boundary_invocation_completed(invocation_id) if engine else None
            )

    def originate_facade_turn_group(self, task_id: str, group: FacadeTurnGroup) -> None:
        """Record a turn-scoped facade group a worker captured on a held model turn.

        The worker facade captures a model turn's native facade calls and cleans the
        turn; the whole ordered membership and its single continuation are persisted
        on the task record before the turn resumes, so the episode's next completion
        routes the group rather than settling the episode DONE, and a restart-replayed
        completion still routes it. At most one group is open per episode; a second one
        while one holds the gate is refused by the busy fence, not stored here.
        """
        with self._transition():
            self._pending_facade_groups[task_id] = group
            if (record := self._tasks.get(task_id)) is not None:
                record.pending_facade_group = group
                self._committer.persist_locked(task_id)

    def receive_worker_facade_group(self, task_id: str, group: FacadeTurnGroup) -> None:
        """Record a facade group the worker carried on its held model turn's completion.

        The worker facade keeps each search member's request private and carries the
        ordered membership with per-member digests on the completion. Recording it under
        the busy fence refuses a second group while one's await-outcome members are
        unresolved, so the open one is never overwritten; the episode's next completion
        routes the recorded group rather than settling DONE.
        """
        if self.has_pending_facade(task_id):
            self._logger.warning(
                "refusing a second facade group for %s while one is open", task_id
            )
            return
        self.originate_facade_turn_group(task_id, group)

    def success_settles_task(self, task_id: str, payload: dict[str, Any]) -> bool:
        """Whether a worker success ends its task or yields the lane back to run again.

        An input preparation, a non-completion episode step, and a turn completion that
        carries or resumes a captured facade group each return the task to the queue,
        so a caller counting task completions counts one per task, not one per dispatch.
        """
        if payload.get("input_materialization") is not None:
            return False
        if (step := payload.get("agent_episode")) is None:
            return True
        if not isinstance(step, dict):
            return False
        if step.get("kind") != HarnessResultKind.COMPLETION:
            return False
        if payload.get("agent_episode_facade_group") is not None:
            return False
        return not self.has_pending_facade(task_id)

    def has_pending_facade(self, task_id: str) -> bool:
        """Whether a facade group is already captured or still open for this episode.

        The busy fence reads this to refuse a distinct second group before the open
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
            return self._episode_dispatch.resolve_model_binding(task_id)

    def gateway_binding_for(
        self, task_id: str
    ) -> tuple[str, AgentModelGatewayBinding] | None:
        """The task's owning workflow and its pinned model binding, for the gateway.

        The workflow id scopes the credential resolution so a vaulted ref yields a
        secret only within the workflow that minted it.
        """
        with self._lock:
            return self._episode_dispatch.gateway_binding_for(task_id)

    def resolve_service_dependency(
        self, task_id: str
    ) -> ResidentAdmissionBinding | None:
        """What the task binds for resident admission, read from its own plan node.

        Resolves for both an agent whose model binding is resident and an inference or
        embedding leaf that consumes a resident family; a non-resident task resolves to
        None. The binding's workflow id scopes admission bookkeeping to the submitting
        workflow.
        """
        with self._lock:
            return self._episode_dispatch.resolve_service_dependency(task_id)

    def boundary_settleable(self, task_id: str, call_correlation: str) -> bool:
        """Whether a mediated boundary still awaits its outcome."""
        with self._lock:
            return self._episode_dispatch.boundary_settleable(task_id, call_correlation)

    def _apply_private_state_seal_locked(self, task_id: str, sealed: Any) -> None:
        """Record the generation a holder sealed, ignoring a fenced-out report."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if engine is None:
            return
        report = PrivateStateSealReport.model_validate(sealed)
        try:
            engine.seal_private_state(task_id, report.manifest, report.write_epoch)
        except PrivateStateUnavailable as exc:
            # A superseded holder cannot advance the lineage; its step is already
            # fenced out of the write, so the binding keeps the generation it had.
            self._logger.warning(
                "refused a private-state seal for %s: %s", task_id, exc
            )

    def private_state_owner(self, task_id: str) -> OwnerFence | None:
        """The holder that must supply a task's bound private state, or None.

        None covers a task with no private state and an agent whose lineage has no
        sealed generation yet, both of which any eligible worker may run.
        """
        with self._lock:
            return self._episode_dispatch.private_state_owner(task_id)

    def agent_episode_dispatch(
        self, task_id: str, holder: OwnerFence
    ) -> AgentEpisodeDispatch | None:
        """The agent-episode context to ship with a dispatch, or None for a non-agent.

        The backend key comes from the operator's pinned harness binding, so a later
        deployment-default change cannot move a live activation. ``holder`` is the
        selected worker incarnation the dispatch grants private-state authority to; the
        grant supersedes any prior epoch, fencing a stale holder out of the write.
        """
        with self._lock:
            return self._episode_dispatch.agent_episode_dispatch(task_id, holder)

    def service_episode_dispatch(
        self, task_id: str
    ) -> ServiceLeafEpisodeDispatch | None:
        """The service-episode context for a resident leaf, or None.

        Only a resident-backed inference/embedding leaf takes this path; an agent whose
        model binding is resident runs its resident boundary through the agent episode.
        A resume ships the settled outcome to inject; a first dispatch ships none.

        A leaf that admits more than one embodiment names a service dependency for its
        resident candidate alone, so the resolved embodiment decides this path rather
        than the dependency's presence.
        """
        with self._lock:
            return self._episode_dispatch.service_episode_dispatch(task_id)

    def serves_from_replica(self, task_id: str) -> bool:
        """Whether a task's dispatch carries its invocation to a resident replica, a
        menu-resolved or pinned resident leaf, rather than loading a model locally."""
        with self._lock:
            return self._episode_dispatch.serves_from_replica(task_id)

    def embodiment_pinned(self, task_id: str) -> bool:
        """Whether a task's resolved embodiment is committed to the run carrying it."""
        with self._lock:
            return self._episode_dispatch.embodiment_pinned(task_id)

    def embodiment_menu(self, task_id: str) -> InferenceEmbodimentMenu | None:
        """The embodiments a ready task's plan node offers, if it offers a menu."""
        with self._lock:
            return self._episode_dispatch.embodiment_menu(task_id)

    def record_embodiment_selection(
        self, task_id: str, alternative_id: str, selector: str, evidence: str
    ) -> str | None:
        """Durably bind a task to one embodiment, returning the bound alternative.

        Recorded before the task's worker message is published, so the choice survives a
        loss between publication and the attempt bookkeeping that follows it. A pinned
        selection is kept and returned unchanged.
        """
        with self._transition():
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None or record is None:
                return None
            selection = engine.record_embodiment_selection(
                task_id, alternative_id, selector, evidence
            )
            if selection is None:
                return None
            self._committer.save_ledger_locked(record.workflow_id)
            return selection.alternative_id

    def episode_feasible(self, task_id: str) -> bool:
        """Whether a ready episode's declared alternative can be placed now.

        The generic live-feasibility handoff from the lowerer's episode annotation to
        the scheduler: an infeasible alternative is deferred by the dispatcher, holding
        no worker. Absent a configured check, or for a task the plan did not cut into an
        episode, placement is always feasible. A menu node is checked through the
        episode of the embodiment already resolved for it, never as one implicit episode
        over the whole menu.
        """
        if self._feasibility_check is None:
            return True
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None:
                spec = None
            elif (menu := engine.embodiment_menu(task_id)) is not None:
                selection = engine.embodiment_selection(task_id)
                candidate = (
                    menu.candidate(selection.alternative_id) if selection else None
                )
                spec = candidate.episode if candidate else None
            else:
                spec = engine.episode_spec(task_id)
        return True if spec is None else self._feasibility_check(spec)

    def credential_pointers(self, task_ids: Iterable[str]) -> dict[str, list[str]]:
        """Where each task's restored credentials sit in its dispatched spec."""
        with self._lock:
            return {
                task_id: sorted(refs)
                for task_id in task_ids
                if (record := self._tasks.get(task_id)) is not None
                and (refs := record.credential_refs)
            }

    def credentialed_task(
        self, record: TaskRecord
    ) -> tuple[TaskEnvelopeTemplate, Callable[[str], str]] | None:
        """A copy of a task with its vaulted credentials restored, and a scrubber for
        any text produced from it; None when a credential is no longer retained.

        The copy is what a dispatch renders and sends. Nothing writes it back, so the
        record keeps only the refs.
        """
        refs = record.credential_refs
        if not refs:
            return record.task, _unscrubbed
        values = self._credential_vault.resolve_values(
            record.workflow_id, refs.values()
        )
        if any(ref not in values for ref in refs.values()):
            return None
        task = record.task.model_copy(deep=True)
        set_spec_values(
            task.spec, {pointer: values[ref] for pointer, ref in refs.items()}
        )
        return task, credential_scrubber(values.values())

    def declared_contract(
        self, task_id: str, spec: TaskSpecBase | None = None
    ) -> CanonicalInferenceContract | None:
        """The contract a leaf carries to the worker, for it to resolve and report.

        A leaf that admits more than one embodiment names its contract here rather than
        in the executor, so every embodiment resolves one request and stores one result
        shape. A leaf pinned to resident serving names one when it declares a batch,
        because the conversations a replica serves under its one claim are the
        contract's, and whenever its prompts come from upstream, because only the worker
        holding that value can resolve them. It names no embodiment: the worker reads a
        contract and never learns which one it is running.

        A leaf that admits one embodiment and serves it locally declares none, whether
        its prompts are literal or come from upstream: it resolves them in its own
        executor and keeps reporting the native result that embodiment has always
        reported. So does a pinned single-prompt literal leaf.

        ``spec``, when given, is the dispatched spec with its credentials restored.
        """
        with self._lock:
            return self._episode_dispatch.declared_contract(task_id, spec)

    def record_input_resolution(
        self,
        task_id: str,
        worker_id: str | None,
        binding_payload: Any,
        dispatch_id: str | None = None,
    ) -> None:
        """Record how a task's inputs resolved on its origin worker, and save it.

        The worker reports this before either embodiment reaches a model, so the
        resolution is durable ahead of a local generation or a resident service issue,
        and the admission that follows is sized from the cardinality that materialized.
        A resolution already recorded stands, and a different one leaves it in place.
        """
        try:
            binding = InputResolutionBinding.model_validate(binding_payload)
        except ValidationError:
            self._logger.warning(
                "[fabric] a task reported an unreadable input resolution: %s", task_id
            )
            return
        with self._transition():
            record = self._tasks.get(task_id)
            if record is None or not self._fence.accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                return
            if (engine := self._engines.get(record.workflow_id)) is None:
                return
            recorded = engine.input_resolution(task_id) is not None
            engine.record_input_resolution(task_id, binding)
            if not recorded and engine.input_resolution(task_id) is not None:
                self._committer.save_ledger_locked(record.workflow_id)
            else:
                # A replay acknowledges only once what it replays is durable.
                self._committer.close_locked(record.workflow_id)

    def input_resolution_binding(self, task_id: str) -> InputResolutionBinding | None:
        """The binding a task's recorded resolution carries, if one was recorded."""
        with self._lock:
            resolution = self._content_bindings.input_resolution_locked(task_id)
        return resolution.binding if resolution is not None else None

    def content_scope(self, task_id: str) -> str:
        """The authorization scope a task's content is written under.

        Control assigns it from the task's own owner, so every write this task makes —
        its dispatch's and the resident completion its invocation materializes — lands
        in one scope rather than each path deriving its own.
        """
        with self._lock:
            record = self._tasks.get(task_id)
        return record.org_id if record is not None else ""

    def renewable_content_scope(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> str | None:
        """The scope a task's store access renews in, or None when it may not renew.

        Renewal serves the dispatch holding a task, on the worker asking for it; a task
        that has settled or moved to another dispatch is given nothing, so a superseded
        attempt's late write fails rather than landing under fresh access.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or not self._fence.dispatch_live_locked(
                record, worker_id, dispatch_id
            ):
                return None
            return record.org_id

    def content_binding_authorizes(
        self, task_id: str, worker_id: str, reference: ContentReference
    ) -> bool:
        """Whether a task's own binding entitles its worker to read this object.

        Read-only evidence for the content authority: the worker must be the one
        running the task, and the task must already be bound to exactly this reference
        — the request it was prepared with, an outcome the engine delivered into its
        episode, the settled result of an upstream task it depends on, a value it reads
        through an incoming edge, or the producer result one of its accepted inputs
        or its fan-out element is frozen to. Naming an object it merely knows of
        authorizes nothing.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            return (
                record is not None
                and record.status not in TERMINAL_TASK_STATUSES
                and self._fence.holds_dispatch_locked(record, worker_id, None)
                and self._content_bindings.consumes_locked(record, reference)
            )

    def upstream_task_ids(self, task_id: str) -> set[str]:
        """Every task of its workflow a task depends on, directly or transitively."""
        with self._lock:
            return self._content_bindings.upstream_task_ids_locked(task_id)

    def input_element(self, task_id: str) -> ResultElementRef | None:
        """The producer element a leaf fan-out child runs on, for its worker to hydrate.

        An agent child receives its element through its accepted input.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None or engine.agent_operator(task_id) is not None:
                return None
            element = self._content_bindings.input_element_locked(task_id)
        return element.ref if element is not None else None

    def recorded_input_reference(self, task_id: str) -> ContentReference | None:
        """Where a task's prepared request is, for the run that hydrates it."""
        with self._lock:
            resolution = self._content_bindings.input_resolution_locked(task_id)
        return resolution.reference if resolution is not None else None

    def prepares_inputs(self, task_id: str) -> bool:
        """Whether this dispatch resolves a task's inputs rather than running it.

        A leaf whose source declares no envelope is screened against the request it
        actually produces, so that request is materialized on a worker before any
        embodiment can be chosen for it. Once the materialization is committed the leaf
        dispatches like any other.
        """
        with self._lock:
            contract = self.declared_contract(task_id)
            if contract is None or not contract.source.prepared_before_selection:
                return False
            resolution = self._content_bindings.input_resolution_locked(task_id)
            return resolution is None or resolution.reference is None

    def _apply_input_materialization_locked(self, task_id: str, payload: Any) -> None:
        """Commit one preparation's binding and request reference, and re-ready.

        Both facts land together, so the request a later run hydrates becomes durable
        exactly when the binding proving what it is does. The task then returns to the
        ready queue for the dispatch that chooses an embodiment and runs it.
        """
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if record is None or engine is None:
            return
        if record.status in TERMINAL_TASK_STATUSES:
            return
        if record.status == TaskStatus.CANCELLING:
            self._settle_cancelled_locked(record, time.time())
            return
        if (standing := engine.input_resolution(task_id)) is not None and (
            standing.reference is not None
        ):
            # The stream is at-least-once: a replayed success finds its own commit
            # standing. Re-readying would pull the run it already released back to the
            # queue and dispatch a second embodiment against one work item.
            return
        try:
            materialization = ResolvedInputMaterialization.model_validate(payload)
        except ValidationError:
            self._settle_preparation_failure_locked(
                record, engine, "a preparation reported an unreadable materialization"
            )
            return
        size = materialization.reference.size_bytes
        if (limit := self._max_prepared_input_bytes) is not None and size > limit:
            self._settle_preparation_failure_locked(
                record,
                engine,
                f"the prepared request is {size} bytes and this deployment admits at "
                f"most {limit} (ORCHESTRATOR_MAX_PREPARED_INPUT_BYTES)",
            )
            return
        engine.record_input_resolution(
            task_id, materialization.binding, materialization.reference
        )
        self._release_dispatch_locked(record, [task_id], front=True)
        self._committer.save_ledger_locked(record.workflow_id)

    def _settle_preparation_failure_locked(
        self, record: TaskRecord, engine: OrchestrationEngine, reason: str
    ) -> None:
        if self._apply_advance_locked(
            record.workflow_id,
            engine.on_failed(record.task_id, reason, retryable=False),
        ):
            self._cv.notify_all()
        self._committer.save_ledger_locked(record.workflow_id)
        self._committer.settle_if_done_locked(record.workflow_id)

    def published_outputs(
        self, workflow_id: str, name: str | None = None
    ) -> PublishedOutputs | None:
        """A workflow's published outputs, or one output's, in no particular order.

        None for a workflow with no ledger.
        """
        with self._lock:
            engine = self._engines.get(workflow_id)
            if engine is None:
                return None
            return PublishedOutputs(
                members=published_members(engine, name),
                open=not self._committer.workflow_settlement_locked(
                    workflow_id
                ).settled,
            )

    def published_output(
        self,
        workflow_id: str,
        name: str,
        scope_id: str | None,
        key: str | None,
        sequence: int | None,
    ) -> PublishedOutput | None:
        """One published output's member at the selectors, or None with no ledger."""
        with self._lock:
            engine = self._engines.get(workflow_id)
            if engine is None:
                return None
            declaration, member = published_member(
                engine, name, scope_id, key, sequence
            )
            return PublishedOutput(
                declaration=declaration,
                member=member,
                open=not self._committer.workflow_settlement_locked(
                    workflow_id
                ).settled,
            )

    def read_output(self, member: OutputMember) -> BaseExecutorResult:
        """The value a published member settled with, as an input reading it would
        read it, read off the lock.

        Raises ``ResultUnavailable`` while the store cannot be reached and
        ``ResultUnreadable`` for a value that is missing, corrupt, or not readable.
        """
        value_ref = member.publication.value_ref if member.publication else None
        if value_ref is None:
            raise ResultUnreadable(f"output {member.name} has no bound result")
        try:
            with self._lock:
                binding = self._content_bindings.value_binding_locked(value_ref)
            return upstream_value(binding, self._results.read)
        except (UnreadableInput, IndexError) as exc:
            raise ResultUnreadable(f"output {member.name}: {exc}") from exc

    def resolve_v2_legacy_result(
        self, workflow_id: str, task_id: str
    ) -> ResultPublication | None:
        """Resolve a legacy task's induced output slot (compatibility adapter)."""
        with self._lock:
            engine = self._engines.get(workflow_id)
            return engine.resolve_legacy_task(task_id) if engine else None

    def result_binding(self, task_id: str) -> ResultBinding | None:
        """What a task's result reads as, or None when it has no readable result.

        A v2 task resolves through the ledger — its induced output slot, or for a task
        materialized at run time the value its work item settled with. A v1 task
        resolves through the reference its success bound on its record.
        """
        with self._lock:
            return self._content_bindings.result_binding_locked(task_id)

    def scoped_inputs(self, task_id: str) -> dict[str, ScopedInput] | None:
        """The values a task reads through its incoming edges, by the names its spec
        reads them through; None for a root task fed only by tasks.

        Raises ``UnreadableInput`` for a value no binding can carry.
        """
        with self._lock:
            return self._content_bindings.scoped_inputs_locked(task_id)

    def read_binding(self, binding: ResultBinding) -> ResultEnvelope:
        """The result envelope a binding reads; raises when unreadable."""
        return self._results.read(binding)

    def read_result(self, task_id: str) -> ResultEnvelope | None:
        """A task's result envelope, None when it has none; raises when unreadable."""
        binding = self.result_binding(task_id)
        return self._results.read(binding) if binding is not None else None

    def read_result_bytes(self, task_id: str) -> bytes | None:
        """A task's stored result envelope bytes, None when it has none."""
        binding = self.result_binding(task_id)
        return self._results.read_bytes(binding) if binding is not None else None

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
        with self._transition():
            return self._resolve_uncertain_locked(task_id)

    def _resolve_uncertain_locked(
        self, task_id: str, error: str | None = None
    ) -> Advance:
        """Resolve an in-flight work item's uncertainty; a failure terminalizes the
        boundary invocations it held, whose credits release once the ledger is saved.
        ``error`` is the executor's message for a reported failure."""
        record = self._tasks.get(task_id)
        if record is None or (engine := self._engines.get(record.workflow_id)) is None:
            return Advance()
        advance = engine.on_uncertain(task_id, error)
        if advance.retry:
            self._release_dispatch_locked(record, [task_id], front=True)
        elif advance.failed:
            # A lost child's failure can release its scope's join, readying what
            # follows.
            if self._apply_advance_locked(record.workflow_id, advance):
                self._cv.notify_all()
            self._actions.file_locked(
                record.workflow_id,
                *_failed_credits(engine.terminalize_unsettled_invocations([task_id])),
                *self._mediated_ops.reap_ops_for_agents_locked(advance.failed),
            )
        self._committer.save_ledger_locked(record.workflow_id)
        self._committer.settle_if_done_locked(record.workflow_id)
        return advance

    def _apply_advance_locked(self, workflow_id: str, advance: Advance) -> bool:
        """Apply an engine advance: fail and persist what it failed, settle what a dead
        route skipped, cancel what a residual policy cancelled, then record the inputs
        its agents accept, give the work it materialized its tasks, and ready its work.
        Returns whether it changed any task.

        The failed, skipped and cancelled records persist here; the records of the
        work it materialized land with the caller's ledger write, which every caller
        makes once the transition ends, so the ledger never leads them. A value still
        to read for a branch or spawn is read off the lock by the workflow's re-drive.
        """
        # A ready/settle advance never carries a retry; the failure path drives those.
        assert not advance.retry, "retry is applied by the failure path"
        engine = self._engines.get(workflow_id)
        changed = bool(advance.failed)
        self._fail_v2_advance_locked(engine, advance)
        changed |= self._skip_dead_routes_locked(advance.skipped)
        if engine is not None and advance.cancelled:
            changed |= self._cancel_residual_locked(
                workflow_id, engine, advance.cancelled
            )
        if engine is not None:
            staged = Advance()
            self._agent_inputs.stage_agent_inputs_locked(workflow_id, engine, staged)
            changed |= bool(staged.failed)
            self._fail_v2_advance_locked(engine, staged)
            advance.extend(staged)
            if missing := self._occurrences.materialize_locked(
                workflow_id, engine, advance
            ):
                self._fail_workflow_locked(workflow_id, blueprint_missing(missing[0]))
                return True
            if engine.awaits_control_reads():
                self._redrive.drive_now(workflow_id)
        for task_id in advance.ready:
            if self._ready.enqueue_ready_locked(task_id):
                changed = True
        return changed

    def _skip_dead_routes_locked(self, task_ids: list[str]) -> bool:
        """Settle each task no route reaches as done without running, with an empty
        result that says so; returns whether any settled.

        It takes no attempt and never reports a success: the engine already settled
        its work item dead, and only its record follows.
        """
        skipped: list[str] = []
        for task_id in task_ids:
            record = self._tasks.get(task_id)
            if record is None or record.status in TERMINAL_TASK_STATUSES:
                continue
            record.status = TaskStatus.DONE
            record.assigned_worker = None
            record.finished_ts = time.time()
            record.result_skip = dict(ROUTE_NOT_TAKEN)
            self._completed.add(task_id)
            self._dag.forget_pending(task_id)
            self._ready.remove_from_ready_locked(task_id)
            skipped.append(task_id)
        if skipped:
            self._committer.commit_locked(*skipped)
        return bool(skipped)

    def _fail_v2_advance_locked(
        self,
        engine: OrchestrationEngine | None,
        advance: Advance,
        *,
        persist: bool = True,
    ) -> list[tuple[str, str]]:
        """Fail the tasks an advance settled failed, each for the reason the engine
        names; returns each one changed with its reason. Persists them here when
        ``persist`` is set.
        """
        changed: list[tuple[str, str]] = []
        for task_id in advance.failed:
            text = (
                engine.failure_reason(task_id) if engine is not None else None
            ) or "declared-failure obligation"
            if self._fail_v2_records_locked([task_id], text, persist=False):
                changed.append((task_id, text))
        if persist and changed:
            self._committer.commit_locked(*(task_id for task_id, _ in changed))
        return changed

    def _fan_out_children_locked(
        self,
        workflow_id: str,
        engine: OrchestrationEngine,
        producer_task_id: str,
        read: FanoutRead | None,
    ) -> Advance:
        """Fan a settled producer's collection out to the spawn it feeds.

        Each element of the producer's result becomes one child, then the spawn seals.
        A producer that feeds no spawn yields no children.
        """
        advance = Advance()
        spawn_op = engine.fanout_spawn(producer_task_id)
        if spawn_op is None:
            return advance
        if not engine.spawn_is_open(spawn_op):
            return advance  # already sealed: a re-driven fan-out is a no-op
        if read is None:
            # Nothing read the collection ahead of the lock: read it off the lock.
            self._redrive.drive_now(workflow_id)
            return advance
        if read.unavailable:
            # The collection is in a store that could not be reached: the spawn stays
            # open, holding the workflow, until a re-drive reads it.
            self._logger.warning(
                "Deferring fan-out for %s until the content store answers: %s",
                producer_task_id,
                read.error,
            )
            self._redrive.schedule(workflow_id)
            return advance
        if read.error is not None:
            # A settled producer's result was stored before its success was reported,
            # so one that cannot be read is a broken store or binding, never an empty
            # collection: sealing zero children here would close the join on a lie.
            self._logger.error("Failing workflow %s: %s", workflow_id, read.error)
            self._fail_workflow_locked(workflow_id, read.error)
            return advance
        binding = self._content_bindings.result_binding_locked(producer_task_id)
        content = binding.reference if binding is not None else None
        if content != read.reference:
            # The read predates the binding the producer settled with: read again.
            self._redrive.drive_now(workflow_id)
            return advance
        value = ValueRef(
            kind="legacy_task_result", legacy_task_id=producer_task_id, content=content
        )
        return self._spawn_children_locked(
            workflow_id, engine, spawn_op, value, read.count
        )

    def _spawn_children_locked(
        self,
        workflow_id: str,
        engine: OrchestrationEngine,
        spawn: str,
        value: ValueRef,
        count: int,
    ) -> Advance:
        """Create one child of a spawn occurrence per element of ``value``'s
        collection, then seal it.

        A child receives its element as a frozen reference into the value: an agent
        child through the typed accepted-input channel of its declared entry port, a
        leaf child as its child-init input, which its worker hydrates, and a region
        definition's child as the definition's param.
        """
        advance = Advance()
        region = engine.spawn_region(spawn)
        handle = engine.spawn_handle(spawn)
        if region is None:
            return advance
        template = region.child_template_ref
        if region.child_definition_ref is None and (
            template is None
            or self._occurrences.blueprint_locked(workflow_id, template) is None
        ):
            # Compile-time validation rejects an unresolved or non-leaf child template,
            # so reaching here is an internal inconsistency; fail the workflow rather
            # than defer a join that could never close.
            self._logger.error(
                "Spawn %s child template %r is unresolvable; failing the workflow",
                spawn,
                template,
            )
            self._fail_workflow_locked(
                workflow_id,
                f"spawn child template {template!r} is not a dispatchable leaf",
            )
            return advance
        child_is_agent = (
            template is not None and engine.agent_entry_port(template) is not None
        )
        for index in range(count):
            element = _element(value, index)
            try:
                if region.child_definition_ref is not None:
                    advance.extend(engine.enter_definition_child(spawn, index, element))
                    continue
                assert template is not None
                if child_is_agent:
                    child_task_id = engine.create_fanout_child(handle, element)
                    self._occurrences.register_locked(
                        workflow_id, child_task_id, template
                    )
                    agent_inputs.mint_fanout_facet_locked(
                        engine,
                        child_task_id,
                        element.legacy_task_id or "",
                        index,
                        element,
                    )
                    advance.extend(engine.reconsider_admission(child_task_id))
                    continue
                advance.extend(engine.materialize_child(handle, value_ref=element))
            except RegionError:
                break  # a budget, seal, or denial stops further children
        advance.extend(engine.seal_spawn(handle))
        return advance

    def _drive_workflow(self, workflow_id: str) -> None:
        try:
            self._check_unavailable_inputs(workflow_id)
        except Exception:
            # A check that stopped partway leaves its tasks held, so it runs again.
            self._logger.exception(
                "Checking the held inputs of workflow %s failed", workflow_id
            )
            self._redrive.schedule(workflow_id)
        self._redrive_workflow(workflow_id)

    def _check_unavailable_inputs(self, workflow_id: str) -> None:
        """Read each held task's unreadable inputs from here, off the lock.

        Content missing or corrupt here fails the task as a report of the dispatch that
        could not read it, published on the task-event stream like the worker's own
        report. The check stays until that failure has committed, so a verdict whose
        handling failed a durable write is reported again. A store control cannot reach
        either, or any other error reading it, holds the task until a later re-drive.
        Content control reads fine returns the task to the queue without blaming its
        worker: one later read with control's own access says nothing about the path the
        worker read through.
        """
        with self._lock:
            checks = {
                task_id: check
                for task_id, check in self._inputs.input_checks.items()
                if (record := self._tasks.get(task_id)) is not None
                and record.workflow_id == workflow_id
            }
        if not checks:
            return
        verdicts = {
            task_id: self._inputs.verify_inputs(check.references)
            for task_id, check in checks.items()
            if check.unreadable is None
        }
        with self._transition():
            failures = self._inputs.settle_input_checks_locked(
                workflow_id, checks, verdicts
            )
        for event in failures:
            self._report_failure(event)

    def set_failure_reporter(self, report: Callable[[TaskEvent], None]) -> None:
        """Install the handler a failure control decides on is reported through, so it
        runs the side effects of a worker's failure report."""
        self._failure_reporter = report

    def _report_failure(self, event: TaskEvent) -> None:
        if self._failure_reporter is not None:
            self._failure_reporter(event)
            return
        self.fail_dispatch(
            event.task_id,
            event.worker_id or "",
            {},
            event.ts,
            event.dispatch_id,
            error=event.error,
            retryable=event.retryable,
            failure_kind=event.failure_kind,
        )

    def _redrive_workflow(self, workflow_id: str) -> None:
        """Drive the advances of a workflow that wait on a read of stored results.

        Every settled producer whose spawn has yet to fan out has its collection read,
        every branch awaiting its selector and every other spawn awaiting its fan-out
        has the value reaching it read, and every blocked agent whose bound inputs
        need reading has them read, all off the lock. The results apply under it, each
        only while what it was read for still waits on it: an agent's inputs record
        only while the snapshot they were read for holds, and are read again
        otherwise. A read that cannot reach the store schedules the next re-drive.
        """
        with self._lock:
            engine = self._engines.get(workflow_id)
            if engine is None:
                return
            producers = [
                (task_id, self._content_bindings.result_binding_locked(task_id))
                for task_id, spawn_op in engine.fanout_producers().items()
                if (record := self._tasks.get(task_id)) is not None
                and record.status == TaskStatus.DONE
                and engine.spawn_awaits_children(spawn_op)
            ]
            produced = {engine.fanout_producers()[task_id] for task_id, _ in producers}
            controls = [
                (key, value, self._task_result_binding_locked(value))
                for key, value in (
                    *engine.pending_branch_reads(),
                    *(
                        (key, value)
                        for key, value in engine.awaiting_fanouts()
                        if key not in produced
                    ),
                )
            ]
            snapshots = [
                snapshot
                for task_id in engine.blocked_input_agents()
                if (
                    snapshot := self._agent_inputs.agent_input_snapshot_locked(
                        engine, task_id
                    )
                )
                is not None
                and snapshot.references
            ]
        reads = {
            task_id: fanout.read_fanout(self._results, task_id, binding)
            for task_id, binding in producers
        }
        control_values = {
            key: control_reads.read_control_value(self._results, value, binding)
            for key, value, binding in controls
        }
        values = agent_inputs.read_input_values(self._results, snapshots)
        with self._transition():
            if (engine := self._engines.get(workflow_id)) is None:
                return
            advance = Advance()
            for task_id, _binding in producers:
                advance.extend(
                    self._fan_out_children_locked(
                        workflow_id, engine, task_id, reads[task_id]
                    )
                )
            for key, value, _ in controls:
                advance.extend(
                    self._apply_control_read_locked(
                        workflow_id, engine, key, value, control_values[key]
                    )
                )
            for snapshot in snapshots:
                if self._agent_inputs.agent_input_snapshot_locked(
                    engine, snapshot.task_id
                ) != (snapshot):
                    self._redrive.drive_now(workflow_id)
                    continue
                self._agent_inputs.settle_agent_inputs_locked(
                    workflow_id,
                    engine,
                    snapshot,
                    values,
                    advance,
                )
            if self._apply_advance_locked(workflow_id, advance):
                self._cv.notify_all()
            self._committer.save_ledger_locked(workflow_id)
            self._committer.settle_if_done_locked(workflow_id)

    def _task_result_binding_locked(self, value: ValueRef) -> ResultBinding | None:
        """The stored result a value reference names, when it names a task's."""
        if value.legacy_task_id is None:
            return None
        if value.content is not None:
            return ResultBinding(task_id=value.legacy_task_id, reference=value.content)
        return self._content_bindings.result_binding_locked(value.legacy_task_id)

    def _apply_control_read_locked(
        self,
        workflow_id: str,
        engine: OrchestrationEngine,
        key: str,
        value: ValueRef,
        read: control_reads.ControlRead,
    ) -> Advance:
        """Route a branch or fan out a spawn by the value read for it, while it still
        waits on that value.

        A value the store could not answer for is read again later; one that cannot be
        read at all fails the occurrence.
        """
        if (rule := engine.selection_rule(key)) is not None:
            if dict(engine.pending_branch_reads()).get(key) != value:
                return Advance()
            if read.unavailable:
                self._redrive.schedule(workflow_id)
                return Advance()
            if read.error is not None:
                return engine.accept_branch_selection(key, None, error=read.error)
            return engine.accept_branch_selection(
                key, control_reads.dig(read.value, rule.field)
            )
        if dict(engine.awaiting_fanouts()).get(key) != value:
            return Advance()
        if read.unavailable:
            self._redrive.schedule(workflow_id)
            return Advance()
        if read.error is not None or read.elements is None:
            reason = read.error or "the spawn's input is not a collection"
            return engine.fail_control(key, f"fan-out input unreadable: {reason}")
        content = value.content
        if content is None and (binding := self._task_result_binding_locked(value)):
            content = binding.reference
        return self._spawn_children_locked(
            workflow_id,
            engine,
            key,
            value.model_copy(update={"content": content}),
            read.elements,
        )

    def _prefetch_fanout(
        self,
        task_id: str,
        reference: ContentReference | None,
        skip: dict[str, Any] | None,
    ) -> FanoutRead | None:
        """Read a spawn producer's fan-out collection before its success takes the lock.

        The read goes to the shared store, so it runs, and retries, outside the runtime
        lock; only a success feeding a spawn that has yet to fan out pays for it. A
        skipped producer has no collection, so its spawn fans out to no children.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            spawn_op = engine.fanout_spawn(task_id) if engine else None
            if (
                record is None
                or record.status in TERMINAL_TASK_STATUSES
                or record.status == TaskStatus.CANCELLING
                or engine is None
                or spawn_op is None
                or not engine.spawn_awaits_children(spawn_op)
            ):
                return None
            reference = self._content_bindings.accepted_reference(record, reference)
        return fanout.read_fanout(
            self._results,
            task_id,
            ResultBinding(task_id=task_id, reference=reference, skip=skip),
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
            if not record or record.status in SETTLING_TASK_STATUSES:
                # A settling task is already on its way to a terminal; failing it would
                # overwrite the cancellation a settle path is still waiting to apply.
                continue
            self._record_failures.fail_record_locked(record, reason)
            failed_now.append(task_id)
        if persist and failed_now:
            self._committer.commit_locked(*failed_now)
        return failed_now

    def _fail_workflow_locked(self, workflow_id: str, reason: str) -> None:
        """Fail a workflow in its ledger and every non-terminal task of it, persist the
        terminal facts, and file what its work held for delivery once they are
        durable."""
        owed = self._terminate_workflow_locked(workflow_id, reason, reason)
        non_terminal = [
            record.task_id
            for record in self._tasks.of_workflow(workflow_id)
            if record.status not in TERMINAL_TASK_STATUSES
        ]
        self._fail_v2_records_locked(non_terminal, reason, persist=True)
        self._actions.file_locked(workflow_id, *owed)
        self._committer.save_ledger_locked(workflow_id)
        self._committer.settle_if_done_locked(workflow_id)
        self._cv.notify_all()

    def plan_merge(
        self, task_id: str, max_batch_size: int, assigned_worker: str
    ) -> list[str]:
        if max_batch_size <= 1:
            return []
        with self._transition():
            return self._merges.plan_merge_locked(
                task_id, max_batch_size, assigned_worker
            )

    def release_merge(self, task_id: str) -> None:
        with self._transition():
            self._merges.release_merge_locked(task_id)

    def merged_child_record(self, task_id: str, child_id: str) -> TaskRecord | None:
        """A child's record while it is still merged into ``task_id``'s dispatch."""
        with self._cv:
            return self._merges.merged_child_record_locked(task_id, child_id)

    def release_merged_child(
        self, task_id: str, child_id: str, merge_key: str | None
    ) -> None:
        """Take one child out of a task's merge and return it to the ready queue, to
        merge next under ``merge_key``, or to run alone when it is None."""
        with self._transition():
            self._merges.release_merged_child_locked(task_id, child_id, merge_key)

    # ------------------------------------------------------------------ #
    # State updates (dispatch & events)
    # ------------------------------------------------------------------ #

    def holds_dispatch(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> bool:
        """Whether an event from ``worker_id`` belongs to the task's dispatch."""
        with self._lock:
            record = self._tasks.get(task_id)
            return record is not None and self._fence.holds_dispatch_locked(
                record, worker_id, dispatch_id
            )

    def begin_publish(
        self,
        task_id: str,
        worker: Worker,
        dispatch_id: str | None,
        *,
        input_preparation: bool = False,
    ) -> PublishGate:
        """Mark a dispatch as being published, so its worker's earliest events apply.

        Everything the dispatch carries is made durable first: the held writes of its
        task's workflow and of each workflow a merged child belongs to. A dispatch whose
        writes stay held, or whose task's last report waits to be handled again, is not
        marked and returns ``NOT_DURABLE``. One whose task's report is being handled
        returns ``REPORTING``, and the task is queued again once the handling ends.
        """
        publish = Publish(
            worker.id, dispatch_id, supplier_id(worker), input_preparation
        )
        with self._transition():
            spanned = {
                record.workflow_id
                for spanned_id in (
                    task_id,
                    *self._merges.merge_children_map.get(task_id, ()),
                )
                if (record := self._tasks.get(spanned_id)) is not None
            }
            # Every spanned workflow is closed, whether or not an earlier one is.
            try:
                closed = [self._committer.close_locked(w) for w in sorted(spanned)]
            except Exception:
                # A fault of the write's own recurs on every write of its workflow.
                # The dispatch drops what it merged from a faulted workflow, whose
                # owed rewrite carries those tasks' return, and a task of the faulted
                # workflow waits aside until a write of it is made.
                self._merges.unmerge_locked(task_id)
                if (record := self._tasks.get(task_id)) is None:
                    return PublishGate.NOT_PENDING
                if record.workflow_id not in self._committer.faulted:
                    return PublishGate.NOT_DURABLE
                self._write_faulted.setdefault(record.workflow_id, set()).add(task_id)
                return PublishGate.WRITE_FAULTED
            if not all(closed):
                return PublishGate.NOT_DURABLE
            if task_id in self._committer.unacknowledged:
                return PublishGate.NOT_DURABLE
            if self._committer.reporting(task_id):
                self._awaiting_reports.add(task_id)
                return PublishGate.REPORTING
            if not self._fence.begin_publish_locked(task_id, publish):
                return PublishGate.NOT_PENDING
            return PublishGate.PUBLISH

    def abandon_publish(self, task_id: str) -> bool:
        """Drop the mark of a dispatch whose publish failed.

        Returns whether the task still needs returning: a dispatch its worker reported
        on stands, and one a cancel recorded settles CANCELLED.
        """
        with self._transition():
            publish = self._fence.take_publish(task_id)
            if publish is None or not publish.recorded:
                return True
            record = self._tasks.get(task_id)
            if (
                not publish.reported
                and record is not None
                and record.status == TaskStatus.CANCELLING
                and record.dispatch_id == publish.dispatch_id
            ):
                self._settle_cancelled_locked(record, time.time())
            return False

    def mark_dispatched(self, task_id: str) -> bool:
        """Record the publish `begin_publish` marked; returns whether it holds the task.

        A dispatch an event of its worker recorded first is not recorded again, and one
        that ended before its record, or whose task already settles, records nothing.
        """
        with self._transition():
            return self._fence.mark_dispatched_locked(task_id)

    def mark_started(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None = None,
    ) -> EventEffect:
        """Record that a task's worker started it."""
        started_ts = parse_iso_ts(str(payload.get("started_at") or ts))
        with self._transition():
            record = self._tasks.get(task_id)
            if not record or not self._fence.accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                return EventEffect.STALE
            if record.status in SETTLING_TASK_STATUSES:
                # A replayed or late start must not regress a settling task.
                return EventEffect.SETTLED
            record.status = TaskStatus.DISPATCHED
            record.started_ts = started_ts
            if record.first_started_ts is None and payload.get("executing", True):
                record.first_started_ts = time.time()
            self._committer.commit_records_locked(record.workflow_id, [task_id])
            if engine := self._engines.get(record.workflow_id):
                engine.on_started(task_id)
                self._committer.save_ledger_locked(record.workflow_id)
            return EventEffect.APPLIED

    def mark_updated(
        self,
        task_id: str,
        worker_id: str,
        payload: dict[str, Any],
        dispatch_id: str | None = None,
    ) -> EventEffect:
        """Store a task's latest progress update."""
        with self._transition():
            record = self._tasks.get(task_id)
            if record is None or not self._fence.accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                return EventEffect.STALE
            if record.status in TERMINAL_TASK_STATUSES:
                # A replayed or late progress update must not touch a terminal task.
                return EventEffect.SETTLED
            record.latest_update = payload
            record.latest_update_dispatch_id = record.dispatch_id
            self._committer.persist_locked(task_id)
            return EventEffect.APPLIED

    def rewrite_update(
        self, task_id: str, previous: dict[str, Any], payload: dict[str, Any]
    ) -> bool:
        """Replace a task's latest update with a root rewrite of ``previous``.

        The rewrite keeps the dispatch that reported the update, and applies only while
        ``previous`` is still the latest update, so a worker update that landed since is
        never relabelled. Returns whether it applied.
        """
        with self._transition():
            record = self._tasks.get(task_id)
            if (
                record is None
                or record.status in TERMINAL_TASK_STATUSES
                or record.latest_update is not previous
            ):
                return False
            record.latest_update = payload
            self._committer.persist_locked(task_id)
            return True

    def mark_succeeded(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None = None,
        *,
        skip: dict[str, Any] | None = None,
    ) -> SettleOutcome:
        """
        Mark a task as completed and enqueue any dependents that have become ready.
        Returns what the report did to the task. ``skip`` records a conditional-skip
        settlement and why, which resolves a v2 output to an explicit-empty
        publication.

        The success binds the result reference it reports, once, at the commit that
        makes the task DONE; a success that does not settle the task binds nothing.
        """
        with self._transition(locked=False):
            return self._committer.reported(
                "TASK_SUCCEEDED",
                task_id,
                worker_id,
                dispatch_id,
                lambda: self._apply_success(
                    task_id, worker_id, payload, ts, dispatch_id, skip
                ),
            )

    def _apply_success(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None,
        skip: dict[str, Any] | None,
    ) -> SettleOutcome:
        reference = reported_reference(payload.get("result_reference"))
        child_references = reported_child_references(payload)
        fanout = self._prefetch_fanout(task_id, reference, skip)
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
            if record is not None and not self._fence.accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                if worker_id is not None:
                    self._fence.heal_returned_locked(task_id, worker_id, dispatch_id)
                    self._reap_stale_captures_locked(record, worker_id, payload)
                return SettleOutcome(EventEffect.STALE, record.status, [], [])
            effect = (
                EventEffect.SETTLED
                if record is not None and record.status in TERMINAL_TASK_STATUSES
                else EventEffect.APPLIED
            )
            if (prepared := payload.get("input_materialization")) is not None:
                # A preparation resolved the task's inputs and ran nothing else, so it
                # commits what it materialized and the task goes back to the queue for
                # the dispatch that chooses an embodiment and runs it.
                self._apply_input_materialization_locked(task_id, prepared)
                return settle_outcome(effect, record, [], [])
            episode_step = payload.get("agent_episode")
            if episode_step is not None and record is not None:
                harness_result = HarnessResult.model_validate(episode_step)
                # The holder seals its private state at the step's quiescence fence, so
                # the generation the next resume binds is recorded before the step is
                # routed and the episode can be re-dispatched.
                if (sealed := payload.get("agent_episode_private_state")) is not None:
                    self._apply_private_state_seal_locked(task_id, sealed)
                carried = payload.get("agent_episode_facade_group")
                carried_group = (
                    FacadeTurnGroup.model_validate(carried)
                    if carried is not None
                    else None
                )
                if record.status in TERMINAL_TASK_STATUSES:
                    # A late report of a dispatch its task already settled routes
                    # nothing it carries, so its worker drops what it holds for it.
                    self._file_locked(
                        task_id,
                        *self._mediated_ops.reap_captures_locked(
                            worker_id,
                            task_id,
                            _captured_calls(harness_result, carried_group),
                        ),
                    )
                    if harness_result.kind is not HarnessResultKind.COMPLETION:
                        return settle_outcome(effect, record, [], [])
                elif carried_group is not None:
                    # A facade group the worker captured on this turn rides the
                    # completion's own metadata on the durable task stream, so it is
                    # ingested here rather than on a separate channel that could
                    # deliver it after the completion (settling the episode DONE and
                    # dropping its searches) or drop it.
                    self.receive_worker_facade_group(task_id, carried_group)
                # The durable record is the source of truth: a restart drops the
                # in-memory stash, but a replayed completion still finds its captured
                # boundary and reroute rather than settling the episode DONE.
                group = self._pending_facade_groups.pop(task_id, None)
                if group is None:
                    group = record.pending_facade_group
                if harness_result.kind is not HarnessResultKind.COMPLETION:
                    # A non-terminal episode step routes its boundary and re-dispatches;
                    # a completion falls through to the terminal path below.
                    self._apply_episode_step_locked(task_id, harness_result, group)
                    self._reservations.release_ended_dispatches_locked([task_id])
                    return settle_outcome(
                        effect, record, [], in_flight_usage(task_id, payload)
                    )
                if record.status == TaskStatus.CANCELLING:
                    # A completion racing the cancel settles it before routing a
                    # captured facade group or consulting the reroute guard.
                    self._file_locked(
                        task_id,
                        *self._mediated_ops.reap_captures_locked(
                            worker_id, task_id, _captured_calls(harness_result, group)
                        ),
                    )
                    usages = self._settle_cancelled_usage_locked(
                        record, payload, finished_ts, started_ts
                    )
                    return settle_outcome(effect, record, [], usages)
                if group is not None:
                    # The gateway captured a turn-scoped facade group: the clean
                    # turn-completion is a yield on that group, not the episode's
                    # terminal result, so it routes the whole ordered membership
                    # kind-specifically, consumed durably in the same reroute save.
                    record.pending_facade_group = None
                    self._route_and_dispatch_facade_group_locked(
                        task_id, group, harness_result.capsule
                    )
                    self._reservations.release_ended_dispatches_locked([task_id])
                    return settle_outcome(
                        effect, record, [], in_flight_usage(task_id, payload)
                    )
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
                    return settle_outcome(EventEffect.STALE, record, [], [])
            if record:
                if record.status == TaskStatus.CANCELLED:
                    return settle_outcome(effect, record, [], usages)
                if record.status == TaskStatus.DONE:
                    # Idempotent: a replayed TASK_SUCCEEDED must not re-apply, but
                    # re-persist in case the original completion's write failed
                    # after its in-memory commit.
                    self._committer.recommit_terminal_locked(record)
                    # Recover a settlement or fan-out lost to a failed commit or a crash
                    # between the producer's terminal persist and its children: both
                    # are no-ops once applied.
                    if self._finish_success_locked(record, fanout):
                        self._cv.notify_all()
                    return settle_outcome(effect, record, [], [])
                if record.status == TaskStatus.FAILED:
                    self._logger.warning(
                        "Ignoring TASK_SUCCEEDED for task %s in terminal status FAILED",
                        task_id,
                    )
                    return settle_outcome(effect, record, [], [])
                if record.status == TaskStatus.CANCELLING:
                    # The cancel already resolved this task's declared output to its
                    # cancellation outcome and withheld the dispatch that would have
                    # carried a terminal back, so its completion settles the
                    # cancellation rather than reporting a success the ledger does
                    # not publish.
                    return settle_outcome(
                        effect,
                        record,
                        [],
                        self._settle_cancelled_usage_locked(
                            record, payload, finished_ts, started_ts
                        ),
                    )
                record.status = TaskStatus.DONE
                record.error = None
                record.finished_ts = finished_ts
                if started_ts:
                    record.started_ts = started_ts
                record.merged_children = None
                record.merged_dispatch_worker = None
                self._content_bindings.bind_result_locked(record, reference, skip)
                if usage is not None:
                    record.usages.append(usage)

            self._completed.add(task_id)
            self._failed.discard(task_id)
            self._dag.forget_pending(task_id)
            ready_children: list[str] = []
            merged_children_ids: list[str] = self._merges.take_children(task_id)
            self._ready.forget_merge_key(task_id)

            dependents = list(self._dag.take_dependents(task_id))
            for child in dependents:
                pending = self._dag.pending_deps.get(child)
                if pending is None:
                    continue
                self._dag.discard_dependency(child, task_id)
                if not pending:
                    child_record = self._tasks.get(child)
                    if child_record and child_record.status == TaskStatus.PENDING:
                        if self._ready.enqueue_ready_locked(child):
                            ready_children.append(child)

            settled_children, unsettled_children = (
                self._merges.partition_merged_children_locked(
                    merged_children_ids, child_references
                )
            )
            if record is not None:
                record.merged_children = settled_children or None
            for merged_child in settled_children:
                ready_children.extend(
                    self._merges.finalize_merged_child_success_locked(
                        merged_child,
                        worker_id,
                        finished_ts,
                        started_ts,
                        usage,
                        child_references[merged_child],
                    )
                )
            returned = self._merges.return_merged_children_locked(
                unsettled_children, unmerge=True
            )

            if record is not None:
                for workflow_id in self._merges.settle_workflows_locked(record):
                    ready_children.extend(
                        self._ready.try_advance_epoch_frontier_locked(workflow_id)
                    )

            self._committer.commit_locked(task_id, *settled_children, *returned)

            notify = bool(ready_children)
            if record is not None:
                notify |= self._finish_success_locked(record, fanout)
            if notify:
                self._cv.notify_all()

            return settle_outcome(effect, record, settled_children, usages)

    def _finish_success_locked(
        self, record: TaskRecord, fanout: FanoutRead | None
    ) -> bool:
        """Settle a committed success in the ledger and fan out what it produced.

        Returns whether it readied any task.
        """
        readied = False
        if engine := self._engines.get(record.workflow_id):
            advance = engine.on_succeeded(
                record.task_id,
                empty=record.result_skip is not None,
                content=record.result_reference,
            )
            advance.extend(
                self._fan_out_children_locked(
                    record.workflow_id, engine, record.task_id, fanout
                )
            )
            readied = self._apply_advance_locked(record.workflow_id, advance)
            self._committer.save_ledger_locked(record.workflow_id)
        self._committer.settle_if_done_locked(record.workflow_id)
        return readied

    def fail_dispatch(
        self,
        task_id: str,
        worker_id: str,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None = None,
        *,
        error: str | None = None,
        retryable: bool | None = None,
        failure_kind: TaskFailureKind | None = None,
        unavailable_inputs: Sequence[ContentReference] | None = None,
        ambiguous: bool = False,
    ) -> FailureOutcome:
        """Apply a worker's report that its dispatch of a task failed.

        A report from the dispatch holding a running task returns a merged dispatch to
        run its tasks alone. A task whose inputs were in a store its worker could not
        reach returns without spending an attempt and is held until control has read
        them itself; a report naming no input the task consumes is an ordinary failure.
        An ``ambiguous`` retryable v2 failure, after the task's external effect may
        have happened, settles as its worker's loss would. Otherwise the failure is
        charged to the worker and the task either returns to the head of the queue for
        another attempt or settles: FAILED, or CANCELLED when a cancel is already under
        way. A report on a settled task persists its settlement again, and one from any
        other dispatch is dropped.
        """
        # Control's verdict on a held task's input is a report of its own, so it never
        # replays the worker report of the same dispatch.
        report = (
            INPUT_VERDICT_REPORT
            if failure_kind is TaskFailureKind.INPUT_UNREADABLE
            else "TASK_FAILED"
        )
        with self._transition(locked=False):
            return self._committer.reported(
                report,
                task_id,
                worker_id,
                dispatch_id,
                lambda: self._apply_failure(
                    task_id,
                    worker_id,
                    payload,
                    ts,
                    dispatch_id,
                    error,
                    retryable,
                    failure_kind,
                    unavailable_inputs,
                    ambiguous,
                ),
            )

    def _apply_failure(
        self,
        task_id: str,
        worker_id: str,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None,
        error: str | None,
        retryable: bool | None,
        failure_kind: TaskFailureKind | None,
        unavailable_inputs: Sequence[ContentReference] | None,
        ambiguous: bool,
    ) -> FailureOutcome:
        with self._cv:
            record = self._tasks.get(task_id)
            if record is not None and self._inputs.is_input_verdict_locked(
                record, worker_id, dispatch_id, failure_kind
            ):
                # The worker's own report of this dispatch, stashed after a failed
                # write, is superseded: make what it held back and drop it, so its
                # redelivery finds the task settled.
                if (
                    (stash := self._committer.unacknowledged.get(task_id)) is not None
                    and stash.report == "TASK_FAILED"
                    and stash.worker_id == worker_id
                    and stash.dispatch_id in (None, dispatch_id)
                ):
                    self._committer.close_locked(record.workflow_id)
                    self._committer.drop_unacknowledged(task_id)
                record.last_error = error
                impacted, usages = self._mark_failed(
                    task_id, worker_id, payload, ts, error=error
                )
                return FailureOutcome(
                    DispatchEnd.FAILED, record.attempts, impacted, usages
                )
            if record is None or not self._fence.accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                self._fence.heal_returned_locked(task_id, worker_id, dispatch_id)
                return FailureOutcome(DispatchEnd.STALE, 0, [], [])
            if record.status in TERMINAL_TASK_STATUSES:
                self._committer.recommit_terminal_locked(record)
                return FailureOutcome(DispatchEnd.SETTLED, record.attempts, [], [])
            if self._fence.return_failed_merge_locked(record, worker_id):
                return FailureOutcome(
                    DispatchEnd.MERGE_RETURNED, record.attempts, [], []
                )
            if error:
                record.last_error = error
            if (
                failure_kind is TaskFailureKind.INPUT_UNAVAILABLE
                and record.status != TaskStatus.CANCELLING
            ):
                if consumed := self._content_bindings.consumed_inputs_locked(
                    record,
                    unavailable_inputs or (),
                ):
                    held_dispatch = record.dispatch_id
                    end = self._return_dispatch_locked(
                        record, increment_retry=False, front=False
                    )
                    self._inputs.hold_for_input_check_locked(
                        record, worker_id, held_dispatch, consumed
                    )
                    return FailureOutcome(end, record.attempts, [], [])
                self._logger.warning(
                    "Task %s: worker %s reported unreachable inputs the task does not "
                    "consume; charging the failure to it",
                    task_id,
                    worker_id,
                )
            if worker_id not in record.failed_workers:
                record.failed_workers.append(worker_id)
            if (
                ambiguous
                and retryable is not False
                and record.status == TaskStatus.DISPATCHED
                and record.workflow_id in self._engines
            ):
                return self._settle_ambiguous_failure_locked(record, payload, error)
            if failed_task_can_retry(record, retryable):
                end = self._return_dispatch_locked(
                    record, increment_retry=True, front=True
                )
                if end is DispatchEnd.RETURNED:
                    return FailureOutcome(end, record.attempts, [], [])
            impacted, usages = self._mark_failed(
                task_id, worker_id, payload, ts, error=record.last_error or error
            )
            end = (
                DispatchEnd.CANCELLED
                if record.status == TaskStatus.CANCELLED
                else DispatchEnd.FAILED
            )
            return FailureOutcome(end, record.attempts, impacted, usages)

    def _settle_ambiguous_failure_locked(
        self,
        record: TaskRecord,
        payload: dict[str, Any],
        error: str | None,
    ) -> FailureOutcome:
        """Settle a v2 failure reported after the task's external effect may have
        happened, as its worker's loss settles it: the task runs again only when its
        effect is safe to replay."""
        loss = self._resolve_lost_locked(
            record, spend_attempt=True, error=error or "task failed"
        )
        usages: list[tuple[str, TaskUsage]] = []
        if loss.end is DispatchEnd.FAILED and (
            usage := TaskUsage.from_payload(payload, TaskStatus.FAILED)
        ):
            record.usages.append(usage)
            usages.append((record.task_id, usage))
            self._committer.commit_locked(record.task_id)
        return FailureOutcome(loss.end, record.attempts, list(loss.impacted), usages)

    def return_dispatch(
        self,
        task_id: str,
        holder: str | None,
        *,
        increment_retry: bool,
        front: bool,
    ) -> DispatchEnd:
        """Return a task whose dispatch ended without a result to the ready queue.

        ``holder`` names the worker whose dispatch ended, when the caller knows it;
        only a task that worker holds is returned. A settled task stays as it is. A
        lost merged dispatch returns its tasks to run alone, and a task being cancelled
        settles CANCELLED, its merged children returning to run alone. A task whose
        last attempt this would spend is left for the caller to fail.
        """
        with self._transition():
            record = self._tasks.get(task_id)
            if record is None or not self._fence.holds_dispatch_locked(
                record, holder, None
            ):
                if holder is not None:
                    self._fence.heal_returned_locked(task_id, holder, None)
                return DispatchEnd.STALE
            if (
                holder is not None
                and self._episode_dispatch.dispatch_ended_at_suspension_locked(record)
            ):
                return DispatchEnd.STALE
            if record.status in TERMINAL_TASK_STATUSES:
                return DispatchEnd.SETTLED
            if record.status == TaskStatus.CANCELLING:
                self._settle_cancelled_locked(record, time.time(), unmerge=True)
                return DispatchEnd.CANCELLED
            if holder is not None and self._fence.return_failed_merge_locked(
                record, holder
            ):
                return DispatchEnd.MERGE_RETURNED
            return self._return_dispatch_locked(
                record, increment_retry=increment_retry, front=front
            )

    def _return_dispatch_locked(
        self, record: TaskRecord, *, increment_retry: bool, front: bool
    ) -> DispatchEnd:
        """Return a task to the ready queue, its merged children still mergeable."""
        task_id = record.task_id
        if increment_retry:
            record.attempts += 1
            if 0 <= record.max_attempts <= record.attempts:
                record.attempts = record.max_attempts
                return DispatchEnd.EXHAUSTED
        record.merged_children = None
        moved = [
            task_id,
            *self._merges.return_merged_children_locked(
                self._merges.take_children(task_id)
            ),
        ]
        engine = self._engines.get(record.workflow_id)
        changed = False
        if engine is not None:
            # A retry reuses the work item and its invocation; the engine records the
            # failed attempt, or closes an uncharged one, and readies the work item.
            if increment_retry:
                engine.on_failed(
                    task_id, record.last_error or "task failed", retryable=True
                )
                changed = True
            else:
                changed = engine.on_returned(task_id)
        self._release_dispatch_locked(record, moved, front=front)
        if engine is not None and changed:
            self._committer.save_ledger_locked(record.workflow_id)
        return DispatchEnd.RETURNED

    def _release_dispatch_locked(
        self, record: TaskRecord, moved: list[str], *, front: bool
    ) -> None:
        """Return a task whose dispatch ended to the ready queue and commit what moved.

        The return is complete in memory, queue included, before it commits, and a
        report of the same dispatch handled again commits it again.
        """
        task_id = record.task_id
        if record.assigned_worker is not None:
            self._fence.remember_return(
                task_id, (record.assigned_worker, record.dispatch_id, moved)
            )
        reset_to_pending(record)
        if self._ready.enqueue_ready_locked(task_id, front=front):
            self._cv.notify_all()
        self._committer.commit_locked(*moved)

    def _reap_stale_captures_locked(
        self, record: TaskRecord, worker_id: str, payload: dict[str, Any]
    ) -> None:
        """Reap the requests a stale agent step captured, which control never runs.

        A worker that holds the task again may have captured the same boundary anew,
        so its requests are left to that dispatch.
        """
        if self._fence.holds_dispatch_locked(record, worker_id, None):
            return
        step = payload.get("agent_episode")
        carried = payload.get("agent_episode_facade_group")
        reaps = self._mediated_ops.reap_captures_locked(
            worker_id,
            record.task_id,
            _captured_calls(
                HarnessResult.model_validate(step) if step is not None else None,
                (
                    FacadeTurnGroup.model_validate(carried)
                    if carried is not None
                    else None
                ),
            ),
        )
        self._file_locked(record.task_id, *reaps)

    def mark_failed(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        *,
        error: str | None = None,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, TaskUsage]]]:
        """
        Mark a task as failed. Dependent tasks still waiting on this task are
        automatically failed to avoid running without prerequisites, and merged
        children return to the queue to run on their own.

        Returns (impacted_dependents, usages).
        """
        with self._transition():
            return self._mark_failed(task_id, worker_id, payload, ts, error=error)

    def _mark_failed(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        *,
        error: str | None = None,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, TaskUsage]]]:
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
                    return [], usages
                if record.status == TaskStatus.FAILED:
                    # Idempotent: a replayed TASK_FAILED must not re-apply, but
                    # re-persist in case the original failure's write (including
                    # its cascade) failed after the in-memory commit.
                    self._committer.recommit_terminal_locked(record)
                    return [], []
                if record.status == TaskStatus.DONE:
                    self._logger.warning(
                        "Ignoring TASK_FAILED for task %s in terminal status DONE",
                        task_id,
                    )
                    return [], []
                if record.status == TaskStatus.CANCELLING:
                    # The cancellation is the settled outcome, so a failure racing it
                    # neither overrides it nor cascades into dependents the cancel has
                    # already settled.
                    return (
                        [],
                        self._settle_cancelled_usage_locked(
                            record, payload, finished_ts, started_ts, unmerge=True
                        ),
                    )
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
                record.merged_dispatch_worker = None
                if usage is not None:
                    record.usages.append(usage)

            self._failed.add(task_id)
            self._completed.discard(task_id)
            self._dag.forget_pending(task_id)
            self._ready.remove_from_ready_locked(task_id)
            merged_children_ids = self._merges.take_children(task_id)
            self._ready.forget_merge_key(task_id)

            impacted = self._record_failures.fail_v1_dependents_locked(task_id)

            engine = self._engines.get(record.workflow_id) if record else None
            advance = Advance()
            if engine is not None:
                advance = engine.on_failed(task_id, message, retryable=False)
                impacted.extend(
                    self._fail_v2_advance_locked(engine, advance, persist=False)
                )

            returned = self._merges.return_merged_children_locked(
                merged_children_ids, unmerge=True
            )

            failed_epoch = self._epochs.task_epoch_index.get(task_id)
            if record and failed_epoch is not None:
                blocked, blocked_returned = self._fail_later_epochs_locked(
                    record.workflow_id,
                    failed_epoch,
                    f"Blocked by failed task {task_id} in earlier epoch",
                )
                impacted.extend(blocked)
                returned += blocked_returned

            self._committer.commit_locked(
                task_id, *(dep_id for dep_id, _ in impacted), *returned
            )
            # The ledger writes after the task terminal records -- the retire a ready
            # applies included -- so a crash can only leave the ledger behind durable
            # task state, never ahead of it, and rehydration then reconciles it.
            if record is not None and engine is not None:
                # A child's failure can release its scope's join, readying what follows.
                if self._apply_advance_locked(
                    record.workflow_id,
                    Advance(ready=advance.ready, cancelled=advance.cancelled),
                ):
                    self._cv.notify_all()
                self._committer.save_ledger_locked(record.workflow_id)

            if record is not None:
                self._committer.settle_if_done_locked(record.workflow_id)
            return impacted, usages

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def cancel_workflow(self, workflow_id: str, reason: str = "cancelled") -> list[str]:
        """Cancel every unsettled task of a workflow, returning the tasks it moved."""
        touched: list[str] = []
        returned: list[str] = []
        with self._transition():
            workflow_tasks = [
                (record.task_id, record)
                for record in self._tasks.of_workflow(workflow_id)
            ]
            if not workflow_tasks:
                return touched  # Unknown workflow: no records to move
            owed = self._terminate_workflow_locked(workflow_id, reason, None)
            for task_id, record in workflow_tasks:
                if (moved := self._cancel_record_locked(record, reason)) is None:
                    continue
                returned += moved
                touched.append(task_id)
            self._committer.commit_cancelled_locked(workflow_id, touched, returned)
            # The ledger snapshot follows the committed task state so it never leads
            # it.
            if workflow_id in self._engines:
                self._settle_suspended_cancels_locked(self._engines[workflow_id])
                self._committer.save_ledger_locked(workflow_id)

            # A whole-workflow cancel commits its terminals here rather than through the
            # per-task terminal persist, so it notifies for itself. A cancel that leaves
            # tasks CANCELLING settles nothing yet; the finalizer reads that and waits
            # for their own terminals.
            self._actions.file_locked(
                workflow_id,
                *owed,
                Settled(workflow_id),
                Purge(workflow_id, settled_only=False),
            )
        return touched

    def _terminate_workflow_locked(
        self, workflow_id: str, reason: str, failure: str | None
    ) -> list[AfterCommit]:
        """Settle a workflow's ledger terminally and take what its work still holds.

        Runs before the caller moves the task records. A cancel (``failure`` None)
        resolves the unpublished outputs as cancelled and a control failure as declared
        failures. Either way a dispatch being published is recorded so its worker is
        interrupted with every other running task, the agents' mediated operations are
        taken for reaping, the pending re-drive and held input checks are dropped, and
        every unsettled boundary invocation is terminalized. It writes nothing: the
        caller's terminal commit persists it, and the caller files what it returns for
        delivery once that commit is durable.
        """
        self._redrive.settle(workflow_id)
        owed = self._take_task_work_locked(self._tasks.of_workflow(workflow_id), reason)
        self._epochs.forget_workflow(workflow_id)
        if (engine := self._engines.get(workflow_id)) is None:
            return owed
        if failure is None:
            engine.cancel_instance()
        else:
            engine.fail_instance(failure)
        return [*_failed_credits(engine.terminalize_unsettled_invocations()), *owed]

    def _take_task_work_locked(
        self, records: list[TaskRecord], reason: str
    ) -> list[AfterCommit]:
        """Take what the given tasks' work still holds, before a cancel moves them.

        A dispatch being published is recorded so its worker is interrupted with every
        other running task, the agents' mediated operations are taken for reaping, and
        the held input checks are dropped. It writes nothing.
        """
        owed: list[AfterCommit] = []
        for record in records:
            publish = self._fence.publishing.get(record.task_id)
            if publish and not publish.recorded and record.status == TaskStatus.PENDING:
                # The worker may already be running the task.
                self._fence.take_dispatch_locked(record, publish)
            if record.status == TaskStatus.DISPATCHED and (
                interrupt := self._actions.interrupt_for(record, reason)
            ):
                owed.append(interrupt)
            self._inputs.drop_check(record.task_id)
            self._epochs.forget_task(record.task_id)
        owed += self._mediated_ops.reap_ops_for_agents_locked(
            [r.task_id for r in records]
        )
        return owed

    def _settle_suspended_cancels_locked(self, engine: OrchestrationEngine) -> None:
        """Settle each cancelled episode suspended at a mediated boundary: it has no
        dispatch to interrupt and returns no terminal."""
        for suspended in engine.suspended_boundary_tasks():
            record = self._tasks.get(suspended)
            if record is not None and record.status == TaskStatus.CANCELLING:
                self._settle_cancelled_locked(record, time.time())

    def _cancel_residual_locked(
        self, workflow_id: str, engine: OrchestrationEngine, task_ids: list[str]
    ) -> bool:
        """Cancel the tasks of the children a residual policy cancelled in the ledger,
        as a workflow cancel moves its tasks; returns whether it moved any.

        What their work held -- a worker's interrupt, the agents' mediated operations,
        and the credits of their boundary invocations -- releases once the ledger is
        durable.
        """
        records = [
            record
            for task_id in dict.fromkeys(task_ids)
            if (record := self._tasks.get(task_id)) is not None
            and record.status not in SETTLING_TASK_STATUSES
        ]
        if not records:
            return False
        owed = self._take_task_work_locked(records, _RESIDUAL_CANCEL_REASON)
        touched: list[str] = []
        returned: list[str] = []
        for record in records:
            moved = self._cancel_record_locked(record, _RESIDUAL_CANCEL_REASON)
            if moved is None:
                continue
            record.residual_cancel = True
            touched.append(record.task_id)
            returned += moved
        self._committer.commit_cancelled_locked(workflow_id, touched, returned)
        self._settle_suspended_cancels_locked(engine)
        self._actions.file_locked(
            workflow_id,
            Settled(workflow_id),
            *_failed_credits(engine.terminalize_unsettled_invocations(touched)),
            *owed,
        )
        return bool(touched)

    def _cancel_record_locked(
        self, record: TaskRecord, reason: str
    ) -> list[str] | None:
        """Move a task a cancel reaches, in memory: a pending task, or a merged child
        whose batch keeps running for its siblings, settles CANCELLED in place, and a
        dispatched one goes CANCELLING until its worker's terminal settles it.

        Returns the children merged into it that go back to the queue, or None for a
        task a cancel leaves as it is.
        """
        match record.status:
            case TaskStatus.PENDING:
                return self._cancel_in_place_locked(record, reason)
            case TaskStatus.DISPATCHED if record.merged_parent_id:
                return self._cancel_in_place_locked(record, reason)
            case TaskStatus.DISPATCHED if record.assigned_worker:
                record.status = TaskStatus.CANCELLING
                record.error = reason
                return []
            case _:
                return None

    def _cancel_in_place_locked(self, record: TaskRecord, reason: str) -> list[str]:
        """Cancel a pending task or a merged child in place.

        Returns any children merged into it, which go back to the queue.
        """
        record.error = reason
        return self._mark_cancelled_locked(record, time.time())

    def _mark_cancelled_locked(
        self, record: TaskRecord, finished_ts: float, *, unmerge: bool = False
    ) -> list[str]:
        """Move a task to CANCELLED in memory, returning the children merged into it.

        The children stay mergeable, unless ``unmerge`` says the task's merged dispatch
        failed or was lost, which runs each of them alone.
        """
        task_id = record.task_id
        if (parent_id := self._merges.take_parent(task_id)) is not None:
            if (siblings := self._merges.merge_children_map.get(parent_id)) and (
                task_id in siblings
            ):
                self._merges.drop_merged_child(parent_id, task_id)
            if (parent := self._tasks.get(parent_id)) and parent.merged_children:
                parent.merged_children = [
                    child for child in parent.merged_children if child != task_id
                ] or None
        record.status = TaskStatus.CANCELLED
        record.finished_ts = finished_ts
        record.merged_children = None
        record.merged_dispatch_worker = None
        record.merged_parent_id = None
        # TODO(kaiitunnz): Handle usages for cancelled tasks
        self._completed.discard(task_id)
        self._failed.discard(task_id)
        self._dag.forget_pending(task_id)
        self._ready.remove_from_ready_locked(task_id)
        self._ready.merge_bucket_remove(task_id)
        self._ready.forget_merge_key(task_id)
        return self._merges.return_merged_children_locked(
            self._merges.take_children(task_id), unmerge
        )

    def mark_cancelled(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None = None,
    ) -> SettleOutcome:
        """Apply a worker's cancellation report; returns what it did to the task.

        A task being cancelled settles CANCELLED. A cancel nothing requested is the
        worker giving the task up, as a draining worker does, so the task returns
        without spending an attempt, or fails as on its worker's loss when it is a v2
        task that cannot safely re-run.
        """
        with self._transition(locked=False):
            return self._committer.reported(
                "TASK_CANCELLED",
                task_id,
                worker_id,
                dispatch_id,
                lambda: self._apply_cancellation(
                    task_id, worker_id, payload, ts, dispatch_id
                ),
            )

    def _apply_cancellation(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None,
    ) -> SettleOutcome:
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        usage = TaskUsage.from_payload(payload, TaskStatus.CANCELLED)
        usages: list[tuple[str, TaskUsage]] = []
        if usage is not None:
            usages.append((task_id, usage))

        with self._cv:
            record = self._tasks.get(task_id)
            if record is None or not self._fence.accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                return settle_outcome(EventEffect.STALE, record, [], [])
            if record.status == TaskStatus.CANCELLED:
                # Idempotent: a replayed cancellation must not re-apply, but
                # re-persist in case the original cancellation's write failed
                # after its in-memory commit.
                self._committer.recommit_terminal_locked(record)
                return settle_outcome(EventEffect.SETTLED, record, [], [])
            if record.status in (TaskStatus.DONE, TaskStatus.FAILED):
                self._logger.warning(
                    "Ignoring cancellation for task %s in terminal status %s",
                    task_id,
                    record.status,
                )
                return settle_outcome(EventEffect.SETTLED, record, [], [])
            if self._episode_dispatch.dispatch_ended_at_suspension_locked(record):
                return settle_outcome(EventEffect.STALE, record, [], [])
            if record.status == TaskStatus.DISPATCHED:
                return self._return_given_up_locked(record, worker_id, payload)
            self._settle_cancelled_locked(
                record, finished_ts, started_ts=started_ts, usage=usage
            )
            return settle_outcome(EventEffect.APPLIED, record, [], usages)

    def _return_given_up_locked(
        self, record: TaskRecord, worker_id: str | None, payload: dict[str, Any]
    ) -> SettleOutcome:
        """Return a task its worker gave up, without spending an attempt.

        A v2 task resolves as its worker's loss does: one that can safely re-run
        returns, and one that cannot fails, billed for the dispatch it gave up.
        """
        if record.workflow_id in self._engines:
            loss = self._resolve_lost_locked(record, spend_attempt=False)
            usages: list[tuple[str, TaskUsage]] = []
            if loss.end is DispatchEnd.FAILED and (
                usage := TaskUsage.from_payload(payload, TaskStatus.FAILED)
            ):
                record.usages.append(usage)
                self._committer.commit_locked(record.task_id)
                usages.append((record.task_id, usage))
            return settle_outcome(
                LOSS_EFFECTS[loss.end], record, [], usages, loss.impacted
            )
        if worker_id is None or not self._fence.return_failed_merge_locked(
            record, worker_id
        ):
            self._return_dispatch_locked(record, increment_retry=False, front=True)
        return settle_outcome(EventEffect.RETURNED, record, [], [])

    def _resolve_lost_locked(
        self, record: TaskRecord, *, spend_attempt: bool, error: str | None = None
    ) -> LossOutcome:
        """Resolve a v2 task whose worker is lost or gave it up, or whose executor
        failed after its external effect may have happened.

        It returns to the queue when it can safely run again, spending an attempt when
        ``spend_attempt`` is set, and fails once its attempts run out; a task that
        cannot safely run again fails. ``error`` is the executor's message for a
        reported failure, which the task fails with. ``impacted`` names each
        dependent that fails with it. A task nothing resolves ends STALE.
        """
        self._rehydrated_dispatched.pop(record.task_id, None)
        engine = self._engines.get(record.workflow_id)
        spent = False
        if (
            spend_attempt
            and engine is not None
            and engine.retries_on_loss(record.task_id)
        ):
            spent = True
            record.attempts += 1
            if 0 <= record.max_attempts <= record.attempts:
                record.attempts = record.max_attempts
                return self._fail_lost_on_last_attempt_locked(record, engine, error)
        advance = self._resolve_uncertain_locked(record.task_id, error)
        if advance.retry:
            return LossOutcome(record.task_id, DispatchEnd.RETURNED, (), spent)
        if not advance.failed:
            self._logger.warning(
                "Lost task %s of worker %s resolved to no outcome",
                record.task_id,
                record.assigned_worker,
            )
            return LossOutcome(record.task_id, DispatchEnd.STALE, ())
        impacted = tuple(
            (
                task_id,
                (engine.failure_reason(task_id) if engine is not None else None)
                or "declared-failure obligation",
            )
            for task_id in dict.fromkeys(advance.failed)
            if task_id != record.task_id
        )
        return LossOutcome(record.task_id, DispatchEnd.FAILED, impacted)

    def _fail_lost_on_last_attempt_locked(
        self,
        record: TaskRecord,
        engine: OrchestrationEngine,
        error: str | None = None,
    ) -> LossOutcome:
        """Fail a v2 task lost on its last attempt, with the boundary work it held, as
        a failed task: for its executor's ``error`` when it reported one, else for its
        worker's loss."""
        failed, _ = self._mark_failed(
            record.task_id,
            None,
            {},
            now_iso(),
            error=error
            or (
                f"Worker {record.assigned_worker} was lost on the last of "
                f"{record.max_attempts} attempts"
            ),
        )
        reaps = self._mediated_ops.reap_ops_for_agents_locked(
            [record.task_id, *(task_id for task_id, _ in failed)]
        )
        invocation_ids = engine.terminalize_unsettled_invocations([record.task_id])
        self._actions.file_locked(
            record.workflow_id, *_failed_credits(invocation_ids), *reaps
        )
        if invocation_ids:
            self._committer.save_ledger_locked(record.workflow_id)
        return LossOutcome(record.task_id, DispatchEnd.FAILED, tuple(failed))

    def _settle_cancelled_usage_locked(
        self,
        record: TaskRecord,
        payload: dict[str, Any],
        finished_ts: float,
        started_ts: float | None,
        *,
        unmerge: bool = False,
    ) -> list[tuple[str, TaskUsage]]:
        """Settle a cancellation a completion or failure raced, billing its dispatch.

        The dispatch that reports here ran to its end, so it is billed under the
        status the task settles into, not the one its report carried.
        """
        usage = TaskUsage.from_payload(payload, TaskStatus.CANCELLED)
        self._settle_cancelled_locked(
            record, finished_ts, started_ts=started_ts, usage=usage, unmerge=unmerge
        )
        return [(record.task_id, usage)] if usage is not None else []

    def _settle_cancelled_locked(
        self,
        record: TaskRecord,
        finished_ts: float,
        *,
        started_ts: float | None = None,
        usage: TaskUsage | None = None,
        unmerge: bool = False,
    ) -> None:
        """Settle a task being cancelled CANCELLED; the cancel that moved it, of its
        workflow or of its region's residual children, already settled its work item.

        A cancel interrupts the worker and waits for its terminal, but an interrupt
        cannot un-finish a dispatch the worker already completed: that dispatch reports
        a success, and the next one — which is what would carry the worker's terminal
        back — is the one a cancel withholds. A success landing on a CANCELLING task
        therefore settles the cancellation here rather than re-admitting the task or
        waiting for a terminal that never arrives.
        """
        task_id = record.task_id
        # A captured turn's group is meaningless once the episode settles cancelled.
        self._pending_facade_groups.pop(task_id, None)
        record.pending_facade_group = None
        if started_ts:
            record.started_ts = started_ts
        if usage is not None:
            record.usages.append(usage)
        returned = self._mark_cancelled_locked(record, finished_ts, unmerge=unmerge)
        # Persist the task terminal record first and snapshot the ledger last, so the
        # ledger never leads task state.
        self._committer.commit_locked(task_id, *returned, sched=False)
        self._committer.save_ledger_locked(record.workflow_id)
        self._committer.settle_if_done_locked(record.workflow_id)

    def get_record(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def private_state_holders(self) -> list[OwnerFence]:
        """The worker incarnations holding private state an unsettled activation
        resumes on."""
        with self._lock:
            return [
                holder
                for engine in self._engines.values()
                for holder in engine.private_state_holders()
            ]

    def long_lived_allocation(self, task_id: str) -> bool:
        """Whether a task holds its worker for its own life: a model server, resident
        or user-submitted."""
        with self._lock:
            record = self._tasks.get(task_id)
            return record is not None and (
                record.resident or record.task_type in SERVE_TASK_TYPES
            )

    def resident_dispatches_on(self, worker_id: str) -> list[tuple[str, str]]:
        """The resident serve tasks dispatched to ``worker_id``, each with its
        dispatch."""
        with self._lock:
            return self._resident_tasks.dispatches_on_locked(worker_id)

    def request_resident_yield(self, task_id: str, dispatch_id: str) -> bool:
        """Ask resident capacity to free the worker a resident serve task occupies
        under ``dispatch_id``.

        Returns whether the task is one resident capacity started, still on that
        dispatch, and was asked.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            if (
                record is None
                or not record.resident
                or record.status != TaskStatus.DISPATCHED
                or record.dispatch_id != dispatch_id
            ):
                return False
        return self._resident_tasks.request_yield(task_id)

    def live_resident_task_ids(self) -> set[str]:
        """Ids of the resident serve tasks neither settled nor cancelling."""
        with self._lock:
            return {
                task_id
                for task_id, record in self._tasks.items()
                if record.resident and record.status not in SETTLING_TASK_STATUSES
            }

    def workflow_submitted_at(self, workflow_id: str) -> str | None:
        """The workflow's durable submission timestamp, or ``None`` if unknown."""
        record = self._workflow_registry.get_workflow_record(workflow_id)
        return record.submitted_at if record is not None else None

    def set_completion_notifier(self, notify: Callable[[str], None]) -> None:
        """Install the callback that a terminal transition notifies once durable.

        The callback must be cheap and non-blocking: it runs on the thread that
        delivers the transition's actions.
        """
        self._committer.on_workflow_settled = notify

    def workflow_settlement(self, workflow_id: str) -> WorkflowSettlement:
        """Whether every task of a workflow has durably settled, and the last of their
        finishes.

        Read under the scheduler lock, so a caller never observes the moment inside a
        settle in which a producer's tasks have gone terminal but the children they
        fan out do not exist yet.
        """
        with self._lock:
            if not self._committer.durable(workflow_id):
                return WorkflowSettlement(settled=False, finished_ts=None)
            return self._committer.workflow_settlement_locked(workflow_id)

    def describe_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None
            return self._build_task_info_locked(task_id, record)

    def task_statuses(self) -> dict[str, str]:
        with self._lock:
            return {task_id: record.status for task_id, record in self._tasks.items()}

    def task_page(
        self,
        query: QueryFilter,
        limit: int,
        after: TaskOrder | None = None,
        before: TaskOrder | None = None,
        accessible: Collection[str] | None = None,
    ) -> list[TaskInfo]:
        """Return the tasks matching ``query``, ordered by submission: the ``limit``
        just after or before a position, or the newest ``limit``.

        One pass under the lock matches record attributes and copies out the page's
        fields, so the page is one consistent snapshot; building runs outside it.
        """
        workflow_ids = query.values("workflow_id")
        statuses = query.values("status")
        rest = query.without("workflow_id", "status")
        with self._lock:
            computed = self._listing_computed_locked()
            keys = sorted(
                task_order(record)
                for task_id, record in self._tasks.items()
                if (workflow_ids is None or record.workflow_id in workflow_ids)
                and (statuses is None or record.status in statuses)
                and (accessible is None or task_id in accessible)
                and (
                    not rest
                    or rest.matches(_listing_fields(task_id, record, rest, computed))
                )
            )
            window = page_slice(keys, limit, after=after, before=before, newest=True)
            fields = [
                self._task_info_fields_locked(task_id, self._tasks[task_id])
                for _, task_id in keys[window]
            ]
        return [TaskInfo(**task) for task in fields]

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    def task_records(self) -> list[TaskRecord]:
        """Copy every task record, detached, at one moment under the lock."""
        with self._lock:
            return [
                record.model_copy(
                    update={
                        name: value.copy()
                        for name, value in record
                        if isinstance(value, (list, dict))
                    }
                )
                for record in self._tasks.values()
            ]

    def work_item_id(self, task_id: str) -> str | None:
        """Return the id of the ledger work item a v2 task realizes, if it has one."""
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            return engine.work_item_id_for_task(task_id) if engine else None

    def recover_tasks_for_worker(
        self, worker_id: str, *, spend_attempt: bool, node_id: str | None = None
    ) -> WorkerRecovery:
        """Recover the tasks a departed worker held.

        Every dispatch recovered here is revoked on the worker, in case it is still
        reachable. A dispatch to the worker published and not yet recorded is lost
        here: its tasks go back to the head of the queue, a merged batch's to run
        alone, spending no attempt, and the dispatch is never recorded. A v2 task
        resolves as its worker's loss here, spending an attempt unless
        ``spend_attempt`` is False, as for a worker that gave its tasks up; a v1 task
        is left for the caller to return or settle. A task whose dispatch ended at a
        suspension holds nothing on the worker and waits on its boundary, unless the
        worker originated that boundary and holds its request. ``node_id`` names the
        worker's node when its record is already gone.
        """
        with self._transition():
            return self._recover_tasks_for_worker(worker_id, spend_attempt, node_id)

    def _recover_tasks_for_worker(
        self, worker_id: str, spend_attempt: bool, node_id: str | None
    ) -> WorkerRecovery:
        recovered: list[str] = []
        resolved: list[LossOutcome] = []
        with self._cv:
            for task_id, record in list(self._tasks.items()):
                publish = self._fence.publishing.get(task_id)
                if publish and not publish.recorded and publish.worker_id == worker_id:
                    self._fence.mark_publish_lost(task_id)
                    self._file_locked(
                        task_id,
                        self._actions.revoke_for(
                            task_id, worker_id, publish.dispatch_id, node_id
                        ),
                    )
                    children = self._merges.take_children(task_id)
                    record.merged_children = None
                    self._committer.commit_locked(
                        *self._merges.return_merged_children_locked(
                            [task_id, *children], unmerge=bool(children)
                        )
                    )
                    continue
                if record.assigned_worker != worker_id:
                    continue
                if record.status not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                    continue
                # A boundary whose raw request only this worker holds is lost with it.
                if self._episode_dispatch.dispatch_ended_at_suspension_locked(
                    record
                ) and not (
                    self._engines[record.workflow_id].awaits_worker_held_boundary(
                        task_id
                    )
                ):
                    continue
                self._rehydrated_dispatched.pop(task_id, None)
                if not record.merged_parent_id:
                    self._file_locked(
                        task_id,
                        self._actions.revoke_for(
                            task_id, worker_id, record.dispatch_id, node_id
                        ),
                    )
                if (
                    record.status == TaskStatus.DISPATCHED
                    and record.workflow_id in self._engines
                ):
                    resolved.append(
                        self._resolve_lost_locked(record, spend_attempt=spend_attempt)
                    )
                    continue
                recovered.append(task_id)
            # A pending tool operation on the departed worker lost its private request
            # custody with the worker; drop the stale mapping so its boundary re-mints
            # on a fresh worker rather than waiting on an outcome that can never arrive.
            self._mediated_ops.drop_worker_ops(worker_id)
        return WorkerRecovery(recovered, resolved)

    def resolve_disowned_dispatch(
        self, task_id: str, dispatch_id: str, worker_id: str, bound_sec: float
    ) -> SettleOutcome | None:
        """Resolve a dispatch its live worker keeps reporting it does not hold as lost.

        Only a dispatch recorded at least ``bound_sec`` ago with no event of it applied
        resolves: the bound a silent worker gets before it is declared dead. The
        dispatch is revoked, so a frame of it still queued for the worker never runs,
        and it resolves as a lost dispatch does. A task being cancelled settles
        CANCELLED; any other returns without spending an attempt, or fails as on its
        worker's loss when it is a v2 task that cannot safely re-run. The worker is
        excluded from the task's next placement, so a worker that cannot take the task
        never gets it back. A task bound to that worker's private state goes back to
        it, and its return spends an attempt. Returns None when the dispatch does not
        resolve.
        """
        with self._transition():
            record = self._tasks.get(task_id)
            if (
                record is None
                or record.status not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING)
                or record.assigned_worker != worker_id
                or record.dispatch_id != dispatch_id
                or record.started_ts is not None
                or not self._episode_dispatch.awaits_its_dispatch_locked(record)
            ):
                return None
            since = max(
                record.dispatched_ts or 0.0,
                self._rehydrated_dispatched.get(task_id, 0.0),
            )
            if time.time() - since < bound_sec:
                return None
            self._file_locked(
                task_id, self._actions.revoke_for(task_id, worker_id, dispatch_id, None)
            )
            if worker_id not in record.failed_workers:
                record.failed_workers.append(worker_id)
            record.last_error = (
                f"Worker {worker_id} kept reporting it does not hold dispatch "
                f"{dispatch_id}"
            )
            if record.status == TaskStatus.CANCELLING:
                self._settle_cancelled_locked(record, time.time(), unmerge=True)
                return settle_outcome(EventEffect.APPLIED, record, [], [])
            engine = self._engines.get(record.workflow_id)
            if engine is not None and engine.private_state_owner(task_id):
                return self._retry_on_owner_locked(record, worker_id)
            return self._return_given_up_locked(record, worker_id, {})

    def _retry_on_owner_locked(
        self, record: TaskRecord, worker_id: str
    ) -> SettleOutcome:
        """Return a task bound to its worker's private state, spending an attempt.

        The task can run only on that worker, so the attempt budget bounds how often
        the worker disowns it. The disowned dispatch is revoked; should a delivery of
        it still run, its events are fenced, and a retry that finds the private state
        it changed fails closed.
        """
        end = self._return_dispatch_locked(record, increment_retry=True, front=True)
        if end is DispatchEnd.RETURNED:
            return settle_outcome(EventEffect.RETURNED, record, [], [])._replace(
                spent=True
            )
        impacted, usages = self._mark_failed(
            record.task_id, worker_id, {}, now_iso(), error=record.last_error
        )
        return settle_outcome(EventEffect.FAILED, record, [], usages, tuple(impacted))

    def dispatch_in_flight(
        self, task_id: str, dispatch_id: str, worker_id: str
    ) -> bool:
        """Whether a dispatch to a worker is being published or holds its task, and has
        not ended at a suspension."""
        with self._lock:
            return self._fence.dispatch_in_flight_locked(
                task_id, dispatch_id, worker_id
            )

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
        self._redrive.stop()
        self._durability.stop()
        with self._cv:
            self._cv.notify_all()

    def ready_queue_length(self) -> int:
        with self._cv:
            return len(self._ready.ready_queue)

    def task_status_counts(self) -> tuple[int, int, int, int, int]:
        with self._cv:
            queueing = len(self._ready.ready_queue)
            dispatched = 0
            pending = 0
            done = 0
            for task_id, record in self._tasks.items():
                status = record.status
                if status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                    dispatched += 1
                elif status == TaskStatus.DONE:
                    done += 1
                elif (
                    status == TaskStatus.PENDING
                    and task_id not in self._ready.ready_index
                ):
                    pending += 1
            total = len(self._tasks)
            return queueing, dispatched, pending, done, total

    def _listing_computed_locked(self) -> dict[str, Callable[[str], Any]]:
        empty: frozenset[str] = frozenset()
        return {
            "completed": self._completed.__contains__,
            "failed": self._failed.__contains__,
            "depends_on": lambda task_id: self._original_deps.get(task_id, empty),
            "pending_dependencies": lambda task_id: self._dag.pending_deps.get(
                task_id, empty
            ),
            "dependents": lambda task_id: self._dag.dependents.get(task_id, empty),
        }

    def _build_task_info_locked(self, task_id: str, record: TaskRecord) -> TaskInfo:
        return TaskInfo(**self._task_info_fields_locked(task_id, record))

    def _task_info_fields_locked(
        self, task_id: str, record: TaskRecord
    ) -> dict[str, Any]:
        """Return a task's ``TaskInfo`` fields to build from after the lock is
        released, duplicating the containers a record appends to in place."""
        element = self._content_bindings.input_element_locked(task_id)
        engine = self._engines.get(record.workflow_id)
        place = engine.occurrence_place(task_id) if engine is not None else None
        return {
            **{
                name: value.copy() if isinstance(value, (list, dict)) else value
                for name, value in record
            },
            "depends_on": sorted(self._original_deps.get(task_id, set())),
            "pending_dependencies": sorted(self._dag.pending_deps.get(task_id, set())),
            "dependents": sorted(self._dag.dependents.get(task_id, set())),
            "completed": task_id in self._completed,
            "failed": task_id in self._failed,
            "input_element": (
                TaskInputElement(
                    producer_task_id=element.producer_task_id,
                    index=element.ref.element,
                    path=list(element.ref.path),
                )
                if element is not None
                else None
            ),
            "occurrence": (
                TaskOccurrence(
                    member=place.member,
                    context=place.context,
                    time=[
                        TaskLoopTime(loop=loop, iteration=iteration)
                        for loop, iteration in place.time
                    ],
                )
                if place is not None
                else None
            ),
        }
