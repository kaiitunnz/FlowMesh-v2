import heapq
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Any, Self, cast

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
    EpisodeModelBinding,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    InputBinding,
    InputBindingMember,
    ServiceLeafEpisodeDispatch,
)
from shared.inference import (
    CanonicalInferenceContract,
    CanonicalProjectionError,
    InferenceSourceKind,
    InputResolutionBinding,
    ResolvedInputMaterialization,
    canonical_contract,
    element_contract,
)
from shared.outcome import OutcomeManifest
from shared.private_state import (
    OwnerFence,
    PrivateStateAttachment,
    PrivateStateSealReport,
    PrivateStateUnavailable,
)
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentOpOutcome,
    ResidentRouteObservation,
)
from shared.sandbox import (
    SANDBOX_EGRESS_INTERFACE,
    SANDBOX_EXECUTE_INTERFACE,
    LocalSandboxCapability,
    SandboxEgressMode,
)
from shared.schemas.command import InterruptMessage, MediatedOpMessage, RevokeMessage
from shared.schemas.event import TaskEvent, TaskFailureKind
from shared.schemas.result import ResultEnvelope
from shared.schemas.result.binding import collection_elements, value_text
from shared.tasks import TaskEnvelopeTemplate
from shared.tasks.credentials import set_spec_values
from shared.tasks.result_binding import (
    ResultBinding,
    ResultElementRef,
    ResultValueRef,
)
from shared.tasks.specs import (
    InferenceEmbodimentKind,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    ModelBindingMode,
    TaskSpecBase,
)
from shared.telemetry.config import TelemetryConfig
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.telemetry.semconv import ControlPlaneStage, ControlPlaneWindow
from shared.tools.contract import (
    AgentModelTurnProposal,
    MediatedOperationOutcome,
    MediatedOperationPermit,
)
from shared.tools.facade import FacadeDescriptor, FacadeResolution
from shared.utils import new_workflow_id
from shared.utils.redact import credential_scrubber

from ..config import AgentBindingConfig, N8nConfig, OrchestrationConfig
from ..hooks import SUPPLIER_RESOLVERS
from ..orchestration import (
    AcceptedInput,
    AcceptedInputMember,
    Advance,
    InputResolution,
    LedgerSnapshot,
    OrchestrationEngine,
    PublicationOutcome,
    RecoveryDisposition,
    RegionError,
    ResultPublication,
    ScopeBudget,
    ValueRef,
    WorkItemStatus,
    dependency_failed,
)
from ..orchestration.episode import BoundaryEvent
from ..orchestration.harness import to_boundary_event
from ..orchestration.state import TERMINAL_WORK_ITEM_STATUSES
from ..orchestration.telemetry import build_span_emitter
from ..orchestration.tool_dispatch import (
    MODEL_INTERFACE,
    SEARCH_INTERFACE,
    FacadeTurnGroup,
    InputMemberPlan,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from ..registries.worker import Worker, WorkerRegistry
from ..registries.workflow import PersistedTask, WorkflowRegistry, WorkflowSched
from ..services.credential_vault import CredentialVault
from ..utils.time import now_iso, parse_iso_ts, ts_to_iso
from .credentials import (
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
from .models import (
    SETTLING_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    DispatchEnd,
    EventEffect,
    FailureOutcome,
    LossOutcome,
    SettleOutcome,
    TaskInfo,
    TaskInputElement,
    TaskParsingResult,
    TaskRecord,
    TaskStatus,
    TaskUsage,
    WorkerRecovery,
    WorkflowSettlement,
    categorize_task_type,
)
from .outputs import (
    OutputMember,
    PublishedOutput,
    PublishedOutputs,
    published_member,
    published_members,
)
from .parser import ParsedWorkflow, parse_workflow
from .redrive import StoreRedriveScheduler
from .results import ResultReader, ResultUnavailable, ResultUnreadable
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
from .v2.compiler.facades import run_command_schema
from .v2.policy import PolicySurface
from .v2.representations.admission import ResidentAdmissionBinding
from .v2.representations.operators import (
    AgentModelGatewayBinding,
    AgentOperator,
    ResolvedEmbodiment,
    ServiceDependency,
)
from .v2.representations.plan import EpisodeSpec, InferenceEmbodimentMenu

# A live-feasibility check: whether a lowered episode's declared alternative can be
# placed now.
EpisodeFeasibility = Callable[[EpisodeSpec], bool]


def _sandbox_capability(
    op: AgentOperator | None,
    attachment: PrivateStateAttachment | None,
    invoke_face: tuple[str, ...],
) -> LocalSandboxCapability | None:
    """The local execution authority for one dispatch of a sandbox-declaring agent.

    It is minted from the agent's pinned envelope and the attachment that already fences
    this dispatch's writes, so a command runs only under the holder and write epoch that
    owns the workspace it mutates. An agent with no attachment has no workspace to run
    in and gets none.

    Both interfaces are resolved against ``invoke_face`` — what this activation may
    actually invoke once its ancestors have attenuated it — rather than against the
    pinned binding, which only records what the author asked for. A child whose parent
    withheld ``sandbox.execute`` gets no capability at all and cannot run a command; one
    whose parent withheld only ``sandbox.egress`` runs fenced under a binding that names
    the opt-in. Neither a binding nor a command argument can widen what is minted here.
    """
    if op is None or op.sandbox_binding is None or attachment is None:
        return None
    if SANDBOX_EXECUTE_INTERFACE not in invoke_face:
        return None
    egress = (
        op.sandbox_binding.network_egress
        if SANDBOX_EGRESS_INTERFACE in invoke_face
        else SandboxEgressMode.DENY
    )
    return LocalSandboxCapability(
        attachment_id=attachment.attachment_id,
        reference_id=attachment.reference_id,
        worker_id=attachment.worker_id,
        incarnation=attachment.incarnation,
        write_epoch=attachment.write_epoch,
        profile=op.sandbox_binding.profile,
        network_egress=egress,
    )


def _effective_facades(
    op: AgentOperator | None,
    invoke_face: tuple[str, ...],
    sandbox: LocalSandboxCapability | None,
) -> tuple[FacadeDescriptor, ...]:
    """The facades this dispatch offers the model, narrowed to what it may use.

    The compiler pins the ceiling from the operator's declared authority; an
    activation's effective grant can be narrower, so a locally-resolved facade is
    reconciled here against it. Only ``LOCAL_INLINE`` facades are narrowed: a mediated
    call the activation may not invoke settles as a durable authority denial, which is a
    record worth keeping, while a local one is refused inside the worker and would leave
    no trace of the offer at all.
    """
    if op is None:
        return ()
    facades: list[FacadeDescriptor] = []
    for facade in op.facades:
        if facade.resolution is not FacadeResolution.LOCAL_INLINE:
            facades.append(facade)
            continue
        if facade.interface == SANDBOX_EXECUTE_INTERFACE:
            if sandbox is None:
                continue  # no effective execute: the tool is never offered
            facade = facade.model_copy(
                update={"tool_schema": run_command_schema(sandbox.egress_allowed)}
            )
        elif facade.interface is not None and facade.interface not in invoke_face:
            continue
        facades.append(facade)
    return tuple(facades)


# A spawn producer's fan-out read retries off the lock before the workflow fails.
_FANOUT_READ_ATTEMPTS = 3
_FANOUT_READ_BACKOFF_SEC = 0.2


@dataclass(frozen=True)
class _FanoutRead:
    """How many elements a spawn producer's collection holds, or why it went unread."""

    count: int = 0
    # The stored result the count was read from; None for a producer that skipped.
    reference: ContentReference | None = None
    error: str | None = None
    # The store could not be reached; the collection is still there to read later.
    unavailable: bool = False


@dataclass(frozen=True)
class _InputCheck:
    """A task held while control checks the inputs its worker could not read."""

    worker_id: str
    dispatch_id: str | None
    references: tuple[ContentReference, ...]
    # Why control found an input unreadable, once it has; the task then fails as a
    # report of the dispatch that could not read it.
    unreadable: str | None = None


_INPUT_VERDICT_REPORT = "TASK_FAILED:input_unreadable"
_RESIDUAL_CANCEL_REASON = "cancelled by its region's residual policy"


def _input_verdict(task_id: str, check: _InputCheck) -> TaskEvent:
    """Control's report that a held task's input is unreadable, as the dispatch that
    could not read it."""
    return TaskEvent(
        type="TASK_FAILED",
        task_id=task_id,
        worker_id=check.worker_id,
        dispatch_id=check.dispatch_id,
        error=f"input_unreadable: {check.unreadable}",
        retryable=False,
        failure_kind=TaskFailureKind.INPUT_UNREADABLE,
    )


@dataclass(frozen=True)
class _StagedRegistration:
    """A submission's records and plan, built in memory before its durable write."""

    results: list[TaskParsingResult]
    task_records: list[TaskRecord]
    candidate_ready: list[str]
    v2_bundle: PersistedV2Workflow | None
    v2_engine: OrchestrationEngine | None


@dataclass
class _Termination:
    """What a terminated workflow's work still holds, released after its terminal."""

    interrupts: list[InterruptMessage]
    # Each pending mediated operation's worker, agent task, and call.
    reaps: list[tuple[str, str, str]]
    resident_invocation_ids: list[str] = field(default_factory=list)
    revokes: list[RevokeMessage] = field(default_factory=list)


@dataclass(frozen=True)
class _InputElement:
    """The producer element a leaf fan-out child runs on."""

    producer_task_id: str
    ref: ResultElementRef


@dataclass(frozen=True)
class _PortSnapshot:
    """One agent input port whose members are each frozen to what supplies them."""

    target_port: str
    provenance: str
    members: tuple[tuple[InputMemberPlan, ValueRef], ...]


@dataclass(frozen=True)
class _AgentInputSnapshot:
    """An agent's pending input ports as of one look at the ledger."""

    task_id: str
    activation_id: str
    ports: tuple[_PortSnapshot, ...]
    # Why the agent's input can never be read, when a producer settled unbound.
    unreadable: str | None = None

    @property
    def references(self) -> dict[str, ContentReference]:
        """Each stored result the ports read, keyed by the producer it names."""
        return {
            value_ref.legacy_task_id or "": value_ref.content
            for port in self.ports
            for _member, value_ref in port.members
            if value_ref.content is not None
        }


def _literal_text(value_ref: ValueRef | None) -> str | None:
    """The value an inline input member carries itself."""
    if value_ref is None:
        return None
    if value_ref.kind == "inline":
        return value_ref.literal or ""
    if value_ref.kind == "empty":
        return ""
    return None


def _member_text(
    value_ref: ValueRef, values: dict[ContentReference, ResultEnvelope | Exception]
) -> str | None:
    """The string an input member resolves to from the results read for it."""
    if value_ref.content is None:
        return _literal_text(value_ref)
    envelope = values.get(value_ref.content)
    if not isinstance(envelope, ResultEnvelope):
        return None
    return value_text(envelope, _element_of(value_ref))


def _element_of(value_ref: ValueRef) -> int | None:
    """The collection member a value reference selects, or None for the whole result."""
    return (
        int(value_ref.collection_key) if value_ref.collection_key is not None else None
    )


def _settled_at(record: TaskRecord) -> str | None:
    return ts_to_iso(record.finished_ts) if record.finished_ts is not None else None


def _reported_reference(raw: Any) -> ContentReference | None:
    """A result reference a worker reported, or None when it reported none."""
    if raw is None:
        return None
    try:
        return ContentReference.model_validate(raw)
    except ValidationError:
        return None


def _reset_to_pending(record: TaskRecord) -> None:
    """Clear what a task's last dispatch left on it, returning it to PENDING."""
    record.status = TaskStatus.PENDING
    record.assigned_worker = None
    record.dispatch_id = None
    record.topic = None
    record.dispatched_ts = None
    record.started_ts = None
    record.finished_ts = None
    record.error = None


def _membership(record: TaskRecord) -> str:
    """The status set a task's record commits into. A task being cancelled still runs
    on its worker. A child its region's residual policy cancelled settles its workflow
    as a finished task does, never as a cancelled one."""
    if record.status == TaskStatus.CANCELLING:
        return TaskStatus.DISPATCHED
    if record.status == TaskStatus.CANCELLED and record.residual_cancel:
        return TaskStatus.DONE
    return record.status


def _failed_task_can_retry(record: TaskRecord, retryable: bool | None) -> bool:
    """Whether a failed task may be requeued: retryable, within the attempt budget,
    and not settling."""
    if record.status in SETTLING_TASK_STATUSES or retryable is False:
        return False
    return record.max_attempts < 0 or record.attempts < record.max_attempts


def _settle_outcome(
    effect: EventEffect,
    record: TaskRecord | None,
    merged_children: list[str],
    usages: list[tuple[str, TaskUsage]],
    impacted: tuple[tuple[str, str], ...] = (),
) -> SettleOutcome:
    return SettleOutcome(
        effect,
        record.status if record is not None else None,
        usages if effect in (EventEffect.APPLIED, EventEffect.FAILED) else [],
        merged_children,
        impacted,
    )


_LOSS_EFFECTS = {
    DispatchEnd.RETURNED: EventEffect.RETURNED,
    DispatchEnd.FAILED: EventEffect.FAILED,
    DispatchEnd.STALE: EventEffect.STALE,
}


@dataclass
class _Publish:
    """A dispatch published to a worker: what recording it takes, and how far its
    worker's reports have taken it."""

    worker_id: str
    dispatch_id: str | None
    supplier_id: str
    input_preparation: bool
    recorded: bool = False
    reported: bool = False


@dataclass
class _HeldWrites:
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
    held: _HeldWrites | None = None


@dataclass
class _Unacknowledged:
    """A report whose transition completed in memory and failed a durable write: what
    it did to the task, and the writes it held back."""

    report: str
    worker_id: str
    dispatch_id: str | None
    held: _HeldWrites
    outcome: SettleOutcome | FailureOutcome


def _supplier_id(worker: Worker) -> str:
    for resolver in SUPPLIER_RESOLVERS:
        if (resolved := resolver.resolve(worker)) is not None:
            return resolved
    return ""


def _reported_child_references(payload: dict[str, Any]) -> dict[str, ContentReference]:
    """The result references a merged dispatch reported for its children, by child."""
    return {
        str(child_id): parsed
        for child_id, raw in (payload.get("child_result_references") or {}).items()
        if (parsed := _reported_reference(raw)) is not None
    }


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
# A generous bound on a materialized external-model completion; a larger response
# settles by reference under the reference-backed outcome contract.
_MODEL_PERMIT_RESULT_CHAR_CAP = 1_000_000


def _in_flight_usage(
    task_id: str, payload: dict[str, Any]
) -> list[tuple[str, TaskUsage]]:
    """The usage row for a dispatch whose task runs on rather than settling."""
    usage = TaskUsage.from_payload(payload, TaskStatus.DISPATCHED)
    return [(task_id, usage)] if usage is not None else []


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
    ) -> None:
        self._workflow_registry = workflow_registry
        self._worker_registry = worker_registry
        self._logger = logger
        self._results = results
        self._redrive = redrive(self._drive_workflow, logger)
        self._feasibility_check = feasibility_check
        self._policy_surface = surface if surface is not None else PolicySurface()
        self._credential_vault = credential_vault
        self._content_scope_authority = content_scope_authority
        self._n8n_credential_password = (n8n or N8nConfig()).credential_password
        self._control = control if control is not None else NULL_CONTROL_TRACER
        self._tracer = tracer
        self._telemetry = telemetry
        self._scope_budget = ScopeBudget.from_config(orchestration)
        self._web_search = orchestration.web_search
        self._model_egress_timeout_sec = orchestration.gateway.timeout_sec
        self._input_budget_bytes = orchestration.agent_input_budget_bytes
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
        # The dispatch being published for each task, until the dispatcher records it;
        # None when its worker was lost first.
        self._publishing: dict[str, _Publish | None] = {}
        # The last dispatch of each task that ended by returning it to the queue: its
        # worker and dispatch id, and the tasks the return moved.
        self._returned_dispatches: dict[str, tuple[str, str | None, list[str]]] = {}
        # The worker and dispatch holding each dispatched task, until a commit moves the
        # task off it and releases the worker's reservation for it.
        self._held_dispatches: dict[str, tuple[str, str]] = {}
        # Reservations of dispatches that ended, released after the lock; one whose
        # release failed stays here for the next release.
        self._ended_dispatches: list[tuple[str, str]] = []
        self._input_checks: dict[str, _InputCheck] = {}
        self._report_writes = _ReportWrites()
        self._unacknowledged: dict[str, _Unacknowledged] = {}
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
        self._failure_reporter: Callable[[TaskEvent], None] | None = None
        self._pending_terminations: list[_Termination] = []
        # Terminations whose workflow's ledger has not been saved since: each releases
        # only after that save succeeds.
        self._undurable_terminations: dict[str, list[_Termination]] = {}
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
        # Worker-originated tool operations whose permit was relayed to the origin
        # worker's egress sidecar, keyed by permit id -> (agent task, call, worker).
        # Reaps custody on settle or cancel. In-memory and rebuilt on restart from the
        # pending boundary, never durably persisted.
        self._pending_ops: dict[str, tuple[str, str, str]] = {}

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._on_workflow_settled: Callable[[str], None] | None = None

    # ------------------------------------------------------------------ #
    # Registration & submission
    # ------------------------------------------------------------------ #

    def _parse(self, payload: str, format: str) -> ParsedWorkflow:
        return parse_workflow(payload, format, self._n8n_credential_password)

    def validate(self, payload: str, format: str = "native") -> list[TaskParsingResult]:
        parsed_workflow = self._parse(payload, format)
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
        parsed_workflow = self._parse(payload, format)
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
        # Once the durable write is attempted the workflow may exist, so a later failure
        # keeps its credentials; the startup sweep reclaims a vault no workflow owns.
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
        return workflow_id, await self._commit_registration(
            workflow_id, submitted_at, staged
        )

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
        candidate_ready: list[str] = []
        graph_task_ids: dict[str, str] = {}

        v2_bundle: PersistedV2Workflow | None = None
        v2_engine: OrchestrationEngine | None = None
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
                task_credentials = credentials.tasks.get(task_id, TaskCredentials())
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

        return _StagedRegistration(
            results, task_records, candidate_ready, v2_bundle, v2_engine
        )

    async def _commit_registration(
        self, workflow_id: str, submitted_at: str, staged: _StagedRegistration
    ) -> list[TaskParsingResult]:
        task_records = staged.task_records
        v2_engine = staged.v2_engine
        await self._workflow_registry.register_workflow_async(
            workflow_id, task_records, v2=staged.v2_bundle, submitted_at=submitted_at
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
                with self._control.workflow_stage(
                    ControlPlaneStage.DS_INITIAL_ADVANCE,
                    ControlPlaneWindow.SUBMIT,
                    workflow_id,
                ):
                    advance_applied = self._apply_advance_locked(
                        workflow_id, v2_engine.initial_advance()
                    )
                if advance_applied:
                    new_ready = True
                # Saved under the lock after the initial advance persists any
                # authority-denied roots, so the ledger never leads durable task state
                # and no later save lands before it.
                self._save_ledger_locked(workflow_id)
            for task_id in staged.candidate_ready:
                maybe_record = self._tasks.get(task_id)
                if not maybe_record or maybe_record.status != TaskStatus.PENDING:
                    continue
                if self._pending_deps.get(task_id):
                    continue
                if self._enqueue_ready_locked(task_id):
                    new_ready = True
            if new_ready:
                self._cv.notify_all()

        return staged.results

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
            await self._vault_stored_credentials(workflow_id, tasks)
            remaining = await self._workflow_registry.get_remaining_tasks_async(
                workflow_id
            )
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
                # A non-terminal record the remaining set no longer lists was
                # retired before the crash, and nothing will ever dispatch it; the
                # durable set is what carries that fact across a restart.
                self._retired_region_templates.setdefault(workflow_id, set()).update(
                    persisted.record.task_id
                    for persisted in tasks
                    if persisted.record.status not in TERMINAL_TASK_STATUSES
                    and persisted.record.task_id not in remaining
                )
                if snapshot is not None and bundle is not None:
                    self._install_rehydrated_v2_workflow_locked(
                        workflow_id, tasks, snapshot, bundle, rehydrated_at
                    )
                else:
                    self._install_rehydrated_workflow_locked(
                        workflow_id, tasks, sched, rehydrated_at
                    )
                self._interrupt_cancelling_locked(workflow_id)
                # A workflow whose last task settled just before the crash has no
                # event left to close it: replay the completion notification for
                # every restored workflow, and let the finalizer reject the ones
                # still running or already closed.
                self._notify_terminal_transition(workflow_id)
                self._cv.notify_all()
            restored.append(workflow_id)
        with self._cv:
            self._restore_merges_locked()
            self._held_dispatches.update(
                (record.task_id, (record.assigned_worker, record.dispatch_id))
                for record in self._tasks.values()
                if _membership(record) == TaskStatus.DISPATCHED
                and record.assigned_worker is not None
                and record.dispatch_id is not None
                and not self._dispatch_ended_at_suspension_locked(record)
            )
            live = [
                workflow_id
                for workflow_id in restored
                if not self._workflow_settlement_locked(workflow_id).settled
            ]
        self._release_pending_terminations()
        # Rehydrate completes before the API accepts a submission, so no workflow is
        # between vaulting its credentials and registering.
        await self._credential_vault.retain_only(live)
        if restored:
            self._logger.info(
                "Rehydrated %d workflow(s) from durable state", len(restored)
            )
        return len(restored)

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

    def _restore_merges_locked(self) -> None:
        """Rebuild the in-flight merges from durable records.

        A merge may span workflows, so it is rebuilt once every workflow is restored.
        A merge whose parent never dispatched returns its children to the queue, and
        one whose parent settled returns them to run alone.
        """
        for child_id, record in self._tasks.items():
            if record.merged_parent_id and record.status == TaskStatus.DISPATCHED:
                self._merge_parent_map[child_id] = record.merged_parent_id
                self._merge_children_map[record.merged_parent_id].append(child_id)
        parents = {
            *self._merge_children_map,
            *(
                task_id
                for task_id, rec in self._tasks.items()
                if rec.merged_children and rec.status not in TERMINAL_TASK_STATUSES
            ),
        }
        for parent_id in parents:
            parent = self._tasks.get(parent_id)
            if parent is None or parent.status in TERMINAL_TASK_STATUSES:
                children = self._merge_children_map.pop(parent_id, [])
                self._commit_locked(
                    *self._return_merged_children_locked(children, unmerge=True)
                )
            elif parent.status not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                self._release_merge_locked(parent_id)

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
            self._commit_locked(*moved)

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

        engine = OrchestrationEngine(
            snapshot,
            bundle,
            budget=self._scope_budget,
            control=self._control,
            emitter=build_span_emitter(self._tracer, self._telemetry, workflow_id),
        )
        self._engines[workflow_id] = engine
        cancelled = False
        for persisted in tasks:
            record = persisted.record
            if record.status == TaskStatus.DONE:
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
                self._enqueue_ready_locked(record.task_id)
            elif (worker_id := engine.suspending_worker(record.task_id)) is not None:
                # A crash beat the ledger save of a boundary's settle, which had
                # already returned the record to PENDING. The re-issued boundary
                # reaches the worker that captured it through the record.
                record.status = TaskStatus.DISPATCHED
                record.assigned_worker = worker_id
                self._rehydrated_dispatched[record.task_id] = rehydrated_at
                self._commit_locked(record.task_id)

        # Re-drive any DONE producer whose spawn never sealed and any agent waiting on
        # its bound inputs: their terminal events do not replay, so nothing else
        # materializes the children or records the inputs.
        if engine.blocked_input_agents() or any(
            persisted.record.status == TaskStatus.DONE
            and (spawn_op := engine.spawn_successor(persisted.record.task_id))
            is not None
            and engine.spawn_awaits_children(spawn_op)
            for persisted in tasks
        ):
            self._redrive.drive_now(workflow_id)

        # Re-issue the off-lane dispatch for any mediated boundary suspended with no
        # durable outcome: the handler ran on an in-memory executor a crash discarded,
        # so nothing else resumes the agent. The durable envelope routes it to the same
        # handler (a search to the broker, a model to the gateway) it was recorded for.
        for envelope in engine.pending_tool_dispatches():
            self._dispatch_boundary(envelope)
        # The replayed terminals can seal a region whose retire a crash lost.
        self._retire_sealed_region_templates_locked(workflow_id, engine)
        self._save_ledger_locked(workflow_id)

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

    def _commit_locked(self, *task_ids: str, sched: bool = True) -> None:
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
                    moves[record.workflow_id][_membership(record)].append(task_id)
            for workflow_id, by_status in moves.items():
                self._workflow_registry.commit_transition(
                    workflow_id,
                    records=self._records_locked(
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
                    self._notify_terminal_transition(workflow_id)
            self._release_ended_dispatches_locked(task_ids)

        self._write_locked(commit, lambda held: held.task_ids.extend(task_ids))

    def _release_ended_dispatches_locked(self, task_ids: Sequence[str]) -> None:
        """Release each worker reserved for a dispatch that ended: its task moved off
        it, or it ended at a suspension.

        A worker reporting its status names the dispatch it concerns, and its IDLE
        clears the reservation itself; one that names none is fenced while reserved, so
        the end of the dispatch is what frees it. The release is a no-op once the
        worker's own IDLE cleared it, and never frees a later reservation.
        """
        for task_id in dict.fromkeys(task_ids):
            held = self._held_dispatches.get(task_id)
            record = self._tasks.get(task_id)
            if held is None or (
                record is not None
                and _membership(record) == TaskStatus.DISPATCHED
                and record.dispatch_id == held[1]
                and not self._dispatch_ended_at_suspension_locked(record)
            ):
                continue
            del self._held_dispatches[task_id]
            self._ended_dispatches.append(held)

    def _release_ended_workers(self) -> None:
        """Release, off the lock, each worker reserved for a dispatch that ended."""
        with self._lock:
            ended, self._ended_dispatches = self._ended_dispatches, []
        if failed := self._release_workers(ended):
            with self._lock:
                self._ended_dispatches[:0] = failed

    def _release_workers(
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
        self._release_terminated_work(
            _Termination(
                [],
                [],
                revokes=[
                    RevokeMessage(
                        task_id=r.task_id,
                        worker_id=r.worker_id,
                        dispatch_id=r.dispatch_id,
                    )
                    for r in ended
                ],
            )
        )
        self._release_workers([(r.worker_id, r.dispatch_id) for r in ended])

    def _write_locked(
        self, write: Callable[[], None], hold: Callable[[_HeldWrites], None]
    ) -> None:
        """Make one durable write, or hold it back within a worker's report once this
        or an earlier write of the report fails."""
        held = self._report_writes.held
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

    def _writes_held(self) -> bool:
        """Whether a durable write of the report being handled failed."""
        held = self._report_writes.held
        return held is not None and held.error is not None

    def _recommit_locked(self, held: _HeldWrites) -> None:
        """Make the durable writes a report held back: its tasks, then its spawned
        children, then each touched workflow's ledger and credential reclaim.

        The children are committed again after they are added, so each one's
        membership follows its current status.
        """
        self._commit_locked(*held.task_ids)
        for workflow_id, child_task_ids, retire in held.children:
            if (engine := self._engines.get(workflow_id)) is not None:
                self._commit_new_children_locked(
                    workflow_id, engine, child_task_ids, retire
                )
        self._commit_locked(*chain.from_iterable(ids for _, ids, _ in held.children))
        workflow_ids = [
            record.workflow_id
            for task_id in held.task_ids
            if (record := self._tasks.get(task_id)) is not None
        ]
        workflow_ids += [workflow_id for workflow_id, _, _ in held.children]
        for workflow_id in dict.fromkeys(workflow_ids + held.workflow_ids):
            self._save_ledger_locked(workflow_id)
            self._reclaim_vault_if_settled_locked(workflow_id)

    def _notify_terminal_transition(self, workflow_id: str) -> None:
        """Tell the completion finalizer a workflow may have reached its end.

        Every terminal commit funnels through `_commit_locked`, whether a worker
        reported it or the control plane settled it alone, so one notification covers
        both.
        It carries a workflow id and nothing else: the finalizer decides whether the
        workflow is complete, and does so off this thread.
        """
        if self._on_workflow_settled is None:
            return
        try:
            self._on_workflow_settled(workflow_id)
        except Exception as exc:
            self._logger.debug(
                "Failed to notify workflow completion for %s: %s", workflow_id, exc
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
            self._commit_locked(*terminal_ids)
        else:
            self._workflow_registry.commit_transition(
                workflow_id, sched=self._sched_locked(workflow_id)
            )

    def _reclaim_vault_if_settled_locked(self, workflow_id: str) -> None:
        """Purge a workflow's vaulted credentials once its last task has settled.

        Called after an event's advance materializes any new children, so a producer
        that fans out is not reclaimed while its children are still pending.
        """
        if self._writes_held() or self._workflow_settlement_locked(workflow_id).settled:
            self._write_locked(
                lambda: self._credential_vault.purge(workflow_id),
                lambda held: held.workflow_ids.append(workflow_id),
            )

    def _workflow_settlement_locked(self, workflow_id: str) -> WorkflowSettlement:
        # A retired task -- a sealed spawn's child template, replaced by the children it
        # instantiated -- no longer holds the workflow open, and its record stays
        # PENDING forever because it is never dispatched. Counting it would leave every
        # workflow with a spawn region permanently unsettled.
        retired = self._retired_region_templates.get(workflow_id) or set()
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
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Fail the pending tasks of every epoch after ``failed_epoch``.

        Returns the failed tasks with their reason, and the merged children of any of
        them returned to the queue.
        """
        epoch_tasks = self._workflow_epoch_tasks.get(workflow_id)
        if not epoch_tasks:
            return [], []
        frontier = self._workflow_epoch_frontier[workflow_id]

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
                self._pending_deps.pop(task_id, None)
                self._remove_from_ready_locked(task_id)
                self._merge_bucket_remove(task_id)
                self._merge_key_by_task.pop(task_id, None)
                returned += self._return_merged_children_locked(
                    self._merge_children_map.pop(task_id, []), unmerge=True
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
                task_id = self._pop_ready_locked()
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
        try:
            with self._cv:
                record = self._tasks.get(task_id)
                engine = self._engines.get(record.workflow_id) if record else None
                if record is None or engine is None:
                    return False
                advance = engine.route_boundary_event(task_id, event)
                self._synthesize_ready_children_locked(
                    record.workflow_id, engine, advance
                )
                changed = self._apply_advance_locked(record.workflow_id, advance)
                self._save_ledger_locked(record.workflow_id)
                if changed:
                    self._cv.notify_all()
                return changed
        finally:
            self._release_pending_terminations()

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
        if record.status == TaskStatus.CANCELLING:
            self._settle_cancelled_locked(record, time.time())
            self._reap_captures_locked(worker_id, task_id, _captured_calls(hr, group))
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
            self._save_ledger_locked(record.workflow_id)
            self._reap_captures_locked(worker_id, task_id, _captured_calls(hr, group))
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
        denied_capture: tuple[str | None, str] | None = None
        if corr is not None and env is not None and env.denial is not None:
            if request.request_digest is not None:
                denied_capture = (record.assigned_worker, corr)
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
        if denied_capture is not None:
            worker_id, call = denied_capture
            self._reap_captured_request_locked(
                worker_id, task_id, call, request.interface
            )
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
            self._persist_locked(task_id)
        self._save_ledger_locked(record.workflow_id)
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
        try:
            return self._settle_episode_invocation(
                task_id, call_correlation, value, error=error, ref=ref
            )
        finally:
            self._release_pending_terminations()

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
                self._save_ledger_locked(record.workflow_id)
                # A fenced failure terminal releases the resident credit just as a
                # completion does; nothing else may release an accepted credit.
                self._release_resident_credit(invocation_id, failed=True)
                self._reap_mediated_op(captured_on, task_id, call_correlation)
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
            self._save_ledger_locked(record.workflow_id)
            self._release_resident_credit(invocation_id, failed=False)
            self._reap_mediated_op(captured_on, task_id, call_correlation)
            self._cv.notify_all()
            return True

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
        try:
            with self._cv:
                record = self._tasks.get(task_id)
                engine = self._engines.get(record.workflow_id) if record else None
                if record is None or engine is None:
                    return False
                if record.status in TERMINAL_TASK_STATUSES:
                    return False
                envelope = engine.pending_tool_dispatch(task_id, call_correlation)
                if envelope is None:
                    return False
                self._dispatch_boundary(envelope)
            return True
        finally:
            self._release_pending_terminations()

    def redeliver_to_worker(self, worker_id: str) -> None:
        """Re-relay what control holds for a worker whose task stream attached.

        A frame relayed while the worker had no stream attached may be lost. Each
        pending mediated operation the worker originated is re-minted, and the worker
        drops one it already runs; each task being cancelled there is interrupted
        again, keyed to its dispatch. Task dispatches are not re-relayed: a lost one
        resolves as lost.
        """
        with self._cv:
            pending = {
                (task_id, call)
                for task_id, call, op_worker in self._pending_ops.values()
                if op_worker == worker_id
            }
            interrupts = [
                interrupt
                for record in self._tasks.values()
                if record.assigned_worker == worker_id
                and record.status == TaskStatus.CANCELLING
                and (
                    interrupt := self._interrupt_for(
                        record, record.error or "cancelled"
                    )
                )
            ]
            if interrupts:
                self._pending_terminations.append(_Termination(interrupts, []))
        self._release_pending_terminations()
        for task_id, call in sorted(pending):
            self.redispatch_episode_invocation(task_id, call)

    def _dispatch_boundary(self, env: ToolInvocationEnvelope) -> None:
        """Route a recorded mediated boundary to its handler by exact (kind, interface).

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
                if self._is_resident_env(env):
                    self._dispatch_resident_op(env)
                else:
                    self._dispatch_worker_originated_op(env)
            elif self._model_settler is not None:
                self._model_settler(env)
            return
        if (
            env.kind is BoundaryEventKind.INVOCATION
            and env.interface == SEARCH_INTERFACE
        ):
            if env.request_digest is not None:
                self._dispatch_worker_originated_op(env)
            elif self._tool_broker is not None:
                self._tool_broker(env)
            return
        outcome = ToolOutcome(
            status=ToolOutcomeStatus.UNAVAILABLE,
            value=f"no fabric handler for interface {env.interface!r}",
        )
        self._settle_episode_invocation(
            env.task_id, env.call_correlation, outcome.model_dump_json()
        )

    def _is_resident_env(self, env: ToolInvocationEnvelope) -> bool:
        """Whether the task consumes a resident dependency, without reading its plan."""
        with self._lock:
            record = self._tasks.get(env.task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return False
            return engine.service_dependency(env.task_id) is not None

    def _dispatch_resident_op(self, env: ToolInvocationEnvelope) -> None:
        """Originate a worker-captured resident boundary through resident admission."""
        if self._resident_originate is None:
            error = "resident-capacity control is not enabled"
        elif not self._resident_originate(env):
            error = "resident-capacity control is not running"
        else:
            return
        with self._lock:
            worker_id = self._assigned_worker_locked(env.task_id)
            self._settle_episode_invocation(
                env.task_id, env.call_correlation, error=error
            )
            self._relay_resident_reap(worker_id, env.task_id, env.call_correlation)

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

    def _op_permit_budget(self, interface: str) -> tuple[int, float, int]:
        """The (max_results, timeout, result_char_cap) budget a permit runs within."""
        if interface == MODEL_INTERFACE:
            return 1, self._model_egress_timeout_sec, _MODEL_PERMIT_RESULT_CHAR_CAP
        cfg = self._web_search
        return cfg.max_results, cfg.timeout_sec, cfg.result_char_cap

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

    def _assign_content_scope(
        self, permit: MediatedOperationPermit, agent: TaskRecord
    ) -> str:
        """Assign the scope this permit's outcome materializes in, and record it.

        The scope is the task's owner, so a result materializes in the tenant's
        namespace rather than the egressing worker's. Recording it against the permit's
        idempotency key is what lets the finalization this outcome later reports be
        checked against the scope control assigned, rather than one the reporting
        worker names for itself.
        """
        if self._content_scope_authority is not None and permit.idempotency_key:
            self._content_scope_authority(permit.idempotency_key, agent.org_id)
        return agent.org_id

    def _stamped_permit_payload(
        self, permit: MediatedOperationPermit, agent: TaskRecord
    ) -> dict[str, Any]:
        """The permit's wire payload, carrying what only the dispatching record knows.

        The trace stamp is post-mint: the boundary span id derives from the permit's own
        ``invocation_id``, which does not exist as an object until minting returns.
        """
        stamp: dict[str, Any] = {
            "content_scope": self._assign_content_scope(permit, agent)
        }
        if self._control.enabled:
            stamp["traceparent"] = format_traceparent(
                workflow_to_trace_id_int(agent.workflow_id),
                derived_span_id(SpanIdKind.INVOCATION, permit.invocation_id),
            )
        return permit.model_copy(update=stamp).model_dump(mode="json")

    def _dispatch_worker_originated_op(self, env: ToolInvocationEnvelope) -> None:
        """Mint a permit and relay a boundary's egress operation to its origin worker.

        The permit is audience-bound to the agent's own worker and relayed there as an
        ordinary control message on the authenticated attachment, never the raw request
        (the origin worker holds it privately) and never a dispatched task. A restart
        re-mints and re-relays it from the still-pending boundary. If the origin worker
        is gone the request cannot be recovered, so the boundary fails clean.
        """
        agent = self._tasks.get(env.task_id)
        engine = self._engines.get(agent.workflow_id) if agent else None
        if agent is None or engine is None:
            return
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
            return
        op_credential = self._resolve_op_credential(agent, env.interface)
        if isinstance(op_credential, _MissingCredential):
            self._settle_episode_invocation(
                env.task_id, env.call_correlation, error=op_credential.reason
            )
            return
        max_results, timeout_sec, result_char_cap = self._op_permit_budget(
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
            return
        # A re-drive re-mints under a fresh permit id; keep at most one pending op per
        # occurrence.
        occurrence = (env.task_id, env.call_correlation)
        for stale_id, (task_id, call, _) in list(self._pending_ops.items()):
            if (task_id, call) == occurrence:
                del self._pending_ops[stale_id]
        self._pending_ops[permit.permit_id] = (
            env.task_id,
            env.call_correlation,
            worker_id,
        )
        self._worker_registry.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id,
                frame_kind="permit",
                payload=self._stamped_permit_payload(permit, agent),
            ),
        )

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
        with self._cv:
            agent = self._tasks.get(proposal.agent_task_id)
            engine = self._engines.get(agent.workflow_id) if agent else None
            if agent is None or engine is None:
                return
            if proposal.dispatch_id is None or not self._holds_dispatch_locked(
                agent, proposer_id, proposal.dispatch_id
            ):
                self._deny_model_turn(proposal, proposer_id, "model turn not held")
                return
            worker_id = agent.assigned_worker
            worker = self._worker_registry.get_worker(worker_id) if worker_id else None
            if worker_id is None or worker is None:
                # A gone origin worker cannot receive a relay: the held turn fails on
                # its own permit deadline.
                return
            binding = self.resolve_model_binding(proposal.agent_task_id)
            permit = None
            reason = "model turn egress denied"
            if binding is not None and binding.mode is ModelBindingMode.OPENAI:
                op_credential = self._model_credential(agent, binding)
                if isinstance(op_credential, _MissingCredential):
                    reason = op_credential.reason
                else:
                    _, timeout_sec, result_char_cap = self._op_permit_budget(
                        MODEL_INTERFACE
                    )
                    deadline = time.time() + timeout_sec + _OP_PERMIT_SLACK_SEC
                    permit = engine.authorize_model_turn(
                        proposal.agent_task_id,
                        proposal.call_correlation,
                        proposal.request_digest,
                        target_id=worker_id,
                        target_generation=worker.incarnation,
                        timeout_sec=timeout_sec,
                        result_char_cap=result_char_cap,
                        deadline_epoch=deadline,
                        credential=op_credential.credential,
                        deployment_credential=op_credential.deployment_credential,
                    )
            if permit is None:
                self._deny_model_turn(proposal, worker_id, reason)
                return
            self._worker_registry.publish_mediated_op(
                worker,
                MediatedOpMessage(
                    worker_id=worker_id,
                    frame_kind="permit",
                    payload=self._stamped_permit_payload(permit, agent),
                ),
            )

    def _deny_model_turn(
        self, proposal: AgentModelTurnProposal, worker_id: str, reason: str
    ) -> None:
        """Relay a deny frame so a held turn fails fast rather than waiting out its
        deadline."""
        if (worker := self._worker_registry.get_worker(worker_id)) is None:
            return
        self._worker_registry.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id,
                frame_kind="deny",
                payload={
                    "agent_task_id": proposal.agent_task_id,
                    "call_correlation": proposal.call_correlation,
                    "reason": reason,
                },
            ),
        )

    def settle_mediated_operation(self, outcome: MediatedOperationOutcome) -> None:
        """Settle an agent boundary from its origin worker's fenced outcome report.

        The report carries exactly one of a reference-backed outcome, a bounded inline
        outcome, or a worker-fault error; each settles the originating boundary. The
        terminal fact commits before the worker-private request custody is reaped, so a
        lost report leaves the boundary pending for a same-idempotency-key re-drive. A
        duplicate or late report is absorbing at the boundary.
        """
        try:
            self._settle_mediated_operation(outcome)
        finally:
            self._release_pending_terminations()

    def _settle_mediated_operation(self, outcome: MediatedOperationOutcome) -> None:
        with self._cv:
            pending = self._pending_ops.pop(outcome.permit_id, None)
            worker_id = (
                pending[2]
                if pending
                else self._assigned_worker_locked(outcome.agent_task_id)
            )
            agent_task_id = outcome.agent_task_id
            call = outcome.call_correlation
            record = self._tasks.get(agent_task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None or not engine.boundary_settleable(agent_task_id, call):
                # No settle follows a duplicate or late report, so the request its
                # worker holds is reaped here.
                self._reap_mediated_op(worker_id, agent_task_id, call)
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

    def _reap_mediated_op(
        self, worker_id: str | None, agent_task_id: str, call: str
    ) -> None:
        """Relay a best-effort reap so the origin worker drops the request custody."""
        if not worker_id:
            return
        worker = self._worker_registry.get_worker(worker_id)
        if worker is None:
            return
        self._worker_registry.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id,
                frame_kind="reap",
                payload={"agent_task_id": agent_task_id, "call_correlation": call},
            ),
        )

    def _reap_captured_request_locked(
        self, worker_id: str | None, task_id: str, call: str, interface: str | None
    ) -> None:
        """Relay a reap for a request the worker captured for a boundary that will
        never run, from whichever store holds it."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if (
            interface == MODEL_INTERFACE
            and engine is not None
            and engine.service_dependency(task_id) is not None
        ):
            self._relay_resident_reap(worker_id, task_id, call)
        else:
            self._reap_mediated_op(worker_id, task_id, call)

    def _relay_resident_reap(
        self, worker_id: str | None, task_id: str, call: str
    ) -> None:
        """Relay a best-effort reap so the worker drops a captured resident request."""
        if not worker_id:
            return
        worker = self._worker_registry.get_worker(worker_id)
        if worker is None:
            return
        self._worker_registry.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id,
                frame_kind="resident_reap",
                payload={"task_id": task_id, "call_correlation": call},
            ),
        )

    def _reap_ops_for_agents_locked(self, agent_task_ids: Sequence[str]) -> None:
        """Reap pending tool operations whose agent boundary just failed clean."""
        for worker_id, agent_task_id, call in self._take_ops_for_agents_locked(
            agent_task_ids
        ):
            self._reap_mediated_op(worker_id, agent_task_id, call)

    def _take_ops_for_agents_locked(
        self, agent_task_ids: Sequence[str]
    ) -> list[tuple[str, str, str]]:
        """Drop the agents' pending tool operations, returning each one's worker, agent
        and call for reaping."""
        agents = set(agent_task_ids)
        taken: list[tuple[str, str, str]] = []
        for permit_id, (agent_task_id, call, worker_id) in list(
            self._pending_ops.items()
        ):
            if agent_task_id in agents:
                del self._pending_ops[permit_id]
                taken.append((worker_id, agent_task_id, call))
        return taken

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

    def set_resident_terminal_hook(self, hook: Callable[[str, bool], None]) -> None:
        """Install the consumer that releases a resident admission credit on DS
        terminal.

        The hook receives the settled boundary's ``invocation_id`` and whether the
        outcome was a failure, so the Admission controller advances the linked claim to
        terminal on any fenced outcome — the sole normal credit release.
        """
        self._resident_terminal_hook = hook

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

    def _release_resident_credit(
        self, invocation_id: str | None, *, failed: bool
    ) -> None:
        if invocation_id is not None and self._resident_terminal_hook is not None:
            self._resident_terminal_hook(invocation_id, failed)

    def originate_facade_turn_group(self, task_id: str, group: FacadeTurnGroup) -> None:
        """Record a turn-scoped facade group a worker captured on a held model turn.

        The worker facade captures a model turn's native facade calls and cleans the
        turn; the whole ordered membership and its single continuation are persisted
        on the task record before the turn resumes, so the episode's next completion
        routes the group rather than settling the episode DONE, and a restart-replayed
        completion still routes it. At most one group is open per episode; a second one
        while one holds the gate is refused by the busy fence, not stored here.
        """
        with self._lock:
            self._pending_facade_groups[task_id] = group
            if (record := self._tasks.get(task_id)) is not None:
                record.pending_facade_group = group
                self._persist_locked(task_id)

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
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None:
                return None
            op = engine.agent_operator(task_id)
            return op.model_binding if op is not None else None

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
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return None
            return engine.resident_admission_binding(record.workflow_id, task_id)

    def boundary_settleable(self, task_id: str, call_correlation: str) -> bool:
        """Whether a mediated boundary still awaits its outcome."""
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return False
            return engine.boundary_settleable(task_id, call_correlation)

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
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            return engine.private_state_owner(task_id) if engine else None

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
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None:
                return None
            op = engine.agent_operator(task_id)
            harness = op.harness_binding if op is not None else None
            if harness is None:
                return None
            model = op.model_binding if op is not None else None
            model_binding = (
                EpisodeModelBinding(mode=model.mode, url=model.url, model=model.model)
                if model is not None
                else None
            )
            granted = engine.grant_private_state(
                task_id, holder.worker_id, holder.incarnation
            )
            capsule_blob, outcomes = engine.episode_context(task_id)
            invoke_face = engine.effective_invoke_face(task_id)
            sandbox = _sandbox_capability(
                op, granted[1] if granted else None, invoke_face
            )
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
                model_binding=model_binding,
                facade_descriptors=_effective_facades(op, invoke_face, sandbox),
                private_state=granted[0] if granted else None,
                private_state_attachment=granted[1] if granted else None,
                sandbox=sandbox,
            )

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
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None:
                return None
            dependency = self._resident_served_dependency_locked(engine, task_id)
            if dependency is None:
                return None
            _capsule, outcomes = engine.episode_context(task_id)
            return ServiceLeafEpisodeDispatch(
                interface=dependency.interface.value,
                delivered_outcomes=outcomes,
            )

    def serves_from_replica(self, task_id: str) -> bool:
        """Whether a task's dispatch carries its invocation to a resident replica, a
        menu-resolved or pinned resident leaf, rather than loading a model locally."""
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            return (
                engine is not None
                and self._resident_served_dependency_locked(engine, task_id) is not None
            )

    def _resident_served_dependency_locked(
        self, engine: OrchestrationEngine, task_id: str
    ) -> ServiceDependency | None:
        """The resident dependency a leaf's dispatch is served from, or None."""
        dependency = engine.service_dependency(task_id)
        if dependency is None or engine.agent_operator(task_id) is not None:
            return None
        if engine.embodiment_menu(task_id) is not None:
            resolved = self._resolved_embodiment_locked(engine, task_id)
            if (
                resolved is None
                or resolved.kind is not InferenceEmbodimentKind.RESIDENT_SERVED
            ):
                return None
        return dependency

    def _resolved_embodiment_locked(
        self, engine: OrchestrationEngine, task_id: str
    ) -> ResolvedEmbodiment | None:
        menu = engine.embodiment_menu(task_id)
        if menu is None or (selection := engine.embodiment_selection(task_id)) is None:
            return None
        candidate = menu.candidate(selection.alternative_id)
        if candidate is None:
            return None
        return ResolvedEmbodiment(
            alternative_id=candidate.alternative_id, kind=candidate.kind
        )

    def embodiment_pinned(self, task_id: str) -> bool:
        """Whether a task's resolved embodiment is committed to the run carrying it."""
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            return engine.embodiment_pinned(task_id) if engine else False

    def embodiment_menu(self, task_id: str) -> InferenceEmbodimentMenu | None:
        """The embodiments a ready task's plan node offers, if it offers a menu."""
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            return engine.embodiment_menu(task_id) if engine else None

    def record_embodiment_selection(
        self, task_id: str, alternative_id: str, selector: str, evidence: str
    ) -> str | None:
        """Durably bind a task to one embodiment, returning the bound alternative.

        Recorded before the task's worker message is published, so the choice survives a
        loss between publication and the attempt bookkeeping that follows it. A pinned
        selection is kept and returned unchanged.
        """
        try:
            with self._lock:
                record = self._tasks.get(task_id)
                engine = self._engines.get(record.workflow_id) if record else None
                if engine is None or record is None:
                    return None
                selection = engine.record_embodiment_selection(
                    task_id, alternative_id, selector, evidence
                )
                if selection is None:
                    return None
                self._save_ledger_locked(record.workflow_id)
                return selection.alternative_id
        finally:
            self._release_ended_workers()

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
                self._register_child_locked(child_task_id, template)
                new_children.append(child_task_id)
        self._commit_new_children_locked(workflow_id, engine, new_children)

    def _register_child_locked(self, child_task_id: str, template: TaskRecord) -> None:
        """Install a self-contained task record for one materialized child."""
        self._tasks[child_task_id] = self._synthesize_child_record(
            template, child_task_id
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
            self._persist_declared_failures_locked(engine, child_task_ids)
            self._write_locked(
                lambda: self._workflow_registry.commit_dynamic_tasks(
                    workflow_id,
                    self._records_locked(*child_task_ids),
                    engine.to_snapshot(),
                    retire=retire,
                ),
                lambda held: held.children.append(
                    (workflow_id, list(child_task_ids), list(retire))
                ),
            )
            if retire:
                self._retired_region_templates.setdefault(workflow_id, set()).update(
                    retire
                )
                # A retire drains the remaining set as a terminal does, and can drain
                # its last entry: a spawn that seals with no children leaves the
                # workflow complete with no task terminal behind it. The two drains --
                # a terminal commit and a retire -- each notify, and they are the only
                # two, so no completion escapes the finalizer.
                if not self._writes_held():
                    self._notify_terminal_transition(workflow_id)

    def _retire_sealed_region_templates_locked(
        self, workflow_id: str, engine: OrchestrationEngine
    ) -> None:
        """Retire an agent-region child template once its spawn region has sealed.

        A dynamic spawn region's child body is a template, never dispatched as a task;
        once the region seals it no longer holds the workflow open, so it is dropped
        from the remaining set (idempotently, tracked per workflow) with the ledger.
        """
        already = self._retired_region_templates.setdefault(workflow_id, set())
        if pending := engine.sealed_region_child_templates() - already:
            self._commit_new_children_locked(
                workflow_id, engine, [], retire=sorted(pending)
            )

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
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if record is None or engine is None:
                return None
            spec = spec or record.task.spec
            if not isinstance(spec, (InferenceSpecStrict, InferenceSpecTemplate)):
                return None
            element = self._input_element_locked(task_id)
            try:
                contract = (
                    element_contract(
                        spec, element.producer_task_id, element.ref.element
                    )
                    if element is not None
                    else canonical_contract(spec)
                )
            except CanonicalProjectionError:
                self._logger.warning(
                    "[fabric] a contract leaf's request is no longer projectable: %s",
                    task_id,
                )
                return None
            if engine.embodiment_menu(task_id) is None:
                if engine.service_dependency(task_id) is None:
                    return None
                if (
                    contract.source.kind is InferenceSourceKind.LITERAL
                    and len(contract.source.items) <= 1
                ):
                    return None
            return contract

    def record_input_resolution(
        self,
        task_id: str,
        worker_id: str | None,
        binding_payload: Any,
        dispatch_id: str | None = None,
    ) -> None:
        """Record how a task's inputs resolved on its origin worker.

        The worker reports this before either embodiment reaches a model, so the
        resolution is durable ahead of a local generation or a resident service issue,
        and the admission that follows is sized from the cardinality that materialized.
        """
        try:
            binding = InputResolutionBinding.model_validate(binding_payload)
        except ValidationError:
            self._logger.warning(
                "[fabric] a task reported an unreadable input resolution: %s", task_id
            )
            return
        try:
            with self._lock:
                record = self._tasks.get(task_id)
                if record is None or not self._accepts_event_locked(
                    record, worker_id, dispatch_id
                ):
                    return
                if (engine := self._engines.get(record.workflow_id)) is not None:
                    engine.record_input_resolution(task_id, binding)
        finally:
            self._release_ended_workers()

    def input_resolution_binding(self, task_id: str) -> InputResolutionBinding | None:
        """The binding a task's recorded resolution carries, if one was recorded."""
        with self._lock:
            resolution = self._input_resolution_locked(task_id)
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
            if (
                record is None
                or record.status in TERMINAL_TASK_STATUSES
                or record.assigned_worker != worker_id
                or dispatch_id is None
                or not self._holds_dispatch_locked(record, worker_id, dispatch_id)
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
        episode, the settled result of an upstream task it depends on, or the producer
        result one of its accepted inputs or its fan-out element is frozen to. Naming an
        object it merely knows of authorizes nothing.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            return (
                record is not None
                and record.status not in TERMINAL_TASK_STATUSES
                and self._holds_dispatch_locked(record, worker_id, None)
                and self._consumes_locked(record, reference)
            )

    def _consumes_locked(self, record: TaskRecord, reference: ContentReference) -> bool:
        """Whether a task is bound to exactly this object as one of its inputs."""
        if reference.authorization_scope != record.org_id:
            return False
        task_id = record.task_id
        resolution = self._input_resolution_locked(task_id)
        if resolution is not None and resolution.reference == reference:
            return True
        if self._upstream_result_is_locked(record, reference):
            return True
        engine = self._engines.get(record.workflow_id)
        if engine is None:
            return False
        if self._frozen_input_is_locked(engine, task_id, reference):
            return True
        _, outcomes = engine.episode_context(task_id)
        return any(
            outcome.outcome_ref is not None and outcome.outcome_ref.content == reference
            for outcome in outcomes
        )

    def upstream_task_ids(self, task_id: str) -> set[str]:
        """Every task of its workflow a task depends on, directly or transitively."""
        with self._lock:
            return self._upstream_task_ids_locked(task_id)

    def _upstream_task_ids_locked(self, task_id: str) -> set[str]:
        # A record stored with a dependency outside its workflow never reaches it: a
        # stage name resolves only within the workflow that declared it.
        record = self._tasks.get(task_id)
        workflow_id = record.workflow_id if record else None
        pending = list(self._original_deps.get(task_id, ()))
        visited: set[str] = set()
        while pending:
            dep_id = pending.pop()
            upstream = self._tasks.get(dep_id)
            if dep_id in visited or upstream is None:
                continue
            if upstream.workflow_id != workflow_id:
                continue
            visited.add(dep_id)
            pending.extend(self._original_deps.get(dep_id, ()))
        return visited

    def _upstream_result_is_locked(
        self, record: TaskRecord, reference: ContentReference
    ) -> bool:
        """Whether a settled upstream of the task, or of one merged into its dispatch,
        is bound to exactly this result."""
        for member_id in (record.task_id, *(record.merged_children or [])):
            member = self._tasks.get(member_id)
            if member is None:
                continue
            for dep_id in self._upstream_task_ids_locked(member_id):
                upstream = self._tasks.get(dep_id)
                if (
                    upstream is None
                    or upstream.workflow_id != member.workflow_id
                    or upstream.status != TaskStatus.DONE
                ):
                    continue
                binding = self._result_binding_locked(dep_id)
                if binding is not None and binding.reference == reference:
                    return True
        return False

    def _frozen_input_is_locked(
        self, engine: OrchestrationEngine, task_id: str, reference: ContentReference
    ) -> bool:
        """Whether an accepted input of the task, or its fan-out element, is frozen
        to exactly this producer result."""
        child_input = engine.child_input(task_id)
        if child_input is not None and child_input.content == reference:
            return True
        return any(
            (source := self._member_source_locked(member.value_ref)) is not None
            and source.reference == reference
            for accepted in engine.accepted_inputs_for_task(task_id)
            for member in accepted.members
        )

    def input_element(self, task_id: str) -> ResultElementRef | None:
        """The producer element a leaf fan-out child runs on, for its worker to hydrate.

        An agent child receives its element through its accepted input.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            if engine is None or engine.agent_operator(task_id) is not None:
                return None
            element = self._input_element_locked(task_id)
        return element.ref if element is not None else None

    def _input_element_locked(self, task_id: str) -> _InputElement | None:
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if engine is None:
            return None
        child_input = engine.child_input(task_id)
        if (
            child_input is None
            or child_input.content is None
            or child_input.collection_key is None
            or child_input.legacy_task_id is None
        ):
            return None
        return _InputElement(
            child_input.legacy_task_id,
            ResultElementRef(
                reference=child_input.content, element=int(child_input.collection_key)
            ),
        )

    def recorded_input_reference(self, task_id: str) -> ContentReference | None:
        """Where a task's prepared request is, for the run that hydrates it."""
        with self._lock:
            resolution = self._input_resolution_locked(task_id)
        return resolution.reference if resolution is not None else None

    def _input_resolution_locked(self, task_id: str) -> InputResolution | None:
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        return engine.input_resolution(task_id) if engine else None

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
            resolution = self._input_resolution_locked(task_id)
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
        self._save_ledger_locked(record.workflow_id)

    def _settle_preparation_failure_locked(
        self, record: TaskRecord, engine: OrchestrationEngine, reason: str
    ) -> None:
        if self._apply_advance_locked(
            record.workflow_id,
            engine.on_failed(record.task_id, reason, retryable=False),
        ):
            self._cv.notify_all()
        self._save_ledger_locked(record.workflow_id)

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
                open=not self._workflow_settlement_locked(workflow_id).settled,
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
                open=not self._workflow_settlement_locked(workflow_id).settled,
            )

    def read_output(self, member: OutputMember) -> ResultEnvelope:
        """The stored result a published member settled with, read off the lock.

        Raises ``ResultUnavailable`` while the store cannot be reached and
        ``ResultUnreadable`` for bound content that is missing or corrupt.
        """
        value_ref = member.publication.value_ref if member.publication else None
        if value_ref is None or value_ref.content is None:
            raise ResultUnreadable(f"output {member.name} has no bound result")
        return self._results.read_reference(value_ref.content)

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
            return self._result_binding_locked(task_id)

    def read_result(self, task_id: str) -> ResultEnvelope | None:
        """A task's result envelope, None when it has none; raises when unreadable."""
        binding = self.result_binding(task_id)
        return self._results.read(binding) if binding is not None else None

    def read_result_bytes(self, task_id: str) -> bytes | None:
        """A task's stored result envelope bytes, None when it has none."""
        binding = self.result_binding(task_id)
        return self._results.read_bytes(binding) if binding is not None else None

    def _result_binding_locked(self, task_id: str) -> ResultBinding | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        if (engine := self._engines.get(record.workflow_id)) is None:
            if record.result_reference is None and record.result_skip is None:
                return None
            return ResultBinding(
                task_id=task_id,
                reference=record.result_reference,
                skip=record.result_skip,
                settled_at=_settled_at(record),
            )
        settled = engine.legacy_task_value(task_id)
        if settled is None:
            return None
        outcome, value_ref = settled
        if outcome is PublicationOutcome.EXPLICIT_EMPTY:
            if record.result_skip is None:
                return None
            return ResultBinding(
                task_id=task_id, skip=record.result_skip, settled_at=_settled_at(record)
            )
        if (
            outcome is not PublicationOutcome.SUCCESS
            or value_ref is None
            or value_ref.content is None
        ):
            return None
        return ResultBinding(task_id=task_id, reference=value_ref.content)

    def _settled_unbound_locked(self, task_id: str) -> bool:
        """Whether a task settled successfully with no result bound to read."""
        record = self._tasks.get(task_id)
        if record is None or record.status != TaskStatus.DONE:
            return False
        if record.result_skip is not None:
            return False
        engine = self._engines.get(record.workflow_id)
        if engine is None:
            return record.result_reference is None
        settled = engine.legacy_task_value(task_id)
        return settled is not None and settled[0] is PublicationOutcome.SUCCESS

    def _bind_result_locked(
        self,
        record: TaskRecord,
        reference: ContentReference | None,
        skip: dict[str, Any] | None,
    ) -> None:
        """Bind a settling task's result once; a later success never re-points it."""
        if record.result_reference is not None or record.result_skip is not None:
            return
        if skip is not None:
            record.result_skip = skip
            return
        record.result_reference = self._accepted_reference(record, reference)

    def _accepted_reference(
        self, record: TaskRecord, reference: ContentReference | None
    ) -> ContentReference | None:
        """A reported result reference, when it lies in the scope control gave the task.

        Control reads a bound result under its own store access, so a reference naming
        another scope's object is refused here rather than served to this task's owner.
        """
        if reference is None or reference.authorization_scope == record.org_id:
            return reference
        self._logger.error(
            "Task %s reported a result in scope %s outside its own %s; not binding it",
            record.task_id,
            reference.authorization_scope,
            record.org_id,
        )
        return None

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
        try:
            with self._cv:
                return self._resolve_uncertain_locked(task_id)
        finally:
            self._release_pending_terminations()

    def _resolve_uncertain_locked(self, task_id: str) -> Advance:
        """Resolve an in-flight work item's uncertainty; a failure terminalizes the
        boundary invocations it held, whose credits release once the ledger is saved."""
        record = self._tasks.get(task_id)
        if record is None or (engine := self._engines.get(record.workflow_id)) is None:
            return Advance()
        advance = engine.on_uncertain(task_id)
        if advance.retry:
            self._release_dispatch_locked(record, [task_id], front=True)
        elif advance.failed:
            # A lost child's failure can release its scope's join, readying what
            # follows.
            if self._apply_advance_locked(record.workflow_id, advance):
                self._cv.notify_all()
            self._reap_ops_for_agents_locked(advance.failed)
            if invocation_ids := engine.terminalize_unsettled_invocations([task_id]):
                self._hold_termination_locked(
                    record.workflow_id,
                    _Termination([], [], resident_invocation_ids=invocation_ids),
                )
        self._save_ledger_locked(record.workflow_id)
        return advance

    def _save_ledger_locked(self, workflow_id: str) -> None:
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
            self._pending_terminations += self._undurable_terminations.pop(
                workflow_id, []
            )

        if engine is None and workflow_id not in self._undurable_terminations:
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
            self._fail_record_locked(record, reason)
            failed.append(task_id)
        if failed:
            self._commit_locked(*failed)

    def _hold_termination_locked(
        self, workflow_id: str, termination: _Termination
    ) -> None:
        """Hold what a termination releases until the workflow's next ledger save
        succeeds; ``_release_pending_terminations`` releases it after the lock."""
        self._undurable_terminations.setdefault(workflow_id, []).append(termination)

    def _apply_advance_locked(self, workflow_id: str, advance: Advance) -> bool:
        """Apply an engine advance: fail and persist what it failed, cancel what a
        residual policy cancelled, then record the inputs its agents accept, retire
        the region templates it sealed, and ready its work. Returns whether it changed
        any task.

        The failed and cancelled records persist before a retire writes the ledger, so
        the ledger never leads them.
        """
        # A ready/settle advance never carries a retry; the failure path drives those.
        assert not advance.retry, "retry is applied by the failure path"
        engine = self._engines.get(workflow_id)
        changed = bool(advance.failed)
        self._fail_v2_advance_locked(engine, advance)
        if engine is not None and advance.cancelled:
            changed |= self._cancel_residual_locked(
                workflow_id, engine, advance.cancelled
            )
        if engine is not None:
            staged = Advance()
            self._stage_agent_inputs_locked(workflow_id, engine, staged)
            changed |= bool(staged.failed)
            self._fail_v2_advance_locked(engine, staged)
            advance.extend(staged)
            self._retire_sealed_region_templates_locked(workflow_id, engine)
        for task_id in advance.ready:
            if self._enqueue_ready_locked(task_id):
                changed = True
        return changed

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
            self._commit_locked(*(task_id for task_id, _ in changed))
        return changed

    def _fan_out_children_locked(
        self,
        workflow_id: str,
        engine: OrchestrationEngine,
        producer_task_id: str,
        read: _FanoutRead | None,
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
        binding = self._result_binding_locked(producer_task_id)
        content = binding.reference if binding is not None else None
        if content != read.reference:
            # The read predates the binding the producer settled with: read again.
            self._redrive.drive_now(workflow_id)
            return advance
        # A child receives its element as a frozen reference into the producer result:
        # an agent child through the typed accepted-input channel of its declared entry
        # port, a leaf child as its child-init input, which its worker hydrates.
        child_is_agent = engine.agent_entry_port(child_template_id) is not None
        new_children: list[str] = []
        for index in range(read.count):
            value_ref = ValueRef(
                kind="legacy_task_result",
                legacy_task_id=producer_task_id,
                content=content,
                collection_key=str(index),
            )
            try:
                if child_is_agent:
                    child_task_id = engine.create_fanout_child(spawn_op, value_ref)
                    self._register_child_locked(child_task_id, template_record)
                    self._mint_fanout_facet_locked(
                        engine, child_task_id, producer_task_id, index, value_ref
                    )
                    advance.extend(engine.reconsider_admission(child_task_id))
                    new_children.append(child_task_id)
                    continue
                child_advance = engine.materialize_child(spawn_op, value_ref=value_ref)
            except RegionError:
                break  # a budget, seal, or denial stops further children
            for child_task_id in child_advance.ready:
                self._register_child_locked(child_task_id, template_record)
                new_children.append(child_task_id)
            advance.extend(child_advance)
        advance.extend(engine.seal_spawn(spawn_op))
        self._commit_new_children_locked(
            workflow_id,
            engine,
            new_children,
            retire=engine.template_closure(child_template_id),
        )
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
        try:
            self._redrive_workflow(workflow_id)
        finally:
            # A fan-out read here that finds its collection unreadable fails the
            # workflow.
            self._release_pending_terminations()

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
                for task_id, check in self._input_checks.items()
                if (record := self._tasks.get(task_id)) is not None
                and record.workflow_id == workflow_id
            }
        if not checks:
            return
        verdicts = {
            task_id: self._verify_inputs(check.references)
            for task_id, check in checks.items()
            if check.unreadable is None
        }
        failures: list[TaskEvent] = []
        with self._cv:
            for task_id, check in checks.items():
                if self._input_checks.get(task_id) is not check:
                    continue
                record = self._tasks.get(task_id)
                if check.unreadable is not None:
                    if record is not None and (
                        record.status == TaskStatus.PENDING
                        or self._verdict_unacknowledged_locked(task_id)
                    ):
                        failures.append(_input_verdict(task_id, check))
                    else:
                        del self._input_checks[task_id]
                    continue
                if record is None or record.status != TaskStatus.PENDING:
                    del self._input_checks[task_id]
                    continue
                verdict = verdicts[task_id]
                if isinstance(verdict, ResultUnreadable):
                    self._logger.warning(
                        "Task %s input is unreadable at control: %s", task_id, verdict
                    )
                    check = self._input_checks[task_id] = replace(
                        check, unreadable=str(verdict)
                    )
                    failures.append(_input_verdict(task_id, check))
                    continue
                if verdict is not None:
                    self._logger.warning(
                        "Task %s waits: control cannot read its inputs either: %s",
                        task_id,
                        verdict,
                    )
                    self._redrive.schedule(workflow_id)
                    continue
                self._logger.info(
                    "Task %s runs again: control reads the inputs its worker could not",
                    task_id,
                )
                if self._enqueue_ready_locked(task_id, front=False):
                    self._cv.notify_all()
                self._commit_locked(task_id)
                del self._input_checks[task_id]
            if failures:
                # A later drive drops each check whose verdict has committed, or
                # reports it again.
                self._redrive.recheck(workflow_id)
            else:
                self._redrive.reset_recheck(workflow_id)
        for event in failures:
            self._report_failure(event)

    def _verdict_unacknowledged_locked(self, task_id: str) -> bool:
        """Whether control's verdict on a task failed a durable write and waits to be
        handled again."""
        pending = self._unacknowledged.get(task_id)
        return pending is not None and pending.report == _INPUT_VERDICT_REPORT

    def _verify_inputs(
        self, references: tuple[ContentReference, ...]
    ) -> ResultUnreadable | Exception | None:
        """Why control cannot read these objects either, or None when it reads them."""
        pending: Exception | None = None
        for reference in references:
            try:
                self._results.verify(reference)
            except ResultUnreadable as exc:
                return exc
            except Exception as exc:
                # Unreachable, or an error that says nothing about the content itself.
                pending = exc
        return pending

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
        and every blocked agent whose bound inputs need reading has them read, all off
        the lock. The results apply under it: an agent's inputs record only while the
        snapshot they were read for holds, and are read again otherwise. A read
        that cannot reach the store schedules the next re-drive.
        """
        with self._lock:
            engine = self._engines.get(workflow_id)
            if engine is None:
                return
            producers = [
                (task_id, self._result_binding_locked(task_id))
                for task_id, record in self._tasks.items()
                if record.workflow_id == workflow_id
                and record.status == TaskStatus.DONE
                and (spawn_op := engine.spawn_successor(task_id)) is not None
                and engine.spawn_awaits_children(spawn_op)
            ]
            snapshots = [
                snapshot
                for task_id in engine.blocked_input_agents()
                if (snapshot := self._agent_input_snapshot_locked(engine, task_id))
                is not None
                and snapshot.references
            ]
        reads = {
            task_id: self._read_fanout(task_id, binding)
            for task_id, binding in producers
        }
        values = self._read_input_values(snapshots)
        with self._cv:
            if (engine := self._engines.get(workflow_id)) is None:
                return
            advance = Advance()
            for task_id, _binding in producers:
                advance.extend(
                    self._fan_out_children_locked(
                        workflow_id, engine, task_id, reads[task_id]
                    )
                )
            for snapshot in snapshots:
                if self._agent_input_snapshot_locked(engine, snapshot.task_id) != (
                    snapshot
                ):
                    self._redrive.drive_now(workflow_id)
                    continue
                self._settle_agent_inputs_locked(
                    workflow_id, engine, snapshot, values, advance
                )
            if self._apply_advance_locked(workflow_id, advance):
                self._cv.notify_all()
            self._save_ledger_locked(workflow_id)

    def _prefetch_fanout(
        self,
        task_id: str,
        reference: ContentReference | None,
        skip: dict[str, Any] | None,
    ) -> _FanoutRead | None:
        """Read a spawn producer's fan-out collection before its success takes the lock.

        The read goes to the shared store, so it runs, and retries, outside the runtime
        lock; only a success feeding a spawn that has yet to fan out pays for it. A
        skipped producer has no collection, so its spawn fans out to no children.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            engine = self._engines.get(record.workflow_id) if record else None
            spawn_op = engine.spawn_successor(task_id) if engine else None
            if (
                record is None
                or record.status in TERMINAL_TASK_STATUSES
                or record.status == TaskStatus.CANCELLING
                or engine is None
                or spawn_op is None
                or not engine.spawn_awaits_children(spawn_op)
            ):
                return None
            reference = self._accepted_reference(record, reference)
        return self._read_fanout(
            task_id, ResultBinding(task_id=task_id, reference=reference, skip=skip)
        )

    def _read_fanout(self, task_id: str, binding: ResultBinding | None) -> _FanoutRead:
        """Count a producer's collection off the lock, retrying a store that is away."""
        if binding is not None and binding.skip is not None:
            return _FanoutRead()
        if binding is None or binding.reference is None:
            return _FanoutRead(error=f"fan-out producer {task_id} has no bound result")
        for attempt in range(_FANOUT_READ_ATTEMPTS):
            try:
                envelope = self._results.read(binding)
            except ResultUnreadable as exc:
                return _FanoutRead(
                    error=f"fan-out producer {task_id} result is unreadable: {exc}"
                )
            except ResultUnavailable as exc:
                if attempt + 1 == _FANOUT_READ_ATTEMPTS:
                    return _FanoutRead(error=str(exc), unavailable=True)
                time.sleep(_FANOUT_READ_BACKOFF_SEC)
                continue
            return _FanoutRead(
                count=len(collection_elements(envelope)), reference=binding.reference
            )
        raise AssertionError("unreachable")

    def _mint_fanout_facet_locked(
        self,
        engine: OrchestrationEngine,
        child_task_id: str,
        producer_task_id: str,
        index: int,
        value_ref: ValueRef,
    ) -> None:
        """Record a fan-out child's typed entry-port input from its producer element."""
        wi = engine.work_item(child_task_id)
        if wi is None:
            return
        entry_port = engine.agent_entry_port(wi.operator_id)
        if entry_port is None:
            return
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

    def _stage_agent_inputs_locked(
        self, workflow_id: str, engine: OrchestrationEngine, advance: Advance
    ) -> None:
        """Record each edge-bound agent's accepted inputs that need no stored read.

        An agent whose bound inputs have to be read from the store is left for an
        immediate off-lock re-drive, which reads them and records them there.
        """
        drive = False
        for task_id in engine.blocked_input_agents():
            snapshot = self._agent_input_snapshot_locked(engine, task_id)
            if snapshot is None:
                continue
            if snapshot.references:
                drive = True
                continue
            self._settle_agent_inputs_locked(workflow_id, engine, snapshot, {}, advance)
        if drive:
            self._redrive.drive_now(workflow_id)

    def _agent_input_snapshot_locked(
        self, engine: OrchestrationEngine, task_id: str
    ) -> _AgentInputSnapshot | None:
        """What an agent's pending input ports resolve from, as the ledger stands.

        Each member of a port whose producers are all bound is frozen to the result its
        producer settled with. A port with a member whose producer has nothing bound
        waits for a later advance; a producer that settled with nothing bound makes the
        whole input unreadable.
        """
        plan = engine.agent_input_plan(task_id)
        if plan is None:
            return None
        ports: list[_PortSnapshot] = []
        unreadable: str | None = None
        for port in plan.ports:
            members: list[tuple[InputMemberPlan, ValueRef]] = []
            for member in port.members:
                value_ref = ValueRef(
                    kind=member.value_ref_kind,
                    legacy_task_id=member.legacy_task_id,
                    collection_key=member.collection_key,
                    literal=member.literal,
                )
                if value_ref.kind == "legacy_task_result":
                    producer = value_ref.legacy_task_id or ""
                    binding = self._result_binding_locked(producer)
                    if binding is None or binding.reference is None:
                        if self._settled_unbound_locked(producer):
                            unreadable = f"task {producer} settled with no bound result"
                        break
                    value_ref = value_ref.model_copy(
                        update={"content": binding.reference}
                    )
                elif value_ref.kind not in ("inline", "empty"):
                    break
                members.append((member, value_ref))
            if unreadable is not None:
                break
            if len(members) == len(port.members):
                ports.append(
                    _PortSnapshot(
                        target_port=port.target_port,
                        provenance=port.provenance,
                        members=tuple(members),
                    )
                )
        return _AgentInputSnapshot(
            task_id=task_id,
            activation_id=plan.activation_id,
            ports=tuple(ports),
            unreadable=unreadable,
        )

    def _read_input_values(
        self, snapshots: list[_AgentInputSnapshot]
    ) -> dict[ContentReference, ResultEnvelope | Exception]:
        """Read every result the snapshots' inputs are frozen to, off the lock."""
        values: dict[ContentReference, ResultEnvelope | Exception] = {}
        for snapshot in snapshots:
            for reference in snapshot.references.values():
                if reference in values:
                    continue
                try:
                    values[reference] = self._results.read_reference(reference)
                except (ResultUnavailable, ResultUnreadable) as exc:
                    values[reference] = exc
        return values

    def _settle_agent_inputs_locked(
        self,
        workflow_id: str,
        engine: OrchestrationEngine,
        snapshot: _AgentInputSnapshot,
        values: dict[ContentReference, ResultEnvelope | Exception],
        advance: Advance,
    ) -> None:
        """Record an agent's accepted inputs from their read values, then re-admit.

        The budget counts the bytes of the resolved member strings; inputs over it fail
        the agent as a declared failure.
        """
        task_id = snapshot.task_id
        if snapshot.unreadable is not None:
            advance.extend(
                engine.on_failed(
                    task_id, f"input_unreadable: {snapshot.unreadable}", retryable=False
                )
            )
            return
        read = [values.get(ref) for ref in snapshot.references.values()]
        if any(isinstance(value, ResultUnavailable) for value in read):
            self._redrive.schedule(workflow_id)
            return
        if unreadable := next(
            (value for value in read if isinstance(value, ResultUnreadable)), None
        ):
            advance.extend(
                engine.on_failed(
                    task_id, f"input_unreadable: {unreadable}", retryable=False
                )
            )
            return
        total_bytes = 0
        accepted: list[AcceptedInput] = []
        for port in snapshot.ports:
            texts = [
                _member_text(value_ref, values) for _member, value_ref in port.members
            ]
            if any(text is None for text in texts):
                continue
            total_bytes += sum(len((text or "").encode("utf-8")) for text in texts)
            accepted.append(
                AcceptedInput(
                    activation_id=snapshot.activation_id,
                    target_port=port.target_port,
                    provenance=port.provenance,
                    members=tuple(
                        AcceptedInputMember(
                            source_operator_id=member.source_operator_id,
                            source_activation_id=member.source_activation_id,
                            child_index=member.child_index,
                            outcome=PublicationOutcome(member.outcome),
                            value_ref=value_ref,
                            ordinal=member.ordinal,
                        )
                        for member, value_ref in port.members
                    ),
                )
            )
        if total_bytes > self._input_budget_bytes:
            advance.extend(
                engine.on_failed(
                    task_id,
                    f"input_too_large: resolved input is {total_bytes} bytes, "
                    f"over the {self._input_budget_bytes}-byte budget",
                    retryable=False,
                )
            )
            return
        for entry in accepted:
            engine.record_accepted_input(entry)
        advance.extend(engine.reconsider_admission(task_id))

    def _agent_input_bindings(
        self, engine: OrchestrationEngine, task_id: str
    ) -> tuple[InputBinding, ...]:
        """The first-turn input bindings for an agent's input ports.

        A member a producer's result supplies names that result for the worker to
        hydrate; an inline member carries its own literal.
        """
        bindings: list[InputBinding] = []
        for ordinal, accepted in enumerate(engine.accepted_inputs_for_task(task_id)):
            members = tuple(
                InputBindingMember(
                    source_operator_id=member.source_operator_id,
                    source_activation_id=member.source_activation_id,
                    child_index=member.child_index,
                    outcome=member.outcome.value,
                    value=_literal_text(member.value_ref),
                    source=self._member_source_locked(member.value_ref),
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

    def _member_source_locked(
        self, value_ref: ValueRef | None
    ) -> ResultValueRef | None:
        """The stored result an input member reads, when a producer supplies it."""
        if value_ref is None or value_ref.kind != "legacy_task_result":
            return None
        reference = value_ref.content
        if reference is None and value_ref.legacy_task_id:
            binding = self._result_binding_locked(value_ref.legacy_task_id)
            reference = binding.reference if binding is not None else None
        if reference is None:
            return None
        return ResultValueRef(reference=reference, element=_element_of(value_ref))

    def _synthesize_child_record(
        self, template: TaskRecord, child_task_id: str
    ) -> TaskRecord:
        """Clone a child-template record for one materialized child."""
        return template.model_copy(
            deep=True,
            update={
                "task_id": child_task_id,
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
            if not record or record.status in SETTLING_TASK_STATUSES:
                # A settling task is already on its way to a terminal; failing it would
                # overwrite the cancellation a settle path is still waiting to apply.
                continue
            self._fail_record_locked(record, reason)
            failed_now.append(task_id)
        if persist and failed_now:
            self._commit_locked(*failed_now)
        return failed_now

    def _fail_record_locked(self, record: TaskRecord, reason: str) -> None:
        task_id = record.task_id
        record.status = TaskStatus.FAILED
        record.error = reason
        record.assigned_worker = None
        record.finished_ts = time.time()
        self._failed.add(task_id)
        self._pending_deps.pop(task_id, None)
        self._remove_from_ready_locked(task_id)

    def _fail_workflow_locked(self, workflow_id: str, reason: str) -> None:
        """Fail a workflow in its ledger and every non-terminal task of it, and persist
        the terminal facts.

        What its work held is released once the caller leaves the lock, through
        ``_release_pending_terminations``, and only once the terminal ledger is
        durable: a report whose writes are held keeps it until its replay saves the
        ledger.
        """
        termination = self._terminate_workflow_locked(workflow_id, reason, reason)
        non_terminal = [
            task_id
            for task_id, record in self._tasks.items()
            if record.workflow_id == workflow_id
            and record.status not in TERMINAL_TASK_STATUSES
        ]
        self._fail_v2_records_locked(non_terminal, reason, persist=True)
        self._hold_termination_locked(workflow_id, termination)
        self._save_ledger_locked(workflow_id)
        self._reclaim_vault_if_settled_locked(workflow_id)
        self._cv.notify_all()

    def _fail_v1_dependents_locked(self, primary: str) -> list[tuple[str, str]]:
        """Fail every pending task downstream of a failed v1 task, however deep."""
        reason = dependency_failed(primary)
        impacted: list[tuple[str, str]] = []
        frontier = [primary]
        while frontier:
            failed = frontier.pop()
            for child in self._dependents.pop(failed, set()):
                if (pending := self._pending_deps.get(child)) is not None:
                    pending.discard(failed)
                record = self._tasks.get(child)
                if not record or record.status != TaskStatus.PENDING:
                    continue
                self._fail_record_locked(record, reason)
                impacted.append((child, reason))
                frontier.append(child)
        return impacted

    def plan_merge(
        self, task_id: str, max_batch_size: int, assigned_worker: str
    ) -> list[str]:
        if max_batch_size <= 1:
            return []
        try:
            with self._cv:
                return self._plan_merge_locked(task_id, max_batch_size, assigned_worker)
        finally:
            self._release_ended_workers()

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
        self._commit_locked(task_id, *siblings)
        return siblings

    def release_merge(self, task_id: str) -> None:
        try:
            with self._cv:
                self._release_merge_locked(task_id)
        finally:
            self._release_ended_workers()

    def _return_failed_merge_locked(self, record: TaskRecord, worker_id: str) -> bool:
        """Return a merged dispatch that failed or lost ``worker_id``, if it is one.

        Such a failure belongs to no single task in the batch, so the parent and every
        child still merged into it go back to the head of the queue to run alone,
        spending no attempt. Returns whether the report was for a merged dispatch to
        ``worker_id``.
        """
        if (
            record.merged_dispatch_worker != worker_id
            or record.status != TaskStatus.DISPATCHED
        ):
            return False
        task_id = record.task_id
        dispatch_id = record.dispatch_id
        moved = self._return_merged_children_locked(
            [task_id, *self._merge_children_map.pop(task_id, [])], unmerge=True
        )
        self._returned_dispatches[task_id] = (worker_id, dispatch_id, moved)
        record.merged_children = None
        self._commit_locked(*moved)
        return True

    def _release_merge_locked(self, task_id: str) -> None:
        if parent := self._tasks.get(task_id):
            parent.merged_children = None
        returned = self._return_merged_children_locked(
            self._merge_children_map.pop(task_id, [])
        )
        self._commit_locked(task_id, *returned)

    def merged_child_record(self, task_id: str, child_id: str) -> TaskRecord | None:
        """A child's record while it is still merged into ``task_id``'s dispatch."""
        with self._cv:
            record = self._tasks.get(child_id)
            if (
                record is None
                or record.status != TaskStatus.DISPATCHED
                or self._merge_parent_map.get(child_id) != task_id
            ):
                return None
            return record

    def release_merged_child(
        self, task_id: str, child_id: str, merge_key: str | None
    ) -> None:
        """Take one child out of a task's merge and return it to the ready queue, to
        merge next under ``merge_key``, or to run alone when it is None."""
        try:
            with self._cv:
                if parent := self._tasks.get(task_id):
                    parent.merged_children = [
                        sibling
                        for sibling in parent.merged_children or []
                        if sibling != child_id
                    ] or None
                if (siblings := self._merge_children_map.get(task_id)) and (
                    child_id in siblings
                ):
                    siblings.remove(child_id)
                returned: list[str] = []
                if self._merge_parent_map.get(child_id) == task_id and (
                    child := self._tasks.get(child_id)
                ):
                    child.merge_key = merge_key
                    _, selected_worker_hint = self._merge_key_by_task.get(
                        child_id, (None, None)
                    )
                    self._merge_key_by_task[child_id] = (
                        merge_key,
                        selected_worker_hint,
                    )
                    returned = self._return_merged_children_locked([child_id])
                self._commit_locked(task_id, *returned)
        finally:
            self._release_ended_workers()

    def _return_merged_children_locked(
        self, child_ids: list[str], unmerge: bool = False
    ) -> list[str]:
        """Return merged children to the head of the ready queue, spending no attempt.

        ``unmerge`` also drops a child's merge key, so its next dispatch runs it alone
        rather than merging it into another batch that may again leave it without a
        result of its own. Returns the children it moved, for the caller to commit.
        """
        returned: list[str] = []
        for child_id in child_ids:
            self._merge_parent_map.pop(child_id, None)
            child_record = self._tasks.get(child_id)
            if not child_record or child_record.status in TERMINAL_TASK_STATUSES:
                continue
            _reset_to_pending(child_record)
            child_record.merged_parent_id = None
            child_record.merge_slice = None
            if unmerge:
                child_record.merge_key = None
                self._merge_key_by_task.pop(child_id, None)
            self._remove_from_ready_locked(child_id)
            self._enqueue_ready_locked(child_id, front=True)
            returned.append(child_id)
        if returned:
            self._cv.notify_all()
        return returned

    def _partition_merged_children_locked(
        self, child_ids: list[str], child_references: dict[str, ContentReference]
    ) -> tuple[list[str], list[str]]:
        """Split merged children into those with a result of their own and the rest.

        A child's result counts as its own only in the scope control gave the child.
        """
        settled: list[str] = []
        unsettled: list[str] = []
        for child_id in child_ids:
            record = self._tasks.get(child_id)
            reference = child_references.get(child_id)
            if record is not None and self._accepted_reference(record, reference):
                settled.append(child_id)
            else:
                unsettled.append(child_id)
        return settled, unsettled

    def _settle_workflows_locked(self, record: TaskRecord) -> list[str]:
        """The workflows a task's settlement touches: its own and its merged
        children's."""
        return list(
            dict.fromkeys(
                [
                    record.workflow_id,
                    *(
                        child.workflow_id
                        for child_id in record.merged_children or []
                        if (child := self._tasks.get(child_id)) is not None
                    ),
                ]
            )
        )

    def _finalize_merged_child_success(
        self,
        child_id: str,
        worker_id: str | None,
        finished_ts: float,
        started_ts: float | None,
        usage: TaskUsage | None,
        reference: ContentReference,
    ) -> list[str]:
        ready_children: list[str] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return ready_children
        child_record.status = TaskStatus.DONE
        self._bind_result_locked(child_record, reference, None)
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

    # ------------------------------------------------------------------ #
    # State updates (dispatch & events)
    # ------------------------------------------------------------------ #

    def _holds_dispatch_locked(
        self, record: TaskRecord, worker_id: str | None, dispatch_id: str | None
    ) -> bool:
        """Whether an event from ``worker_id`` belongs to the dispatch holding a task.

        A task is held by its recorded dispatch, or by one published and not yet
        recorded. An event naming no dispatch matches on its worker. A root-internal
        transition names no worker and is not fenced.
        """
        if worker_id is None:
            return True
        held = [(record.assigned_worker, record.dispatch_id)]
        if (publish := self._publishing.get(record.task_id)) and not publish.recorded:
            held.append((publish.worker_id, publish.dispatch_id))
        return any(
            worker_id == held_worker
            and (dispatch_id is None or dispatch_id == held_dispatch)
            for held_worker, held_dispatch in held
        )

    def _accepts_event_locked(
        self, record: TaskRecord, worker_id: str | None, dispatch_id: str | None
    ) -> bool:
        """Whether an event belongs to the dispatch holding a task.

        The first event of a dispatch published and not yet recorded records it, so
        the event applies to a recorded dispatch.
        """
        if not self._holds_dispatch_locked(record, worker_id, dispatch_id):
            return False
        if worker_id is not None and (publish := self._publishing.get(record.task_id)):
            publish.reported = True
            if not publish.recorded and record.status == TaskStatus.PENDING:
                self._record_dispatch_locked(record, publish)
        return True

    def holds_dispatch(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> bool:
        """Whether an event from ``worker_id`` belongs to the task's dispatch."""
        with self._lock:
            record = self._tasks.get(task_id)
            return record is not None and self._holds_dispatch_locked(
                record, worker_id, dispatch_id
            )

    def _reported[O: (SettleOutcome, FailureOutcome)](
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
        if worker_id is None or self._report_writes.held is not None:
            return transition()
        with self._lock:
            # A stash matches only the same report from the same worker, naming its
            # dispatch or none.
            pending = self._unacknowledged.get(task_id)
            if pending is not None and (
                pending.report != report
                or pending.worker_id != worker_id
                or dispatch_id not in (None, pending.dispatch_id)
            ):
                pending = None
            if pending is not None:
                # Make the writes the first handling held back, before the replay
                # writes anything after them.
                self._recommit_locked(pending.held)
        # A fresh hold collects every write that fails from here on, so the transition
        # completes in memory and its writes stay ordered.
        held = self._report_writes.held = _HeldWrites()
        try:
            outcome = transition()
        except Exception:
            # A replay that raises keeps its stash, now holding the replay's writes.
            if pending is not None and held.error is not None:
                with self._lock:
                    pending.held = held
            raise
        finally:
            self._report_writes.held = None
        # A replay sees its own event as stale, so it answers with what the first
        # handling did.
        if pending is not None:
            outcome = cast(O, pending.outcome)
        with self._lock:
            # A failed write stashes the report for its next handling; a clean replay
            # clears the stash unless a newer handling replaced it.
            if held.error is not None:
                # A stash of another report is replaced, but what it held stays held.
                replaced = self._unacknowledged.get(task_id)
                if replaced is not None and replaced is not pending:
                    held.follow(replaced.held)
                self._unacknowledged[task_id] = _Unacknowledged(
                    report, worker_id, dispatch_id, held, outcome
                )
            elif pending is not None and self._unacknowledged.get(task_id) is pending:
                del self._unacknowledged[task_id]
        # The caller sees the failure so the report is delivered again; a stashed hold
        # never raises it twice.
        if (error := held.error) is not None:
            held.error = None
            raise error
        return outcome

    def begin_publish(
        self,
        task_id: str,
        worker: Worker,
        dispatch_id: str | None,
        *,
        input_preparation: bool = False,
    ) -> bool:
        """Mark a dispatch as being published, so its worker's earliest events apply.

        Returns whether the task is pending and may be published.
        """
        publish = _Publish(
            worker.id, dispatch_id, _supplier_id(worker), input_preparation
        )
        with self._cv:
            record = self._tasks.get(task_id)
            if record is None or record.status != TaskStatus.PENDING:
                return False
            self._publishing[task_id] = publish
            return True

    def abandon_publish(self, task_id: str) -> bool:
        """Drop the mark of a dispatch whose publish failed.

        Returns whether the task still needs returning: a dispatch its worker reported
        on stands, and one a cancel recorded settles CANCELLED.
        """
        try:
            with self._cv:
                publish = self._publishing.pop(task_id, None)
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
        finally:
            self._release_ended_workers()

    def mark_dispatched(self, task_id: str) -> bool:
        """Record the publish `begin_publish` marked; returns whether it holds the task.

        A dispatch an event of its worker recorded first is not recorded again, and one
        that ended before its record, or whose task already settles, records nothing.
        """
        try:
            with self._cv:
                publish = self._publishing.pop(task_id, None)
                if publish is None:
                    return False
                record = self._tasks.get(task_id)
                if publish.recorded:
                    return (
                        record is not None
                        and record.dispatch_id == publish.dispatch_id
                        and record.status
                        in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING)
                    )
                if not record or record.status in SETTLING_TASK_STATUSES:
                    # A replayed or late dispatch must not regress a settling task.
                    return False
                self._record_dispatch_locked(record, publish)
                return True
        finally:
            self._release_ended_workers()

    def _record_dispatch_locked(self, record: TaskRecord, publish: _Publish) -> None:
        self._take_dispatch_locked(record, publish)
        self._workflow_registry.commit_transition(
            record.workflow_id,
            records=self._records_locked(record.task_id),
            dispatched=[record.task_id],
        )
        self._save_ledger_locked(record.workflow_id)

    def _take_dispatch_locked(self, record: TaskRecord, publish: _Publish) -> None:
        """Record a published dispatch as the one holding its task, in memory."""
        task_id = record.task_id
        publish.recorded = True
        self._returned_dispatches.pop(task_id, None)
        if (stash := self._unacknowledged.get(task_id)) and stash.dispatch_id is None:
            # A report naming no dispatch matches on its worker alone, which the next
            # dispatch may share.
            del self._unacknowledged[task_id]
        record.status = TaskStatus.DISPATCHED
        record.assigned_worker = publish.worker_id
        record.dispatch_id = publish.dispatch_id
        if publish.dispatch_id is not None:
            held = (publish.worker_id, publish.dispatch_id)
            earlier = self._held_dispatches.get(task_id)
            if earlier is not None and earlier != held:
                self._ended_dispatches.append(earlier)
            self._held_dispatches[task_id] = held
        record.merged_dispatch_worker = (
            publish.worker_id if self._merge_children_map.get(task_id) else None
        )
        record.topic = "tasks"
        record.dispatched_ts = time.time()
        record.next_retry_at = None
        record.supplier_id = publish.supplier_id
        self._remove_from_ready_locked(task_id)
        self._merge_bucket_remove(task_id)
        if engine := self._engines.get(record.workflow_id):
            if publish.input_preparation:
                engine.on_input_preparation_dispatched(task_id, publish.worker_id)
            else:
                engine.on_dispatched(task_id, publish.worker_id)

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
        try:
            with self._cv:
                record = self._tasks.get(task_id)
                if not record or not self._accepts_event_locked(
                    record, worker_id, dispatch_id
                ):
                    return EventEffect.STALE
                if record.status in SETTLING_TASK_STATUSES:
                    # A replayed or late start must not regress a settling task.
                    return EventEffect.SETTLED
                record.status = TaskStatus.DISPATCHED
                record.started_ts = started_ts
                self._workflow_registry.commit_transition(
                    record.workflow_id,
                    records=self._records_locked(task_id),
                    dispatched=[task_id],
                )
                if engine := self._engines.get(record.workflow_id):
                    engine.on_started(task_id)
                    self._save_ledger_locked(record.workflow_id)
                return EventEffect.APPLIED
        finally:
            self._release_ended_workers()

    def mark_updated(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        dispatch_id: str | None = None,
    ) -> EventEffect:
        """Store a task's latest progress update."""
        try:
            with self._lock:
                record = self._tasks.get(task_id)
                if record is None or not self._accepts_event_locked(
                    record, worker_id, dispatch_id
                ):
                    return EventEffect.STALE
                if record.status in TERMINAL_TASK_STATUSES:
                    # A replayed or late progress update must not touch a terminal task.
                    return EventEffect.SETTLED
                record.latest_update = payload
                self._persist_locked(task_id)
                return EventEffect.APPLIED
        finally:
            self._release_ended_workers()

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
        try:
            return self._reported(
                "TASK_SUCCEEDED",
                task_id,
                worker_id,
                dispatch_id,
                lambda: self._apply_success(
                    task_id, worker_id, payload, ts, dispatch_id, skip
                ),
            )
        finally:
            # A success whose fan-out cannot be read fails its workflow.
            self._release_pending_terminations()

    def _apply_success(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        dispatch_id: str | None,
        skip: dict[str, Any] | None,
    ) -> SettleOutcome:
        reference = _reported_reference(payload.get("result_reference"))
        child_references = _reported_child_references(payload)
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
            if record is not None and not self._accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                if worker_id is not None:
                    self._heal_returned_locked(task_id, worker_id, dispatch_id)
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
                return _settle_outcome(effect, record, [], [])
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
                    self._reap_captures_locked(
                        worker_id,
                        task_id,
                        _captured_calls(harness_result, carried_group),
                    )
                    if harness_result.kind is not HarnessResultKind.COMPLETION:
                        return _settle_outcome(effect, record, [], [])
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
                    self._release_ended_dispatches_locked([task_id])
                    return _settle_outcome(
                        effect, record, [], _in_flight_usage(task_id, payload)
                    )
                if record.status == TaskStatus.CANCELLING:
                    # A completion racing the cancel settles it before routing a
                    # captured facade group or consulting the reroute guard.
                    usages = self._settle_cancelled_usage_locked(
                        record, payload, finished_ts, started_ts
                    )
                    self._reap_captures_locked(
                        worker_id, task_id, _captured_calls(harness_result, group)
                    )
                    return _settle_outcome(effect, record, [], usages)
                if group is not None:
                    # The gateway captured a turn-scoped facade group: the clean
                    # turn-completion is a yield on that group, not the episode's
                    # terminal result, so it routes the whole ordered membership
                    # kind-specifically, consumed durably in the same reroute save.
                    record.pending_facade_group = None
                    self._route_and_dispatch_facade_group_locked(
                        task_id, group, harness_result.capsule
                    )
                    self._release_ended_dispatches_locked([task_id])
                    return _settle_outcome(
                        effect, record, [], _in_flight_usage(task_id, payload)
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
                    return _settle_outcome(EventEffect.STALE, record, [], [])
            if record:
                if record.status == TaskStatus.CANCELLED:
                    return _settle_outcome(effect, record, [], usages)
                if record.status == TaskStatus.DONE:
                    # Idempotent: a replayed TASK_SUCCEEDED must not re-apply, but
                    # re-persist in case the original completion's write failed
                    # after its in-memory commit.
                    for workflow_id in self._settle_workflows_locked(record):
                        self._repersist_terminal_workflow_locked(workflow_id)
                    # Recover a settlement or fan-out lost to a failed commit or a crash
                    # between the producer's terminal persist and its children: both
                    # are no-ops once applied.
                    if self._finish_success_locked(record, fanout):
                        self._cv.notify_all()
                    return _settle_outcome(effect, record, [], [])
                if record.status == TaskStatus.FAILED:
                    self._logger.warning(
                        "Ignoring TASK_SUCCEEDED for task %s in terminal status FAILED",
                        task_id,
                    )
                    return _settle_outcome(effect, record, [], [])
                if record.status == TaskStatus.CANCELLING:
                    # The cancel already resolved this task's declared output to its
                    # cancellation outcome and withheld the dispatch that would have
                    # carried a terminal back, so its completion settles the
                    # cancellation rather than reporting a success the ledger does
                    # not publish.
                    return _settle_outcome(
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
                self._bind_result_locked(record, reference, skip)
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

            settled_children, unsettled_children = (
                self._partition_merged_children_locked(
                    merged_children_ids, child_references
                )
            )
            if record is not None:
                record.merged_children = settled_children or None
            for merged_child in settled_children:
                ready_children.extend(
                    self._finalize_merged_child_success(
                        merged_child,
                        worker_id,
                        finished_ts,
                        started_ts,
                        usage,
                        child_references[merged_child],
                    )
                )
            returned = self._return_merged_children_locked(
                unsettled_children, unmerge=True
            )

            if record is not None:
                for workflow_id in self._settle_workflows_locked(record):
                    ready_children.extend(
                        self._try_advance_epoch_frontier_locked(workflow_id)
                    )

            self._commit_locked(task_id, *settled_children, *returned)

            notify = bool(ready_children)
            if record is not None and not self._writes_held():
                notify |= self._finish_success_locked(record, fanout)
            if notify:
                self._cv.notify_all()

            return _settle_outcome(effect, record, settled_children, usages)

    def _finish_success_locked(
        self, record: TaskRecord, fanout: _FanoutRead | None
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
            self._save_ledger_locked(record.workflow_id)
        self._reclaim_vault_if_settled_locked(record.workflow_id)
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
    ) -> FailureOutcome:
        """Apply a worker's report that its dispatch of a task failed.

        A report from the dispatch holding a running task returns a merged dispatch to
        run its tasks alone. A task whose inputs were in a store its worker could not
        reach returns without spending an attempt and is held until control has read
        them itself; a report naming no input the task consumes is an ordinary failure.
        Otherwise the failure is charged to the worker and the task either returns to
        the head of the queue for another attempt or settles: FAILED, or CANCELLED when
        a cancel is already under way. A report on a settled task persists its
        settlement again, and one from any other dispatch is dropped.
        """
        # Control's verdict on a held task's input is a report of its own, so it never
        # replays the worker report of the same dispatch.
        report = (
            _INPUT_VERDICT_REPORT
            if failure_kind is TaskFailureKind.INPUT_UNREADABLE
            else "TASK_FAILED"
        )
        try:
            return self._reported(
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
                ),
            )
        finally:
            # A replayed report's recommit makes a held terminal ledger durable.
            self._release_pending_terminations()

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
    ) -> FailureOutcome:
        with self._cv:
            record = self._tasks.get(task_id)
            if record is not None and self._is_input_verdict_locked(
                record, worker_id, dispatch_id, failure_kind
            ):
                # The worker's own report of this dispatch, stashed after a failed
                # write, is superseded: make what it held back and drop it, so its
                # redelivery finds the task settled.
                if (
                    (stash := self._unacknowledged.get(task_id)) is not None
                    and stash.report == "TASK_FAILED"
                    and stash.worker_id == worker_id
                    and stash.dispatch_id in (None, dispatch_id)
                ):
                    self._recommit_locked(stash.held)
                    del self._unacknowledged[task_id]
                record.last_error = error
                impacted, usages = self._mark_failed(
                    task_id, worker_id, payload, ts, error=error
                )
                return FailureOutcome(
                    DispatchEnd.FAILED, record.attempts, impacted, usages
                )
            if record is None or not self._accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                self._heal_returned_locked(task_id, worker_id, dispatch_id)
                return FailureOutcome(DispatchEnd.STALE, 0, [], [])
            if record.status in TERMINAL_TASK_STATUSES:
                self._repersist_terminal_workflow_locked(record.workflow_id)
                return FailureOutcome(DispatchEnd.SETTLED, record.attempts, [], [])
            if self._return_failed_merge_locked(record, worker_id):
                return FailureOutcome(
                    DispatchEnd.MERGE_RETURNED, record.attempts, [], []
                )
            if error:
                record.last_error = error
            if (
                failure_kind is TaskFailureKind.INPUT_UNAVAILABLE
                and record.status != TaskStatus.CANCELLING
            ):
                if consumed := self._consumed_inputs_locked(
                    record, unavailable_inputs or ()
                ):
                    held_dispatch = record.dispatch_id
                    end = self._return_dispatch_locked(
                        record, increment_retry=False, front=False
                    )
                    self._hold_for_input_check_locked(
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
            if _failed_task_can_retry(record, retryable):
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

    def _consumed_inputs_locked(
        self, record: TaskRecord, references: Sequence[ContentReference]
    ) -> tuple[ContentReference, ...]:
        """The named objects the task consumes, each once."""
        return tuple(
            reference
            for reference in dict.fromkeys(references)
            if self._consumes_locked(record, reference)
        )

    def _hold_for_input_check_locked(
        self,
        record: TaskRecord,
        worker_id: str,
        dispatch_id: str | None,
        references: tuple[ContentReference, ...],
    ) -> None:
        """Keep a returned task out of the queue until control has read its inputs."""
        self._remove_from_ready_locked(record.task_id)
        self._input_checks[record.task_id] = _InputCheck(
            worker_id, dispatch_id, references
        )
        self._redrive.drive_now(record.workflow_id)

    def _is_input_verdict_locked(
        self,
        record: TaskRecord,
        worker_id: str,
        dispatch_id: str | None,
        failure_kind: TaskFailureKind | None,
    ) -> bool:
        """Whether a failure is control's verdict that a held task's input is
        unreadable.

        The held check stands in for the dispatch the task returned from, so the verdict
        reports as that dispatch.
        """
        check = self._input_checks.get(record.task_id)
        return (
            failure_kind is TaskFailureKind.INPUT_UNREADABLE
            and check is not None
            and check.unreadable is not None
            and record.status == TaskStatus.PENDING
            and (check.worker_id, check.dispatch_id) == (worker_id, dispatch_id)
        )

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
        try:
            with self._cv:
                record = self._tasks.get(task_id)
                if record is None or not self._holds_dispatch_locked(
                    record, holder, None
                ):
                    if holder is not None:
                        self._heal_returned_locked(task_id, holder, None)
                    return DispatchEnd.STALE
                if holder is not None and self._dispatch_ended_at_suspension_locked(
                    record
                ):
                    return DispatchEnd.STALE
                if record.status in TERMINAL_TASK_STATUSES:
                    return DispatchEnd.SETTLED
                if record.status == TaskStatus.CANCELLING:
                    self._settle_cancelled_locked(record, time.time(), unmerge=True)
                    return DispatchEnd.CANCELLED
                if holder is not None and self._return_failed_merge_locked(
                    record, holder
                ):
                    return DispatchEnd.MERGE_RETURNED
                return self._return_dispatch_locked(
                    record, increment_retry=increment_retry, front=front
                )
        finally:
            self._release_ended_workers()

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
            *self._return_merged_children_locked(
                self._merge_children_map.pop(task_id, [])
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
            self._save_ledger_locked(record.workflow_id)
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
            self._returned_dispatches[task_id] = (
                record.assigned_worker,
                record.dispatch_id,
                moved,
            )
        _reset_to_pending(record)
        if self._enqueue_ready_locked(task_id, front=front):
            self._cv.notify_all()
        self._commit_locked(*moved)

    def _reap_stale_captures_locked(
        self, record: TaskRecord, worker_id: str, payload: dict[str, Any]
    ) -> None:
        """Reap the requests a stale agent step captured, which control never runs.

        A worker that holds the task again may have captured the same boundary anew,
        so its requests are left to that dispatch.
        """
        if self._holds_dispatch_locked(record, worker_id, None):
            return
        step = payload.get("agent_episode")
        carried = payload.get("agent_episode_facade_group")
        self._reap_captures_locked(
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

    def _reap_captures_locked(
        self,
        worker_id: str | None,
        task_id: str,
        captures: list[tuple[str, str | None]],
    ) -> None:
        """Reap the requests a step captured for boundaries control never runs."""
        for call, interface in captures:
            self._reap_captured_request_locked(worker_id, task_id, call, interface)

    def _heal_returned_locked(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> None:
        """Commit again what returning a dispatch moved, on a report of that dispatch.

        Such a report may be handled again because the return's commit failed.
        """
        if (returned := self._returned_dispatches.get(task_id)) is None:
            return
        held_worker, held_dispatch, moved = returned
        if worker_id == held_worker and dispatch_id in (None, held_dispatch):
            self._recommit_locked(_HeldWrites(moved.copy()))

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
        try:
            return self._mark_failed(task_id, worker_id, payload, ts, error=error)
        finally:
            self._release_pending_terminations()

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
                    self._repersist_terminal_workflow_locked(record.workflow_id)
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
            self._pending_deps.pop(task_id, None)
            self._remove_from_ready_locked(task_id)
            merged_children_ids = self._merge_children_map.pop(task_id, [])
            self._merge_key_by_task.pop(task_id, None)

            impacted = self._fail_v1_dependents_locked(task_id)

            engine = self._engines.get(record.workflow_id) if record else None
            advance = Advance()
            if engine is not None:
                advance = engine.on_failed(task_id, message, retryable=False)
                impacted.extend(
                    self._fail_v2_advance_locked(engine, advance, persist=False)
                )

            returned = self._return_merged_children_locked(
                merged_children_ids, unmerge=True
            )

            failed_epoch = self._task_epoch_index.get(task_id)
            if record and failed_epoch is not None:
                blocked, blocked_returned = self._fail_later_epochs_locked(
                    record.workflow_id,
                    failed_epoch,
                    f"Blocked by failed task {task_id} in earlier epoch",
                )
                impacted.extend(blocked)
                returned += blocked_returned

            self._commit_locked(task_id, *(dep_id for dep_id, _ in impacted), *returned)
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
                self._save_ledger_locked(record.workflow_id)

            if record is not None:
                self._reclaim_vault_if_settled_locked(record.workflow_id)
            return impacted, usages

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def cancel_workflow(self, workflow_id: str, reason: str = "cancelled") -> list[str]:
        touched: list[str] = []
        returned: list[str] = []
        with self._cv:
            workflow_tasks = [
                item
                for item in self._tasks.items()
                if item[1].workflow_id == workflow_id
            ]
            if not workflow_tasks:
                return touched  # Unknown workflow: no records to move
            termination = self._terminate_workflow_locked(workflow_id, reason, None)
            for task_id, record in workflow_tasks:
                if (moved := self._cancel_record_locked(record, reason)) is None:
                    continue
                returned += moved
                touched.append(task_id)
            self._commit_cancelled_locked(workflow_id, touched, returned)
            # The ledger snapshot follows the committed task state so it never leads
            # it.
            if workflow_id in self._engines:
                self._settle_suspended_cancels_locked(self._engines[workflow_id])
                self._save_ledger_locked(workflow_id)

            # A whole-workflow cancel commits its terminals here rather than through the
            # per-task terminal persist, so it notifies for itself. A cancel that leaves
            # tasks CANCELLING settles nothing yet; the finalizer reads that and waits
            # for their own terminals.
            self._notify_terminal_transition(workflow_id)

        self._release_terminated_work(termination)
        self._release_ended_workers()
        self._credential_vault.purge(workflow_id)
        return touched

    def _terminate_workflow_locked(
        self, workflow_id: str, reason: str, failure: str | None
    ) -> _Termination:
        """Settle a workflow's ledger terminally and take what its work still holds.

        Runs before the caller moves the task records. A cancel (``failure`` None)
        resolves the unpublished outputs as cancelled and a control failure as declared
        failures. Either way a dispatch being published is recorded so its worker is
        interrupted with every other running task, the agents' mediated operations are
        taken for reaping, the pending re-drive and held input checks are dropped, and
        every unsettled boundary invocation is terminalized. It writes nothing: the
        caller's terminal commit persists it, and ``_release_terminated_work`` releases
        what it returns once that commit is made.
        """
        self._redrive.settle(workflow_id)
        termination = self._take_task_work_locked(
            [
                record
                for record in self._tasks.values()
                if record.workflow_id == workflow_id
            ],
            reason,
        )
        self._workflow_epoch_tasks.pop(workflow_id, None)
        self._workflow_epoch_frontier.pop(workflow_id, None)
        self._workflow_in_epoch_order.pop(workflow_id, None)
        if (engine := self._engines.get(workflow_id)) is None:
            return termination
        if failure is None:
            engine.cancel_instance()
        else:
            engine.fail_instance(failure)
        termination.resident_invocation_ids = engine.terminalize_unsettled_invocations()
        return termination

    def _take_task_work_locked(
        self, records: list[TaskRecord], reason: str
    ) -> _Termination:
        """Take what the given tasks' work still holds, before a cancel moves them.

        A dispatch being published is recorded so its worker is interrupted with every
        other running task, the agents' mediated operations are taken for reaping, and
        the held input checks are dropped. It writes nothing.
        """
        interrupts: list[InterruptMessage] = []
        for record in records:
            publish = self._publishing.get(record.task_id)
            if publish and not publish.recorded and record.status == TaskStatus.PENDING:
                # The worker may already be running the task.
                self._take_dispatch_locked(record, publish)
            if record.status == TaskStatus.DISPATCHED and (
                interrupt := self._interrupt_for(record, reason)
            ):
                interrupts.append(interrupt)
            self._input_checks.pop(record.task_id, None)
            self._task_epoch_index.pop(record.task_id, None)
        reaps = self._take_ops_for_agents_locked([r.task_id for r in records])
        return _Termination(interrupts, reaps)

    @staticmethod
    def _interrupt_for(record: TaskRecord, reason: str) -> InterruptMessage | None:
        # A merged child's batch keeps running for siblings from other workflows.
        if not record.assigned_worker or record.merged_parent_id:
            return None
        return InterruptMessage(
            task_id=record.task_id,
            worker_id=record.assigned_worker,
            reason=reason,
            dispatch_id=record.dispatch_id,
        )

    def _revoke_locked(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> None:
        """Queue the revocation of a dispatch that resolved without its worker ending
        it."""
        if dispatch_id is None:
            return
        self._pending_terminations.append(
            _Termination(
                [],
                [],
                revokes=[
                    RevokeMessage(
                        task_id=task_id, worker_id=worker_id, dispatch_id=dispatch_id
                    )
                ],
            )
        )

    def _interrupt_cancelling_locked(self, workflow_id: str) -> None:
        """Queue an interrupt for each task a restart found still being cancelled,
        whose worker may never have received one."""
        interrupts = [
            interrupt
            for record in self._tasks.values()
            if record.workflow_id == workflow_id
            and record.status == TaskStatus.CANCELLING
            and (interrupt := self._interrupt_for(record, record.error or "cancelled"))
        ]
        if interrupts:
            self._pending_terminations.append(_Termination(interrupts, []))

    def _release_terminated_work(self, termination: _Termination) -> None:
        """Release what a terminated workflow's work held, best effort: each resident
        credit, each interrupt, and each reap is attempted however the others fare."""
        # The fenced terminal releases each in-flight resident invocation's credit, so a
        # lost or draining replica is not held forever.
        for invocation_id in termination.resident_invocation_ids:
            try:
                self._release_resident_credit(invocation_id, failed=True)
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
            try:
                worker = self._worker_registry.get_worker(revoke.worker_id)
                if worker is None:
                    self._logger.warning(
                        "Cannot revoke dispatch %s of %s; worker %s missing",
                        revoke.dispatch_id,
                        revoke.task_id,
                        revoke.worker_id,
                    )
                else:
                    self._worker_registry.publish_revoke(worker.node_id, revoke)
            except Exception:
                self._logger.exception(
                    "Revoking dispatch %s on %s failed",
                    revoke.dispatch_id,
                    revoke.worker_id,
                )
        # The worker drops a reaped operation and its custody.
        for worker_id, agent_task_id, call in termination.reaps:
            try:
                self._reap_mediated_op(worker_id, agent_task_id, call)
            except Exception:
                self._logger.exception(
                    "Reaping the operation of %s on %s failed", agent_task_id, worker_id
                )

    def _release_pending_terminations(self) -> None:
        """Release, off the lock, what workflows control failed under it still hold,
        and each worker reserved for a dispatch that ended."""
        with self._lock:
            pending, self._pending_terminations = self._pending_terminations, []
        for termination in pending:
            self._release_terminated_work(termination)
        self._release_ended_workers()

    def _commit_cancelled_locked(
        self, workflow_id: str, touched: list[str], returned: list[str]
    ) -> None:
        """Commit the tasks a cancel moved, then the merged children it returned to the
        queue."""

        def commit() -> None:
            self._workflow_registry.commit_transition(
                workflow_id,
                records=self._records_locked(*touched),
                dispatched=[
                    task_id
                    for task_id in touched
                    if _membership(self._tasks[task_id]) == TaskStatus.DISPATCHED
                ],
                done=[
                    task_id
                    for task_id in touched
                    if _membership(self._tasks[task_id]) == TaskStatus.DONE
                ],
                cancelled=[
                    task_id
                    for task_id in touched
                    if _membership(self._tasks[task_id]) == TaskStatus.CANCELLED
                ],
                sched=self._sched_locked(workflow_id),
            )

        self._write_locked(commit, lambda held: held.task_ids.extend(touched))
        self._commit_locked(
            *(task_id for task_id in returned if task_id not in touched)
        )

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
        termination = self._take_task_work_locked(records, _RESIDUAL_CANCEL_REASON)
        touched: list[str] = []
        returned: list[str] = []
        for record in records:
            moved = self._cancel_record_locked(record, _RESIDUAL_CANCEL_REASON)
            if moved is None:
                continue
            record.residual_cancel = True
            touched.append(record.task_id)
            returned += moved
        self._commit_cancelled_locked(workflow_id, touched, returned)
        if not self._writes_held():
            self._notify_terminal_transition(workflow_id)
        self._settle_suspended_cancels_locked(engine)
        termination.resident_invocation_ids = engine.terminalize_unsettled_invocations(
            touched
        )
        self._hold_termination_locked(workflow_id, termination)
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
        if (parent_id := self._merge_parent_map.pop(task_id, None)) is not None:
            if (siblings := self._merge_children_map.get(parent_id)) and (
                task_id in siblings
            ):
                siblings.remove(task_id)
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
        self._pending_deps.pop(task_id, None)
        self._remove_from_ready_locked(task_id)
        self._merge_bucket_remove(task_id)
        self._merge_key_by_task.pop(task_id, None)
        return self._return_merged_children_locked(
            self._merge_children_map.pop(task_id, []), unmerge
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
        try:
            return self._reported(
                "TASK_CANCELLED",
                task_id,
                worker_id,
                dispatch_id,
                lambda: self._apply_cancellation(
                    task_id, worker_id, payload, ts, dispatch_id
                ),
            )
        finally:
            # A replayed report's recommit makes a held terminal ledger durable.
            self._release_pending_terminations()

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
            if record is None or not self._accepts_event_locked(
                record, worker_id, dispatch_id
            ):
                return _settle_outcome(EventEffect.STALE, record, [], [])
            if record.status == TaskStatus.CANCELLED:
                # Idempotent: a replayed cancellation must not re-apply, but
                # re-persist in case the original cancellation's write failed
                # after its in-memory commit.
                self._repersist_terminal_workflow_locked(record.workflow_id)
                return _settle_outcome(EventEffect.SETTLED, record, [], [])
            if record.status in (TaskStatus.DONE, TaskStatus.FAILED):
                self._logger.warning(
                    "Ignoring cancellation for task %s in terminal status %s",
                    task_id,
                    record.status,
                )
                return _settle_outcome(EventEffect.SETTLED, record, [], [])
            if self._dispatch_ended_at_suspension_locked(record):
                return _settle_outcome(EventEffect.STALE, record, [], [])
            if record.status == TaskStatus.DISPATCHED:
                return self._return_given_up_locked(record, worker_id, payload)
            self._settle_cancelled_locked(
                record, finished_ts, started_ts=started_ts, usage=usage
            )
            return _settle_outcome(EventEffect.APPLIED, record, [], usages)

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
                self._commit_locked(record.task_id)
                usages.append((record.task_id, usage))
            return _settle_outcome(
                _LOSS_EFFECTS[loss.end], record, [], usages, loss.impacted
            )
        if worker_id is None or not self._return_failed_merge_locked(record, worker_id):
            self._return_dispatch_locked(record, increment_retry=False, front=True)
        return _settle_outcome(EventEffect.RETURNED, record, [], [])

    def _resolve_lost_locked(
        self, record: TaskRecord, *, spend_attempt: bool
    ) -> LossOutcome:
        """Resolve a v2 task whose worker is lost or gave it up.

        It returns to the queue when it can safely run again, spending an attempt when
        ``spend_attempt`` is set, and fails once its attempts run out; a task that
        cannot safely run again fails. ``impacted`` names each dependent that fails
        with it. A task nothing resolves ends STALE.
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
                return self._fail_lost_on_last_attempt_locked(record, engine)
        advance = self._resolve_uncertain_locked(record.task_id)
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
        self, record: TaskRecord, engine: OrchestrationEngine
    ) -> LossOutcome:
        """Fail a v2 task whose worker was lost on its last attempt, with the boundary
        work it held, as a failed task."""
        failed, _ = self._mark_failed(
            record.task_id,
            None,
            {},
            now_iso(),
            error=(
                f"Worker {record.assigned_worker} was lost on the last of "
                f"{record.max_attempts} attempts"
            ),
        )
        self._reap_ops_for_agents_locked(
            [record.task_id, *(task_id for task_id, _ in failed)]
        )
        if invocation_ids := engine.terminalize_unsettled_invocations([record.task_id]):
            self._hold_termination_locked(
                record.workflow_id,
                _Termination([], [], resident_invocation_ids=invocation_ids),
            )
            self._save_ledger_locked(record.workflow_id)
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
        self._commit_locked(task_id, *returned, sched=False)
        self._save_ledger_locked(record.workflow_id)
        self._reclaim_vault_if_settled_locked(record.workflow_id)

    def get_record(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def workflow_submitted_at(self, workflow_id: str) -> str | None:
        """The workflow's durable submission timestamp, or ``None`` if unknown."""
        record = self._workflow_registry.get_workflow_record(workflow_id)
        return record.submitted_at if record is not None else None

    def set_completion_notifier(self, notify: Callable[[str], None]) -> None:
        """Install the callback that a terminal transition notifies.

        The callback must be cheap and non-blocking: it runs under the scheduler lock.
        """
        self._on_workflow_settled = notify

    def workflow_settlement(self, workflow_id: str) -> WorkflowSettlement:
        """Whether every task of a workflow has settled, and the last of their
        finishes.

        Read under the scheduler lock, so a caller never observes the moment inside a
        settle in which a producer's tasks have gone terminal but the children they
        fan out do not exist yet.
        """
        with self._lock:
            return self._workflow_settlement_locked(workflow_id)

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

    def recover_tasks_for_worker(
        self, worker_id: str, *, spend_attempt: bool
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
        worker originated that boundary and holds its request.
        """
        try:
            return self._recover_tasks_for_worker(worker_id, spend_attempt)
        finally:
            self._release_pending_terminations()

    def _recover_tasks_for_worker(
        self, worker_id: str, spend_attempt: bool
    ) -> WorkerRecovery:
        recovered: list[str] = []
        resolved: list[LossOutcome] = []
        with self._cv:
            for task_id, record in list(self._tasks.items()):
                publish = self._publishing.get(task_id)
                if publish and not publish.recorded and publish.worker_id == worker_id:
                    self._publishing[task_id] = None
                    self._revoke_locked(task_id, worker_id, publish.dispatch_id)
                    children = self._merge_children_map.pop(task_id, [])
                    record.merged_children = None
                    self._commit_locked(
                        *self._return_merged_children_locked(
                            [task_id, *children], unmerge=bool(children)
                        )
                    )
                    continue
                if record.assigned_worker != worker_id:
                    continue
                if record.status not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                    continue
                # A boundary whose raw request only this worker holds is lost with it.
                if self._dispatch_ended_at_suspension_locked(record) and not (
                    self._engines[record.workflow_id].awaits_worker_held_boundary(
                        task_id
                    )
                ):
                    continue
                self._rehydrated_dispatched.pop(task_id, None)
                if not record.merged_parent_id:
                    self._revoke_locked(task_id, worker_id, record.dispatch_id)
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
            for permit_id, (_, _, op_worker) in list(self._pending_ops.items()):
                if op_worker == worker_id:
                    del self._pending_ops[permit_id]
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
        try:
            with self._cv:
                record = self._tasks.get(task_id)
                if (
                    record is None
                    or record.status
                    not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING)
                    or record.assigned_worker != worker_id
                    or record.dispatch_id != dispatch_id
                    or record.started_ts is not None
                    or not self._awaits_its_dispatch_locked(record)
                ):
                    return None
                since = max(
                    record.dispatched_ts or 0.0,
                    self._rehydrated_dispatched.get(task_id, 0.0),
                )
                if time.time() - since < bound_sec:
                    return None
                self._revoke_locked(task_id, worker_id, dispatch_id)
                if worker_id not in record.failed_workers:
                    record.failed_workers.append(worker_id)
                record.last_error = (
                    f"Worker {worker_id} kept reporting it does not hold dispatch "
                    f"{dispatch_id}"
                )
                if record.status == TaskStatus.CANCELLING:
                    self._settle_cancelled_locked(record, time.time(), unmerge=True)
                    return _settle_outcome(EventEffect.APPLIED, record, [], [])
                engine = self._engines.get(record.workflow_id)
                if engine is not None and engine.private_state_owner(task_id):
                    return self._retry_on_owner_locked(record, worker_id)
                return self._return_given_up_locked(record, worker_id, {})
        finally:
            self._release_pending_terminations()

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
            return _settle_outcome(EventEffect.RETURNED, record, [], [])._replace(
                spent=True
            )
        impacted, usages = self._mark_failed(
            record.task_id, worker_id, {}, now_iso(), error=record.last_error
        )
        return _settle_outcome(EventEffect.FAILED, record, [], usages, tuple(impacted))

    def _awaits_its_dispatch_locked(self, record: TaskRecord) -> bool:
        """Whether the task's dispatch may still run it: always for a cancelling or v1
        task; for any other v2 task, while its work item is unsettled and its dispatch
        did not end at a suspension."""
        engine = self._engines.get(record.workflow_id)
        if engine is None or record.status == TaskStatus.CANCELLING:
            return True
        wi = engine.work_item(record.task_id)
        return (
            wi is not None
            and wi.status not in TERMINAL_WORK_ITEM_STATUSES
            and not self._dispatch_ended_at_suspension_locked(record)
        )

    def _dispatch_ended_at_suspension_locked(self, record: TaskRecord) -> bool:
        """Whether a task's dispatch ended at a suspension: a step of it ran and
        suspended on a boundary, so its worker holds nothing of it, though the task
        stays DISPATCHED until the boundary settles."""
        engine = self._engines.get(record.workflow_id)
        wi = engine.work_item(record.task_id) if engine is not None else None
        return (
            record.status == TaskStatus.DISPATCHED
            and wi is not None
            and wi.status is WorkItemStatus.BLOCKED
        )

    def dispatch_in_flight(
        self, task_id: str, dispatch_id: str, worker_id: str
    ) -> bool:
        """Whether a dispatch to a worker is being published or holds its task, and has
        not ended at a suspension."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or self._dispatch_ended_at_suspension_locked(record):
                return False
            publish = self._publishing.get(task_id)
            in_flight = record.status in (
                TaskStatus.DISPATCHED,
                TaskStatus.CANCELLING,
            ) or (publish is not None and not publish.recorded)
            return in_flight and self._holds_dispatch_locked(
                record, worker_id, dispatch_id
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
                gpu = record.task.spec.gpu_requirements()
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
        element = self._input_element_locked(task_id)
        return TaskInfo(
            **dict(record),
            depends_on=sorted(self._original_deps.get(task_id, set())),
            pending_dependencies=sorted(self._pending_deps.get(task_id, set())),
            dependents=sorted(self._dependents.get(task_id, set())),
            completed=task_id in self._completed,
            failed=task_id in self._failed,
            input_element=(
                TaskInputElement(
                    producer_task_id=element.producer_task_id, index=element.ref.element
                )
                if element is not None
                else None
            ),
        )
