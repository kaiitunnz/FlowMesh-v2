"""Episode dispatch builders and the episode queries they rest on."""

import logging
from collections.abc import Callable

from shared.harness import (
    AgentEpisodeDispatch,
    EpisodeModelBinding,
    HarnessBackendKey,
    ServiceLeafEpisodeDispatch,
)
from shared.inference import (
    CanonicalInferenceContract,
    CanonicalProjectionError,
    InferenceSourceKind,
    canonical_contract,
    element_contract,
)
from shared.private_state import OwnerFence, PrivateStateAttachment
from shared.sandbox import (
    SANDBOX_EGRESS_INTERFACE,
    SANDBOX_EXECUTE_INTERFACE,
    LocalSandboxCapability,
    SandboxEgressMode,
)
from shared.tasks.specs import (
    InferenceEmbodimentKind,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    TaskSpecBase,
)
from shared.tools.facade import FacadeDescriptor, FacadeResolution

from ...orchestration import OrchestrationEngine, WorkItemStatus
from ...orchestration.state import TERMINAL_WORK_ITEM_STATUSES
from ..models import TaskRecord, TaskStatus
from ..v2.compiler.facades import run_command_schema
from ..v2.representations.admission import ResidentAdmissionBinding
from ..v2.representations.operators import (
    AgentModelGatewayBinding,
    AgentOperator,
    ResolvedEmbodiment,
    ServiceDependency,
)
from ..v2.representations.plan import EpisodeSpec, InferenceEmbodimentMenu
from .agent_inputs import AgentInputs
from .content_bindings import ContentBindings

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


def _resident_served_dependency_locked(
    engine: OrchestrationEngine, task_id: str
) -> ServiceDependency | None:
    """The resident dependency a leaf's dispatch is served from, or None."""
    dependency = engine.service_dependency(task_id)
    if dependency is None or engine.agent_operator(task_id) is not None:
        return None
    if engine.embodiment_menu(task_id) is not None:
        resolved = _resolved_embodiment_locked(engine, task_id)
        if (
            resolved is None
            or resolved.kind is not InferenceEmbodimentKind.RESIDENT_SERVED
        ):
            return None
    return dependency


def _resolved_embodiment_locked(
    engine: OrchestrationEngine, task_id: str
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


class EpisodeDispatch:
    """Builds agent and service episode dispatches and answers the episode queries
    they rest on."""

    def __init__(
        self,
        tasks: dict[str, TaskRecord],
        engines: dict[str, OrchestrationEngine],
        logger: logging.Logger,
        content_bindings: ContentBindings,
        agent_inputs: AgentInputs,
    ) -> None:
        self._tasks = tasks
        self._engines = engines
        self._logger = logger
        self._content_bindings = content_bindings
        self._agent_inputs = agent_inputs

    def resolve_model_binding(self, task_id: str) -> AgentModelGatewayBinding | None:
        """The pinned managed-model binding for a task's agent, for the gateway."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if engine is None:
            return None
        op = engine.agent_operator(task_id)
        return op.model_binding if op is not None else None

    def gateway_binding_for(
        self, task_id: str
    ) -> tuple[str, AgentModelGatewayBinding] | None:
        """The task's owning workflow and its pinned model binding, for the gateway."""
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
        """What the task binds for resident admission, read from its own plan node."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if record is None or engine is None:
            return None
        return engine.resident_admission_binding(record.workflow_id, task_id)

    def boundary_settleable(self, task_id: str, call_correlation: str) -> bool:
        """Whether a mediated boundary still awaits its outcome."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if record is None or engine is None:
            return False
        return engine.boundary_settleable(task_id, call_correlation)

    def private_state_owner(self, task_id: str) -> OwnerFence | None:
        """The holder that must supply a task's bound private state, or None."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        return engine.private_state_owner(task_id) if engine else None

    def agent_episode_dispatch(
        self, task_id: str, holder: OwnerFence
    ) -> AgentEpisodeDispatch | None:
        """The agent-episode context to ship with a dispatch, or None for a
        non-agent."""
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
        sandbox = _sandbox_capability(op, granted[1] if granted else None, invoke_face)
        # First-turn dataflow inputs are delivered only on the first dispatch; a
        # resume injects only the harness's own delivered outcomes.
        input_bindings = (
            self._agent_inputs.agent_input_bindings(engine, task_id)
            if capsule_blob is None
            else ()
        )
        return AgentEpisodeDispatch(
            backend=HarnessBackendKey(backend=harness.backend, version=harness.version),
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
        """The service-episode context for a resident leaf, or None."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if engine is None:
            return None
        dependency = _resident_served_dependency_locked(engine, task_id)
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
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        return (
            engine is not None
            and _resident_served_dependency_locked(engine, task_id) is not None
        )

    def embodiment_pinned(self, task_id: str) -> bool:
        """Whether a task's resolved embodiment is committed to the run carrying it."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        return engine.embodiment_pinned(task_id) if engine else False

    def embodiment_menu(self, task_id: str) -> InferenceEmbodimentMenu | None:
        """The embodiments a ready task's plan node offers, if it offers a menu."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        return engine.embodiment_menu(task_id) if engine else None

    def declared_contract(
        self, task_id: str, spec: TaskSpecBase | None = None
    ) -> CanonicalInferenceContract | None:
        """The contract a leaf carries to the worker, for it to resolve and report."""
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if record is None or engine is None:
            return None
        spec = spec or record.task.spec
        if not isinstance(spec, (InferenceSpecStrict, InferenceSpecTemplate)):
            return None
        element = self._content_bindings.input_element_locked(task_id)
        try:
            contract = (
                element_contract(spec, element.producer_task_id, element.ref.element)
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

    def awaits_its_dispatch_locked(self, record: TaskRecord) -> bool:
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
            and not self.dispatch_ended_at_suspension_locked(record)
        )

    def dispatch_ended_at_suspension_locked(self, record: TaskRecord) -> bool:
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
