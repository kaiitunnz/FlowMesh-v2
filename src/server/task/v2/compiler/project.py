from collections.abc import Mapping
from dataclasses import dataclass, field

from shared.tasks import TaskType
from shared.tasks.specs import (
    AgentSpecStrict,
    AgentSpecTemplate,
    ModelBindingMode,
)
from shared.tasks.specs.common import ModelSpecTemplate

from ...parser import ParsedTask, ParsedWorkflow
from ..representations.operators import (
    AgentOperator,
    BindingKey,
    ConditionGuard,
    EffectBoundary,
    EffectClass,
    EffectReplayContract,
    LeafOperator,
    LogicalOperator,
    ModelRef,
    Port,
    PortKind,
)
from ..representations.plan import (
    PhysicalNode,
    ResidencyIntent,
    ServiceFamilyRequirement,
)
from ..representations.results import (
    CardinalityKind,
    LegacyLogicalTaskProjection,
    ReleaseConditionKind,
    ResultDeclaration,
    Visibility,
)
from ..representations.template import (
    ResourceDeclaration,
    SourceKind,
    SourceMapEntry,
    TemplateEdge,
    ToolDeclaration,
)
from .agent_binding import (
    AgentBindingDefaults,
    resolve_agent_bindings,
    service_family_for_ref,
)
from .bindings import (
    BindingClass,
    binding_class,
    default_agent_authority,
    default_agent_boundary,
    is_training,
    leaf_profile,
)
from .diagnostics import compile_error


def _model_ref(task: ParsedTask) -> ModelRef | None:
    spec = task.task.spec
    if not isinstance(spec, ModelSpecTemplate):
        return None
    name = spec.model_name
    if not name:
        return None
    return ModelRef(architecture=name, version=spec.model_revision)


def _task_source(task: ParsedTask) -> tuple[SourceKind, str]:
    if task.graph_node_name:
        return "graph_node", task.graph_node_name
    if task.local_name:
        return "stage", task.local_name
    return "legacy_task", task.task_id


def _condition_guard(
    task: ParsedTask, name_to_op: dict[str, str], operator_ids: set[str]
) -> ConditionGuard | None:
    condition = task.task.spec.condition
    if condition is None:
        return None
    raw = condition.node.strip()
    operator_id = name_to_op.get(raw, raw)
    if operator_id not in operator_ids:
        source_kind, source_id = _task_source(task)
        raise compile_error(
            "guard.unknown-node",
            f"conditional guard references upstream {condition.node!r}, "
            "which resolves to no operator",
            source_id,
            source_kind,
        )
    return ConditionGuard(
        node=operator_id, field=condition.field, equals=condition.equals
    )


def _source_map_entry(task: ParsedTask) -> SourceMapEntry:
    source_kind, source_id = _task_source(task)
    return SourceMapEntry(
        logical_ref=task.task_id, source_kind=source_kind, source_id=source_id
    )


def _ports(
    task: ParsedTask, task_type: TaskType
) -> tuple[tuple[Port, ...], tuple[Port, ...]]:
    inputs: list[Port] = []
    outputs: list[Port] = [Port(name="out")]
    if task.depends_on:
        inputs.append(Port(name="in"))
    model_ref = _model_ref(task)
    if model_ref is not None:
        if is_training(task_type):
            produced = ModelRef(
                architecture=model_ref.architecture,
                version=f"trained:{task.task_id}",
            )
            outputs.append(
                Port(name="model_out", kind=PortKind.MODEL_REF, model_ref=produced)
            )
        else:
            inputs.append(
                Port(name="model_in", kind=PortKind.MODEL_REF, model_ref=model_ref)
            )
    return tuple(inputs), tuple(outputs)


def _leaf_operator(
    task: ParsedTask,
    task_type: TaskType,
    name_to_op: dict[str, str],
    operator_ids: set[str],
) -> LeafOperator:
    inputs, outputs = _ports(task, task_type)
    return LeafOperator(
        operator_id=task.task_id,
        source_ref=task.task_id,
        inputs=inputs,
        outputs=outputs,
        profile=leaf_profile(task_type),
        guard=_condition_guard(task, name_to_op, operator_ids),
        residency_only=binding_class(task_type) is BindingClass.RESIDENCY,
    )


def _agent_operator(
    task: ParsedTask,
    name_to_op: dict[str, str],
    operator_ids: set[str],
    defaults: AgentBindingDefaults,
    secret_ref: str | None,
) -> AgentOperator:
    inputs, outputs = _ports(task, TaskType.AGENT)
    spec = task.task.spec
    harness = (
        spec.harness if isinstance(spec, (AgentSpecStrict, AgentSpecTemplate)) else None
    )
    model_binding = (
        spec.model_binding
        if isinstance(spec, (AgentSpecStrict, AgentSpecTemplate))
        else None
    )
    harness_binding, gateway_binding = resolve_agent_bindings(
        harness, model_binding, defaults, secret_ref
    )
    return AgentOperator(
        operator_id=task.task_id,
        source_ref=task.task_id,
        inputs=inputs,
        outputs=outputs,
        binding=BindingKey(task_type=TaskType.AGENT),
        harness_binding=harness_binding,
        model_binding=gateway_binding,
        authority=default_agent_authority(),
        boundary=default_agent_boundary(),
        guard=_condition_guard(task, name_to_op, operator_ids),
    )


@dataclass
class LoweringAccumulator:
    """Mutable collector the compiler fills from tasks and structured regions."""

    operators: list[LogicalOperator] = field(default_factory=list)
    edges: list[TemplateEdge] = field(default_factory=list)
    result_declarations: list[ResultDeclaration] = field(default_factory=list)
    legacy_projection: list[LegacyLogicalTaskProjection] = field(default_factory=list)
    effect_boundaries: list[EffectBoundary] = field(default_factory=list)
    tool_declarations: list[ToolDeclaration] = field(default_factory=list)
    resource_declarations: list[ResourceDeclaration] = field(default_factory=list)
    source_map: list[SourceMapEntry] = field(default_factory=list)
    nodes: list[PhysicalNode] = field(default_factory=list)

    @property
    def operator_ids(self) -> set[str]:
        return {op.operator_id for op in self.operators}


def build_name_map(parsed: ParsedWorkflow) -> dict[str, str]:
    """Map source-visible task names to their stable operator ids."""
    name_to_op: dict[str, str] = {}
    for task in parsed.tasks:
        if task.graph_node_name:
            name_to_op[task.graph_node_name] = task.task_id
        if task.local_name:
            name_to_op[task.local_name] = task.task_id
    return name_to_op


def lower_tasks(
    parsed: ParsedWorkflow,
    name_to_op: dict[str, str],
    acc: LoweringAccumulator,
    defaults: AgentBindingDefaults,
    secret_refs: Mapping[str, str],
) -> None:
    """Lower each legacy task into a symbolic leaf/agent/residency operator.

    Each legacy task becomes a symbolic leaf (an ``Agent`` for ``agent`` tasks; a
    residency-only leaf for ``serve``), ``dependsOn`` becomes port wiring, and each
    result-owning task induces one singleton logical-output slot. An agent bound to a
    resident model dependency also emits a plan-derived service-family requirement and
    a required residency intent. Nothing here carries worker/replica/endpoint bindings.
    """
    task_ids: set[str] = {task.task_id for task in parsed.tasks}
    # A task may depend on a region node, whose operator id is its source name.
    known_ids: set[str] = task_ids | {region.name for region in parsed.regions}

    # Pass 1: one operator + source-map entry per task.
    for task in parsed.tasks:
        task_type = task.task.spec.taskType
        if binding_class(task_type) is BindingClass.AGENT:
            acc.operators.append(
                _agent_operator(
                    task, name_to_op, task_ids, defaults, secret_refs.get(task.task_id)
                )
            )
        else:
            acc.operators.append(_leaf_operator(task, task_type, name_to_op, task_ids))
        acc.source_map.append(_source_map_entry(task))

    agent_ops = {
        op.operator_id: op for op in acc.operators if isinstance(op, AgentOperator)
    }

    # Pass 2: wiring, induced outputs, and physical nodes.
    for task in parsed.tasks:
        task_type = task.task.spec.taskType
        operator_id = task.task_id
        for dep in task.depends_on:
            if dep in known_ids:
                acc.edges.append(TemplateEdge(from_op=dep, to_op=operator_id))

        # serve administers resident capacity: a residency node, no result slot.
        if binding_class(task_type) is BindingClass.RESIDENCY:
            model_ref = _model_ref(task)
            family = model_ref.architecture if model_ref else task_type.value
            acc.nodes.append(
                PhysicalNode(
                    node_id=f"phys:{operator_id}",
                    source_ref=operator_id,
                    logical_ref=operator_id,
                    service_family_requirement=ServiceFamilyRequirement(family=family),
                    residency_intent=ResidencyIntent(warmth="warm"),
                )
            )
            continue

        output_id = f"legacy:{operator_id}"
        acc.result_declarations.append(
            ResultDeclaration(
                output_id=output_id,
                source_ref=operator_id,
                cardinality=CardinalityKind.SINGLETON,
                release=ReleaseConditionKind.SOURCE_SETTLED,
                visibility=Visibility.INTERNAL,
                value_type=task_type.value,
            )
        )
        acc.legacy_projection.append(
            LegacyLogicalTaskProjection(
                legacy_task_id=operator_id,
                operator_id=operator_id,
                induced_output_id=output_id,
                value_type=task_type.value,
                source_ref=operator_id,
            )
        )
        requirement, intent = _resident_annotations(agent_ops.get(operator_id))
        acc.nodes.append(
            PhysicalNode(
                node_id=f"phys:{operator_id}",
                source_ref=operator_id,
                logical_ref=operator_id,
                service_family_requirement=requirement,
                residency_intent=intent,
            )
        )


def _resident_annotations(
    agent: AgentOperator | None,
) -> tuple[ServiceFamilyRequirement | None, ResidencyIntent | None]:
    """Derive the plan-derived resident requirement for a resident-bound agent.

    Detection only: it pins the finite dependency a resident model binding needs, with
    no allocation, claim, or replica. The service family is derived canonically from
    the reference, so identical references pin one shared demand family; engine-batch
    and isolation policy are the residency scheduler's to set.
    """
    if agent is None or agent.model_binding is None:
        return None, None
    binding = agent.model_binding
    if binding.mode is not ModelBindingMode.RESIDENT or not binding.service_model_ref:
        return None, None
    family = service_family_for_ref(binding.service_model_ref)
    return (
        ServiceFamilyRequirement(family=family),
        ResidencyIntent(service_family=family, required=True),
    )


def induce_effect_boundaries(acc: LoweringAccumulator) -> None:
    """Declare one effect boundary per external-effect leaf, from its final profile.

    Runs after ``spec.v2`` overrides apply, so a leaf whose effect is overridden to
    or from ``external_effect`` gets its boundary set induced consistently.
    """
    for op in acc.operators:
        if (
            isinstance(op, LeafOperator)
            and not op.residency_only
            and op.profile.effect == EffectClass.EXTERNAL_EFFECT
        ):
            acc.effect_boundaries.append(
                EffectBoundary(
                    effect_class=EffectClass.EXTERNAL_EFFECT,
                    replay_contract=EffectReplayContract.AMBIGUITY_TERMINAL,
                    source_ref=op.operator_id,
                )
            )
