from shared.tasks import TaskType
from shared.tasks.specs.common import ModelSpecTemplate

from ..parser import ParsedTask, ParsedWorkflow
from .operators import (
    AgentOperator,
    AuthorityCeiling,
    BindingKey,
    BoundaryEventKind,
    BoundarySignature,
    ConditionGuard,
    DeterminismClass,
    EffectBoundary,
    EffectClass,
    EffectReplayContract,
    EqualityRelation,
    EqualityRelationKind,
    InputProvenanceKind,
    LeafOperator,
    LeafProfile,
    LogicalOperator,
    ModelRef,
    Port,
    PortKind,
    RecoveryClass,
)
from .plan import (
    PhysicalExecutionPlan,
    PhysicalNode,
    ResidencyIntent,
    ServiceFamilyRequirement,
)
from .results import (
    CardinalityKind,
    LegacyLogicalTaskProjection,
    ReleaseConditionKind,
    ResultDeclaration,
    Visibility,
)
from .source import FrontendWorkflowSource
from .template import LogicalWorkflowTemplate, SourceMapEntry, TemplateEdge
from .versioning import VersionId, content_digest

_TRAINING_TYPES: frozenset[TaskType] = frozenset(
    {
        TaskType.SFT,
        TaskType.LORA_SFT,
        TaskType.PPO,
        TaskType.DPO,
        TaskType.IMAGE_CLASSIFICATION_TRAINING,
    }
)

_SEMANTIC = EqualityRelation(kind=EqualityRelationKind.SEMANTIC)
_BITWISE = EqualityRelation(kind=EqualityRelationKind.BITWISE)


def _leaf_profile(task_type: TaskType) -> LeafProfile:
    """Return the indicative binding profile for a legacy task type."""
    binding = BindingKey(task_type=task_type)
    det = DeterminismClass.SAMPLED
    effect = EffectClass.PURE
    recovery = RecoveryClass.RECORD
    provenance = InputProvenanceKind.EXTERNAL_PINNED
    equality: EqualityRelation | None = None

    match task_type:
        case TaskType.EMBEDDING:
            det, recovery, equality = (
                DeterminismClass.DETERMINISTIC_SEMANTIC,
                RecoveryClass.RECOMPUTE,
                _SEMANTIC,
            )
        case TaskType.RAG:
            det, recovery, equality, provenance = (
                DeterminismClass.DETERMINISTIC_SEMANTIC,
                RecoveryClass.RECORD,
                _SEMANTIC,
                InputProvenanceKind.LIVE_INPUT,
            )
        case TaskType.DATA_PROFILING:
            det, recovery, equality = (
                DeterminismClass.DETERMINISTIC_SEMANTIC,
                RecoveryClass.RECOMPUTE,
                _SEMANTIC,
            )
        case TaskType.ECHO:
            det, recovery, equality = (
                DeterminismClass.DETERMINISTIC_BITWISE,
                RecoveryClass.RECOMPUTE,
                _BITWISE,
            )
        case TaskType.DATA_RETRIEVAL:
            provenance = InputProvenanceKind.LIVE_INPUT
        case TaskType.API | TaskType.SSH | TaskType.SERVE:
            effect, provenance = (
                EffectClass.EXTERNAL_EFFECT,
                InputProvenanceKind.LIVE_INPUT,
            )
        case (
            TaskType.SFT
            | TaskType.LORA_SFT
            | TaskType.PPO
            | TaskType.DPO
            | TaskType.IMAGE_CLASSIFICATION_TRAINING
        ):
            effect = EffectClass.PRIVATE_STATE
        case _:
            pass

    return LeafProfile(
        determinism=det,
        effect=effect,
        recovery=recovery,
        input_provenance=provenance,
        binding=binding,
        output_equality=equality,
    )


def _model_ref(task: ParsedTask) -> ModelRef | None:
    spec = task.task.spec
    if not isinstance(spec, ModelSpecTemplate):
        return None
    name = spec.model_name
    if not name:
        return None
    return ModelRef(architecture=name, version=spec.model_revision)


def _condition_guard(
    task: ParsedTask, name_to_op: dict[str, str], operator_ids: set[str]
) -> ConditionGuard | None:
    condition = task.task.spec.condition
    if condition is None:
        return None
    raw = condition.node.strip()
    operator_id = name_to_op.get(raw, raw)
    if operator_id not in operator_ids:
        raise ValueError(
            f"conditional guard on task {task.task_id!r} references upstream "
            f"{condition.node!r}, which resolves to no operator."
        )
    return ConditionGuard(
        node=operator_id, field=condition.field, equals=condition.equals
    )


def _source_map_entry(task: ParsedTask) -> SourceMapEntry:
    if task.graph_node_name:
        return SourceMapEntry(
            logical_ref=task.task_id,
            source_kind="graph_node",
            source_id=task.graph_node_name,
        )
    if task.local_name:
        return SourceMapEntry(
            logical_ref=task.task_id, source_kind="stage", source_id=task.local_name
        )
    return SourceMapEntry(
        logical_ref=task.task_id, source_kind="legacy_task", source_id=task.task_id
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
        if task_type in _TRAINING_TYPES:
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
        profile=_leaf_profile(task_type),
        guard=_condition_guard(task, name_to_op, operator_ids),
        residency_only=task_type == TaskType.SERVE,
    )


def _agent_operator(
    task: ParsedTask, name_to_op: dict[str, str], operator_ids: set[str]
) -> AgentOperator:
    inputs, outputs = _ports(task, TaskType.AGENT)
    return AgentOperator(
        operator_id=task.task_id,
        source_ref=task.task_id,
        inputs=inputs,
        outputs=outputs,
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(),
        boundary=BoundarySignature(
            events=(
                BoundaryEventKind.INVOCATION,
                BoundaryEventKind.SPAWN,
                BoundaryEventKind.YIELD,
                BoundaryEventKind.EXTERNAL_EFFECT,
                BoundaryEventKind.STATE_ACCESS,
            )
        ),
        guard=_condition_guard(task, name_to_op, operator_ids),
    )


def project_acyclic(
    workflow_id: str,
    parsed: ParsedWorkflow,
    source: FrontendWorkflowSource,
) -> tuple[LogicalWorkflowTemplate, PhysicalExecutionPlan]:
    """Project an acyclic parsed workflow into symbolic v2 representations.

    Each legacy task becomes a symbolic leaf (an ``Agent`` for ``agent`` tasks; a
    residency-only leaf for ``serve``), ``dependsOn`` becomes port wiring, and
    each result-owning task induces one singleton logical-output slot. The result
    carries no worker/replica/endpoint bindings and no activation tags.
    """
    operators: list[LogicalOperator] = []
    edges: list[TemplateEdge] = []
    result_declarations: list[ResultDeclaration] = []
    legacy_projection: list[LegacyLogicalTaskProjection] = []
    effect_boundaries: list[EffectBoundary] = []
    source_map: list[SourceMapEntry] = []
    nodes: list[PhysicalNode] = []

    operator_ids: set[str] = {task.task_id for task in parsed.tasks}
    name_to_op: dict[str, str] = {}
    for task in parsed.tasks:
        if task.graph_node_name:
            name_to_op[task.graph_node_name] = task.task_id
        if task.local_name:
            name_to_op[task.local_name] = task.task_id

    for task in parsed.tasks:
        task_type = task.task.spec.taskType
        if task_type == TaskType.AGENT:
            operators.append(_agent_operator(task, name_to_op, operator_ids))
        else:
            operators.append(_leaf_operator(task, task_type, name_to_op, operator_ids))
        source_map.append(_source_map_entry(task))

    for task in parsed.tasks:
        task_type = task.task.spec.taskType
        operator_id = task.task_id
        for dep in task.depends_on:
            if dep in operator_ids:
                edges.append(TemplateEdge(from_op=dep, to_op=operator_id))

        if task_type == TaskType.SERVE:
            model_ref = _model_ref(task)
            family = model_ref.architecture if model_ref else task_type.value
            nodes.append(
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
        result_declarations.append(
            ResultDeclaration(
                output_id=output_id,
                source_ref=operator_id,
                cardinality=CardinalityKind.SINGLETON,
                release=ReleaseConditionKind.SOURCE_SETTLED,
                visibility=Visibility.INTERNAL,
                value_type=task_type.value,
            )
        )
        legacy_projection.append(
            LegacyLogicalTaskProjection(
                legacy_task_id=operator_id,
                operator_id=operator_id,
                induced_output_id=output_id,
                value_type=task_type.value,
                source_ref=operator_id,
            )
        )
        if task_type in (TaskType.API, TaskType.SSH):
            effect_boundaries.append(
                EffectBoundary(
                    effect_class=EffectClass.EXTERNAL_EFFECT,
                    replay_contract=EffectReplayContract.AMBIGUITY_TERMINAL,
                    source_ref=operator_id,
                )
            )
        nodes.append(
            PhysicalNode(
                node_id=f"phys:{operator_id}",
                source_ref=operator_id,
                logical_ref=operator_id,
            )
        )

    lineage = f"{workflow_id}:template"
    provisional = LogicalWorkflowTemplate(
        version=VersionId(lineage=lineage, content_digest=""),
        operators=tuple(operators),
        edges=tuple(edges),
        result_declarations=tuple(result_declarations),
        legacy_projection=tuple(legacy_projection),
        effect_boundaries=tuple(effect_boundaries),
        source_map=tuple(source_map),
    )
    digest = content_digest(
        source.digest + provisional.model_dump_json(exclude={"version"})
    )
    template = provisional.model_copy(
        update={"version": VersionId(lineage=lineage, content_digest=digest)}
    )
    plan = _finalize_plan(workflow_id, template.version, tuple(nodes))
    return template, plan


def _finalize_plan(
    workflow_id: str, template_version: VersionId, nodes: tuple[PhysicalNode, ...]
) -> PhysicalExecutionPlan:
    lineage = f"{workflow_id}:plan"
    provisional = PhysicalExecutionPlan(
        plan_version=VersionId(lineage=lineage, content_digest=""),
        template_version=template_version,
        nodes=nodes,
    )
    digest = content_digest(provisional.model_dump_json(exclude={"plan_version"}))
    return provisional.model_copy(
        update={"plan_version": VersionId(lineage=lineage, content_digest=digest)}
    )
