from ..parser import ParsedWorkflow
from .bundle import PersistedV2Workflow
from .diagnostics import CompileError
from .plan import PhysicalExecutionPlan, PhysicalNode
from .project import LoweringAccumulator, build_name_map, lower_tasks
from .source import FrontendWorkflowSource
from .template import LogicalWorkflowTemplate
from .validation import has_errors, validate_compilation
from .versioning import VersionId, content_digest


def _assemble_template(
    workflow_id: str, source: FrontendWorkflowSource, acc: LoweringAccumulator
) -> LogicalWorkflowTemplate:
    lineage = f"{workflow_id}:template"
    provisional = LogicalWorkflowTemplate(
        version=VersionId(lineage=lineage, content_digest=""),
        operators=tuple(acc.operators),
        edges=tuple(acc.edges),
        tool_declarations=tuple(acc.tool_declarations),
        resource_declarations=tuple(acc.resource_declarations),
        result_declarations=tuple(acc.result_declarations),
        legacy_projection=tuple(acc.legacy_projection),
        effect_boundaries=tuple(acc.effect_boundaries),
        source_map=tuple(acc.source_map),
    )
    digest = content_digest(
        source.digest + provisional.model_dump_json(exclude={"version"})
    )
    return provisional.model_copy(
        update={"version": VersionId(lineage=lineage, content_digest=digest)}
    )


def _finalize_plan(
    workflow_id: str,
    template_version: VersionId,
    nodes: tuple[PhysicalNode, ...],
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


def compile_workflow(
    workflow_id: str,
    parsed: ParsedWorkflow,
    source: FrontendWorkflowSource,
    validate: bool = True,
) -> tuple[LogicalWorkflowTemplate, PhysicalExecutionPlan]:
    """Compile a parsed workflow into symbolic v2 representations.

    Lowers legacy tasks (and, when present, structured regions) into the acyclic
    subset of a logical template, builds a transparent compatibility physical plan
    with one boundary per legacy task/executor boundary, and runs the validation
    passes. Raises :class:`CompileError` when any error-severity diagnostic is
    produced. The result carries no worker/replica/endpoint bindings and no
    activation tags.
    """
    acc = LoweringAccumulator()
    name_to_op = build_name_map(parsed)
    lower_tasks(parsed, name_to_op, acc)
    template = _assemble_template(workflow_id, source, acc)
    plan = _finalize_plan(workflow_id, template.version, tuple(acc.nodes))
    if validate:
        diagnostics = validate_compilation(template, plan)
        if has_errors(diagnostics):
            raise CompileError(tuple(diagnostics))
    return template, plan


def compile_bundle(
    workflow_id: str,
    parsed: ParsedWorkflow,
    source: FrontendWorkflowSource,
) -> PersistedV2Workflow:
    """Compile a parsed workflow into the durable plan-time bundle."""
    template, plan = compile_workflow(workflow_id, parsed, source)
    return PersistedV2Workflow(source=source, template=template, plan=plan)


def project_acyclic(
    workflow_id: str,
    parsed: ParsedWorkflow,
    source: FrontendWorkflowSource,
) -> tuple[LogicalWorkflowTemplate, PhysicalExecutionPlan]:
    """Compile a parsed workflow into the logical template and physical plan."""
    return compile_workflow(workflow_id, parsed, source)
