from pydantic import ValidationError

from ...parser import ParsedWorkflow
from ..mode import LoweringStrategy
from ..representations.bundle import PersistedV2Workflow
from ..representations.plan import PhysicalExecutionPlan, PhysicalNode
from ..representations.source import FrontendWorkflowSource
from ..representations.template import LogicalWorkflowTemplate
from ..representations.versioning import VersionId, content_digest
from .diagnostics import CompileError, Diagnostic
from .episodes import lower_to_episodes
from .project import (
    LoweringAccumulator,
    build_name_map,
    induce_effect_boundaries,
    lower_tasks,
)
from .regions import lower_frontend_v2
from .validation import has_errors, validate_compilation


def _assemble_template(
    workflow_id: str, source: FrontendWorkflowSource, acc: LoweringAccumulator
) -> LogicalWorkflowTemplate:
    lineage = f"{workflow_id}:template"
    try:
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
    except ValidationError as exc:
        raise CompileError(
            (
                Diagnostic(
                    code="template.malformed",
                    message="; ".join(e["msg"] for e in exc.errors()),
                ),
            )
        ) from exc
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
    strategy: LoweringStrategy = LoweringStrategy.TRANSPARENT,
) -> tuple[LogicalWorkflowTemplate, PhysicalExecutionPlan]:
    """Compile a parsed workflow into symbolic v2 representations.

    Lowers legacy tasks (and, when present, structured regions) into the acyclic
    subset of a logical template, then builds a physical plan under ``strategy``: the
    transparent lowering mints one boundary per legacy task/executor, while the
    episode-cut lowering rewrites that into run-to-yield episodes. Raises
    :class:`CompileError` when any error-severity diagnostic is produced. The result
    carries no worker/replica/endpoint bindings and no activation tags.
    """
    acc = LoweringAccumulator()
    name_to_op = build_name_map(parsed)
    lower_tasks(parsed, name_to_op, acc)
    lower_frontend_v2(parsed, acc)
    induce_effect_boundaries(acc)
    template = _assemble_template(workflow_id, source, acc)
    nodes = tuple(acc.nodes)
    if strategy is LoweringStrategy.EPISODE_CUT:
        nodes = lower_to_episodes(template, nodes)
    plan = _finalize_plan(workflow_id, template.version, nodes)
    if validate:
        diagnostics = validate_compilation(template, plan)
        if has_errors(diagnostics):
            raise CompileError(tuple(diagnostics))
    return template, plan


def compile_bundle(
    workflow_id: str,
    parsed: ParsedWorkflow,
    source: FrontendWorkflowSource,
    strategy: LoweringStrategy = LoweringStrategy.TRANSPARENT,
) -> PersistedV2Workflow:
    """Compile a parsed workflow into the durable plan-time bundle."""
    template, plan = compile_workflow(workflow_id, parsed, source, strategy=strategy)
    return PersistedV2Workflow(source=source, template=template, plan=plan)
