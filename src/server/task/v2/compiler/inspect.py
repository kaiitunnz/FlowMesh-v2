from pydantic import BaseModel, ConfigDict

from ...parser import ParsedWorkflow
from ..representations.plan import PhysicalExecutionPlan
from ..representations.source import FrontendWorkflowSource
from ..representations.template import LogicalWorkflowTemplate
from .diagnostics import Diagnostic, Severity
from .pipeline import compile_workflow
from .validation import validate_compilation


class InspectionReport(BaseModel):
    """A dry-run view of a compiled v2 workflow before runtime submission.

    Carries the compiled logical template and physical plan plus any validation
    diagnostics, so an author or operator can see the legal structure without
    executing it.
    """

    model_config = ConfigDict(frozen=True)

    workflow_id: str
    template: LogicalWorkflowTemplate
    plan: PhysicalExecutionPlan
    diagnostics: tuple[Diagnostic, ...] = ()
    region_bearing: bool = False

    @property
    def ok(self) -> bool:
        return not any(diag.severity is Severity.ERROR for diag in self.diagnostics)

    def render_text(self) -> str:
        """Render a compact human-readable summary of the compiled template."""
        lines: list[str] = [f"workflow {self.workflow_id}"]
        lines.append(f"  template {self.template.version.lineage}")
        lines.append("  operators:")
        for op in self.template.operators:
            ports = "".join(
                f" {p.name}:{p.kind.value}" for p in (*op.inputs, *op.outputs)
            )
            lines.append(f"    {op.operator_id} [{op.kind.value}]{ports}")
        if self.template.edges:
            lines.append("  edges:")
            for edge in self.template.edges:
                arrow = "==>" if edge.feedback else "-->"
                lines.append(f"    {edge.from_op} {arrow} {edge.to_op}")
        if self.template.tool_declarations:
            names = ", ".join(t.name for t in self.template.tool_declarations)
            lines.append(f"  tools: {names}")
        if self.template.result_declarations:
            lines.append("  results:")
            for decl in self.template.result_declarations:
                lines.append(
                    f"    {decl.output_id} [{decl.cardinality.value}/"
                    f"{decl.visibility.value}]"
                )
        lines.append(f"  physical nodes: {len(self.plan.nodes)}")
        if self.diagnostics:
            lines.append("  diagnostics:")
            for diag in self.diagnostics:
                lines.append(f"    {diag.severity.value}: {diag.render()}")
        if self.region_bearing:
            lines.append(
                "  note: structured regions are inspect-only via this endpoint"
            )
        return "\n".join(lines)


def build_inspection(
    workflow_id: str,
    parsed: ParsedWorkflow,
    source: FrontendWorkflowSource,
) -> InspectionReport:
    """Compile a parsed workflow into an inspection report.

    Structural frontend errors raise :class:`CompileError`; semantic validation
    findings are returned as diagnostics on the report rather than raised.
    """
    template, plan = compile_workflow(workflow_id, parsed, source, validate=False)
    diagnostics = validate_compilation(template, plan)
    return InspectionReport(
        workflow_id=workflow_id,
        template=template,
        plan=plan,
        diagnostics=tuple(diagnostics),
        region_bearing=bool(parsed.regions),
    )
