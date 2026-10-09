from pydantic import BaseModel, ConfigDict

from ...parser import ParsedWorkflow
from ..mode import LoweringStrategy
from ..policy.lowering import PolicySurface
from ..representations.plan import PhysicalExecutionPlan
from ..representations.source import FrontendWorkflowSource
from ..representations.template import LogicalWorkflowTemplate
from .agent_binding import AgentBindingDefaults, neutral_defaults
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
                source = _endpoint(edge.from_op, edge.from_port)
                target = _endpoint(edge.to_op, edge.to_port)
                lines.append(f"    {source} {arrow} {target} [{edge.use.value}]")
        for definition in self.template.definitions:
            lines.append(
                f"  template {definition.definition_id} [{definition.kind.value}]"
            )
            lines.append(f"    members: {', '.join(definition.members)}")
            for port in definition.inputs:
                lines.append(f"    input {port.name} [{port.role.value}]")
            for entry in definition.entries:
                target = _endpoint(entry.to_op, entry.to_port)
                lines.append(f"    $ingress.{entry.port} --> {target}")
            for binding in definition.return_bindings:
                source = _endpoint(binding.from_op, binding.from_port)
                lines.append(f"    {source} --> ${binding.kind.value}.{binding.port}")
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
        if (lowering := self.plan.lowering) is not None:
            lines.append(
                f"  lowering: {lowering.strategy}"
                f" fusion={lowering.fusion}"
                f" residency={lowering.residency}"
                f" service_family={lowering.service_family}"
            )
        if self.diagnostics:
            lines.append("  diagnostics:")
            for diag in self.diagnostics:
                lines.append(f"    {diag.severity.value}: {diag.render()}")
        if self.region_bearing:
            lines.append(
                "  note: structured regions are inspect-only via this endpoint"
            )
        return "\n".join(lines)


def _endpoint(operator_id: str, port: str | None) -> str:
    return f"{operator_id}.{port}" if port else operator_id


def build_inspection(
    workflow_id: str,
    parsed: ParsedWorkflow,
    source: FrontendWorkflowSource,
    bindings: AgentBindingDefaults | None = None,
    strategy: LoweringStrategy = LoweringStrategy.TRANSPARENT,
    surface: PolicySurface | None = None,
) -> InspectionReport:
    """Compile a parsed workflow into an inspection report.

    Structural frontend errors raise :class:`CompileError`; semantic validation
    findings, including agent-binding resolution failures, are returned as
    diagnostics on the report rather than raised. ``bindings``, ``strategy``, and the
    policy ``surface`` must match the deployment defaults a real submission uses so a
    dry-run agrees with it.
    """
    defaults = bindings if bindings is not None else neutral_defaults()
    template, plan = compile_workflow(
        workflow_id,
        parsed,
        source,
        validate=False,
        strategy=strategy,
        bindings=defaults,
        surface=surface,
    )
    diagnostics = validate_compilation(template, plan, defaults.sandbox_egress_enabled)
    return InspectionReport(
        workflow_id=workflow_id,
        template=template,
        plan=plan,
        diagnostics=tuple(diagnostics),
        region_bearing=bool(parsed.regions),
    )
