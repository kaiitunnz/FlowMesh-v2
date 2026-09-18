"""Bundle and engine construction for the span-emitter tests.

Builds the smallest compiled bundles that exercise the emitter's derivations: a
static two-leaf chain (a settle that publishes a slot and releases a successor) and
an agent declaring one spawn region (a scope whose extent spans N children).
"""

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from server.orchestration import OrchestrationEngine, ScopeBudget
from server.orchestration.telemetry import TelemetrySpanEmitter
from server.task.v2 import FrontendWorkflowSource, PersistedV2Workflow
from server.task.v2.compiler.bindings import leaf_profile
from server.task.v2.representations.operators import (
    AgentOperator,
    AuthorityCeiling,
    BindingKey,
    BoundaryEventKind,
    BoundarySignature,
    ChildRegionRef,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LogicalOperator,
    OperatorKind,
    Port,
    SpawnRegion,
)
from server.task.v2.representations.plan import PhysicalExecutionPlan, PhysicalNode
from server.task.v2.representations.results import (
    CardinalityKind,
    ReleaseConditionKind,
    ResultDeclaration,
    Visibility,
)
from server.task.v2.representations.template import (
    LogicalWorkflowTemplate,
    SourceMapEntry,
    TemplateEdge,
)
from server.task.v2.representations.versioning import VersionId
from shared.tasks import TaskType
from shared.telemetry.config import TelemetryConfig, TelemetryLevel

WORKFLOW_ID = "wfl-emitter-test"
GRANTED = frozenset({"model"})

_SIGNATURE = BoundarySignature(
    events=(
        BoundaryEventKind.INVOCATION,
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.SPAWN_SEAL,
    )
)
_REGION_KINDS = {
    OperatorKind.BRANCH,
    OperatorKind.MERGE,
    OperatorKind.SPAWN,
    OperatorKind.JOIN,
    OperatorKind.LOOP_CONTEXT,
}


def _leaf(op_id: str) -> LeafOperator:
    return LeafOperator(
        operator_id=op_id,
        source_ref=op_id,
        outputs=(Port(name="out"),),
        profile=leaf_profile(TaskType.ECHO),
    )


def _decl(
    output_id: str,
    source_ref: str,
    *,
    release: ReleaseConditionKind = ReleaseConditionKind.SOURCE_SETTLED,
) -> ResultDeclaration:
    return ResultDeclaration(
        output_id=output_id,
        source_ref=source_ref,
        cardinality=CardinalityKind.SINGLETON,
        release=release,
        visibility=Visibility.INTERNAL,
    )


def _bundle(
    ops: list[LogicalOperator],
    edges: list[TemplateEdge],
    results: tuple[ResultDeclaration, ...] = (),
) -> PersistedV2Workflow:
    tv = VersionId(lineage="wfl-e:template", content_digest="td")
    pv = VersionId(lineage="wfl-e:plan", content_digest="pd")
    template = LogicalWorkflowTemplate(
        version=tv,
        operators=tuple(ops),
        edges=tuple(edges),
        result_declarations=results,
        source_map=tuple(
            SourceMapEntry(
                logical_ref=op.operator_id,
                source_kind="region" if op.kind in _REGION_KINDS else "graph_node",
                source_id=op.operator_id,
            )
            for op in ops
        ),
    )
    plan = PhysicalExecutionPlan(
        plan_version=pv,
        template_version=tv,
        nodes=tuple(
            PhysicalNode(
                node_id=f"phys:{op.operator_id}",
                source_ref=op.operator_id,
                logical_ref=op.operator_id,
            )
            for op in ops
        ),
    )
    source = FrontendWorkflowSource.capture("emitter: true", "native", name="wf")
    return PersistedV2Workflow(source=source, template=template, plan=plan)


def chain_bundle() -> PersistedV2Workflow:
    """Leaf ``A`` publishing ``out:A``, with leaf ``B`` as its static successor."""
    return _bundle(
        [_leaf("A"), _leaf("B")],
        [TemplateEdge(from_op="A", to_op="B")],
        (_decl("out:A", "A"), _decl("out:B", "B")),
    )


def spawning_agent_bundle() -> PersistedV2Workflow:
    """Agent ``A`` with one ``worker`` region whose child body is leaf ``wbody``."""
    spawn = SpawnRegion(
        operator_id="worker:spawn",
        source_ref="worker:spawn",
        outputs=(Port(name="children"),),
        child_template_ref="wbody",
    )
    join = JoinRegion(
        operator_id="worker:spawn:join",
        source_ref="worker:spawn:join",
        inputs=(Port(name="children"),),
        outputs=(Port(name="out"),),
        completion=JoinCompletion.ALL_SETTLED,
    )
    agent = AgentOperator(
        operator_id="A",
        source_ref="A",
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(invoke=("model",), delegate=()),
        boundary=_SIGNATURE,
        child_region_refs=(ChildRegionRef(name="worker", spawn_ref="worker:spawn"),),
        outputs=(Port(name="out"),),
    )
    return _bundle(
        [agent, _leaf("wbody"), spawn, join],
        [TemplateEdge(from_op="worker:spawn", to_op="worker:spawn:join")],
        (_decl("out:A", "A"),),
    )


def emitter(
    level: TelemetryLevel = TelemetryLevel.FULL,
) -> tuple[TelemetrySpanEmitter, InMemorySpanExporter]:
    tracer, exporter = _tracer()
    config = TelemetryConfig(
        level=level,
        traces_enabled=level is not TelemetryLevel.OFF,
        metrics_enabled=False,
        sample_ratio=1.0,
        otlp_endpoint=None,
    )
    return TelemetrySpanEmitter(tracer, config, WORKFLOW_ID), exporter


def _tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def engine(
    bundle: PersistedV2Workflow, *, emitter: TelemetrySpanEmitter | None = None
) -> OrchestrationEngine:
    return OrchestrationEngine.build(
        WORKFLOW_ID,
        "owner",
        "org",
        bundle,
        granted_interfaces=GRANTED,
        emitter=emitter,
    )


def rehydrated(
    live: OrchestrationEngine, *, emitter: TelemetrySpanEmitter | None = None
) -> OrchestrationEngine:
    return OrchestrationEngine(
        live.to_snapshot(),
        live._bundle,  # noqa: SLF001 - the same compiled bundle the runtime holds
        budget=ScopeBudget(),
        emitter=emitter,
    )


def span_signature(span: ReadableSpan) -> tuple[object, ...]:
    return (
        span.name,
        span.context.trace_id if span.context else None,
        span.context.span_id if span.context else None,
        span.parent.span_id if span.parent is not None else None,
        span.start_time,
        span.end_time,
        tuple(sorted(dict(span.attributes or {}).items())),
    )
