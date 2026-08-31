"""Canonical v2 agent dataflow: manifest-gated readiness, typed inputs, frozen resolve.

An agent that declares input ports is admissible only once its accepted-input manifest
is recorded; recording an input is authority-neutral. A graph fanout mints a typed
child-entry input, and a join feeding an agent yields an aggregate ordered by child idx.
The runtime resolves each member's frozen value and digest, projects it into the first
dispatch only, and fails an over-budget input as a typed declared failure rather than
truncating.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from server.config import OrchestrationConfig
from server.orchestration import (
    Advance,
    OrchestrationEngine,
    PublicationOutcome,
    WorkItemStatus,
)
from server.orchestration.state import (
    AcceptedInput,
    AcceptedInputMember,
    AuthorityDecisionKind,
    ValueRef,
)
from server.task.runtime import TaskRuntime
from server.task.v2.compiler.bindings import leaf_profile
from server.task.v2.representations.operators import (
    AgentOperator,
    AuthorityCeiling,
    BindingKey,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    Port,
    SpawnRegion,
)
from server.task.v2.representations.template import TemplateEdge
from shared.schemas.result import ResultEnvelope, result_file_path
from shared.tasks import TaskType
from tests.server.task.test_v2_agent_harness import _bundle, _decl, _engine, _leaf
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _WorkerRegistryStub,
)


def _input_agent(op_id: str, ports: tuple[str, ...]) -> AgentOperator:
    return AgentOperator(
        operator_id=op_id,
        source_ref=op_id,
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(invoke=("model",), delegate=()),
        inputs=tuple(Port(name=name) for name in ports),
        outputs=(Port(name="out"),),
        declared_input_ports=ports,
    )


def _accepted(activation: str, port: str, value_ref: ValueRef) -> AcceptedInput:
    return AcceptedInput(
        activation_id=activation,
        target_port=port,
        members=(
            AcceptedInputMember(
                source_operator_id="P",
                source_activation_id="act-P",
                outcome=PublicationOutcome.SUCCESS,
                value_ref=value_ref,
            ),
        ),
    )


def test_declared_input_agent_blocks_until_its_manifest_is_recorded() -> None:
    producer = _leaf("P")
    merge = _input_agent("M", ("reviews",))
    edge = TemplateEdge(from_op="P", to_op="M", to_port="reviews")
    engine = _engine(
        _bundle([producer, merge], [edge], (_decl("out:M", "M"),)),
        granted=frozenset({"model"}),
    )
    engine.on_succeeded("P")  # the producer settles and delivers its record
    wi = engine.work_item("M")
    assert (
        wi is not None and wi.status is WorkItemStatus.BLOCKED
    )  # manifest unsatisfied

    decisions_before = len(engine.to_snapshot().authority_decisions)
    engine.record_accepted_input(
        _accepted(wi.activation_id, "reviews", ValueRef(kind="inline", literal="x"))
    )
    # Recording an input widens no authority: it delivers data, not an invoke right.
    added = engine.to_snapshot().authority_decisions[decisions_before:]
    assert all(d.kind is not AuthorityDecisionKind.DENIED for d in added)

    advance = engine.reconsider_admission("M")
    assert advance.ready == ["M"]
    assert engine.work_item("M").status is WorkItemStatus.READY


def test_join_feeding_an_agent_aggregates_children_in_declared_order() -> None:
    producer = _leaf("P")
    child = LeafOperator(
        operator_id="R",
        source_ref="R",
        outputs=(Port(name="out"),),
        profile=leaf_profile(TaskType.ECHO),
    )
    spawn = SpawnRegion(
        operator_id="S",
        source_ref="S",
        outputs=(Port(name="children"),),
        child_template_ref="R",
    )
    join = JoinRegion(
        operator_id="J",
        source_ref="J",
        inputs=(Port(name="children"),),
        outputs=(Port(name="out"),),
        completion=JoinCompletion.ALL_SETTLED,
    )
    merge = _input_agent("M", ("reviews",))
    edges = [
        TemplateEdge(from_op="P", to_op="S"),
        TemplateEdge(from_op="S", to_op="J"),
        TemplateEdge(from_op="J", to_op="M", to_port="reviews"),
    ]
    engine = _engine(
        _bundle([producer, child, spawn, join, merge], edges, (_decl("out:M", "M"),)),
        granted=frozenset({"model"}),
    )
    engine.on_succeeded("P")  # fires the spawn, opening its child-init scope
    kids = [engine.materialize_child("S").ready[0] for _ in range(3)]
    engine.seal_spawn("S")
    # Settle the children out of order; the aggregate still orders by child index.
    for task_id in (kids[2], kids[0], kids[1]):
        engine.on_succeeded(task_id)

    plan = engine.agent_input_plan("M")
    assert plan is not None
    (port,) = plan.ports
    assert port.target_port == "reviews" and port.provenance == "join_aggregate"
    assert [m.child_index for m in port.members] == [0, 1, 2]
    assert [m.ordinal for m in port.members] == [0, 1, 2]
    assert [m.legacy_task_id for m in port.members] == kids


def test_spawn_mints_a_typed_child_entry_input_not_spec_data() -> None:
    producer = _leaf("P")
    reviewer = _input_agent("R", ("facet",))
    spawn = SpawnRegion(
        operator_id="S",
        source_ref="S",
        outputs=(Port(name="children"),),
        child_template_ref="R",
    )
    join = JoinRegion(
        operator_id="J",
        source_ref="J",
        inputs=(Port(name="children"),),
        outputs=(Port(name="out"),),
        completion=JoinCompletion.ALL_SETTLED,
    )
    edges = [TemplateEdge(from_op="P", to_op="S"), TemplateEdge(from_op="S", to_op="J")]
    engine = _engine(
        _bundle([producer, reviewer, spawn, join], edges, (_decl("out:J", "J"),)),
        granted=frozenset({"model"}),
    )
    engine.on_succeeded("P")
    child = engine.materialize_child(
        "S", value_ref=ValueRef(kind="inline", literal='{"facet": "retrieval methods"}')
    ).ready[0]
    (accepted,) = engine.accepted_inputs_for_task(child)
    assert accepted.target_port == "facet" and accepted.provenance == "spawn_element"
    assert accepted.members[0].value_ref.literal == '{"facet": "retrieval methods"}'


def _fanout_bundle(
    completion: JoinCompletion = JoinCompletion.ALL_SETTLED,
    first_k: int | None = None,
    residual: str | None = None,
):
    child = LeafOperator(
        operator_id="R",
        source_ref="R",
        outputs=(Port(name="out"),),
        profile=leaf_profile(TaskType.ECHO),
    )
    spawn = SpawnRegion(
        operator_id="S",
        source_ref="S",
        outputs=(Port(name="children"),),
        child_template_ref="R",
    )
    join = JoinRegion(
        operator_id="J",
        source_ref="J",
        inputs=(Port(name="children"),),
        outputs=(Port(name="out"),),
        completion=completion,
        first_k=first_k,
        residual_policy=residual,
    )
    edges = [
        TemplateEdge(from_op="P", to_op="S"),
        TemplateEdge(from_op="S", to_op="J"),
        TemplateEdge(from_op="J", to_op="M", to_port="reviews"),
    ]
    return _bundle(
        [_leaf("P"), child, spawn, join, _input_agent("M", ("reviews",))],
        edges,
        (_decl("out:M", "M"),),
    )


def _member_indices(engine: OrchestrationEngine, task_id: str) -> list[int | None]:
    plan = engine.agent_input_plan(task_id)
    assert plan is not None
    return [m.child_index for m in plan.ports[0].members]


def _member_pairs(
    engine: OrchestrationEngine, task_id: str
) -> list[tuple[int | None, str | None]]:
    plan = engine.agent_input_plan(task_id)
    assert plan is not None
    return [(m.child_index, m.legacy_task_id) for m in plan.ports[0].members]


def test_region_aggregate_replays_identically_after_restart() -> None:
    bundle = _fanout_bundle()
    engine = _engine(bundle, granted=frozenset({"model"}))
    engine.on_succeeded("P")
    kids = [engine.materialize_child("S").ready[0] for _ in range(3)]
    engine.seal_spawn("S")
    for task_id in (kids[2], kids[0], kids[1]):
        engine.on_succeeded(task_id)
    before = _member_pairs(engine, "M")
    # A restart rebuilds the engine from the durable snapshot; the frozen aggregate
    # replays identically — same members, same order, no re-derivation.
    restored = OrchestrationEngine(engine.to_snapshot(), bundle)
    after = _member_pairs(restored, "M")
    assert before == after == [(0, kids[0]), (1, kids[1]), (2, kids[2])]


def test_early_join_freezes_membership_against_a_residual_child() -> None:
    bundle = _fanout_bundle(
        completion=JoinCompletion.FIRST_K, first_k=2, residual="continue"
    )
    engine = _engine(bundle, granted=frozenset({"model"}))
    engine.on_succeeded("P")
    kids = [engine.materialize_child("S").ready[0] for _ in range(3)]
    engine.on_succeeded(kids[0])
    engine.on_succeeded(kids[1])  # first_k=2 met -> early release freezes membership
    frozen = _member_indices(engine, "M")
    engine.on_succeeded(kids[2])  # a residual child settling later must not mutate it
    assert frozen == _member_indices(engine, "M") == [0, 1]


_REGION_OUTPUT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: region-output}
spec:
  graph:
    nodes:
      - name: lead
        spec:
          taskType: agent
          task: spawn reviewers
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            boundary: [spawn, spawn_seal, yield]
            child: [{name: reviewer, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: reviewer
        spec:
          taskType: agent
          task: research the facet
          v2:
            inputs: [facet]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: merge
        spec:
          taskType: agent
          task: merge the reviews
          v2:
            inputs: [{name: reviews, from: lead, region: reviewer}]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def test_region_output_binds_the_child_region_join_to_the_merge_input() -> None:
    from server.task.parser import parse_workflow
    from server.task.v2 import FrontendWorkflowSource, compile_workflow
    from server.task.v2.compiler.agent_binding import AgentBindingDefaults
    from server.task.v2.representations.operators import JoinRegion as _Join
    from shared.tasks.specs import ModelBindingMode

    binding = AgentBindingDefaults(
        default_backend="scripted",
        default_version="v1",
        default_mode=ModelBindingMode.OPENAI,
        default_url="http://x/v1",
        default_model="m",
    )
    parsed = parse_workflow(_REGION_OUTPUT_WF, "native")
    source = FrontendWorkflowSource.capture(_REGION_OUTPUT_WF, "native", name="wf")
    template, _ = compile_workflow("wfl-x", parsed, source, bindings=binding)
    by_id = {op.operator_id: op for op in template.operators}
    reviews_edges = [
        e for e in template.edges if e.to_port == "reviews" and not e.feedback
    ]
    assert len(reviews_edges) == 1
    # The merge's input is delivered by the lead's reviewer child-region join aggregate.
    assert isinstance(by_id.get(reviews_edges[0].from_op), _Join)
    assert reviews_edges[0].from_op.endswith(":reviewer:spawn:join")


def _runtime(budget: int | None = None) -> TaskRuntime:
    config = OrchestrationConfig()
    if budget is not None:
        config.agent_input_budget_bytes = budget
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerRegistryStub()),
        config,
        Path(tempfile.mkdtemp()),
        logging.getLogger("dataflow-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def _write_result(runtime: TaskRuntime, task_id: str, payload: dict[str, Any]) -> None:
    path = result_file_path(runtime._results_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ResultEnvelope.model_validate(
            {"task_id": task_id, "task_type": "echo", "result": payload}
        ).model_dump_json(),
        "utf-8",
    )


def test_frozen_resolver_reads_inline_element_and_producer_value() -> None:
    runtime = _runtime()
    assert (
        runtime._resolve_value_ref(ValueRef(kind="inline", literal="the facet"))
        == "the facet"
    )
    _write_result(
        runtime,
        "tsk-p",
        {"taskType": "echo", "items": [{"output": "facet-a"}, "facet-b"]},
    )
    assert (
        runtime._resolve_value_ref(
            ValueRef(
                kind="legacy_task_result", legacy_task_id="tsk-p", collection_key="0"
            )
        )
        == "facet-a"
    )
    assert (
        runtime._resolve_value_ref(
            ValueRef(
                kind="legacy_task_result", legacy_task_id="tsk-p", collection_key="1"
            )
        )
        == "facet-b"
    )
    _write_result(runtime, "tsk-r", {"taskType": "agent", "value": "grounded findings"})
    assert (
        runtime._resolve_value_ref(
            ValueRef(kind="legacy_task_result", legacy_task_id="tsk-r")
        )
        == "grounded findings"
    )
    # A not-yet-readable result defers rather than fabricating a value.
    assert (
        runtime._resolve_value_ref(
            ValueRef(kind="legacy_task_result", legacy_task_id="tsk-missing")
        )
        is None
    )


def test_input_bindings_projection_is_deterministic() -> None:
    runtime = _runtime()
    merge = _input_agent("M", ("reviews",))
    engine = _engine(
        _bundle(
            [_leaf("P"), merge],
            [TemplateEdge(from_op="P", to_op="M", to_port="reviews")],
            (_decl("out:M", "M"),),
        ),
        granted=frozenset({"model"}),
    )
    activation = engine.work_item("M").activation_id
    engine.record_accepted_input(
        _accepted(activation, "reviews", ValueRef(kind="inline", literal="grounded")),
    )
    first = runtime._agent_input_bindings(engine, "M")
    second = runtime._agent_input_bindings(engine, "M")
    assert first == second  # stable projection over the durable manifest
    assert first[0].port == "reviews" and first[0].members[0].value == "grounded"


def test_oversized_input_fails_the_agent_rather_than_truncating() -> None:
    runtime = _runtime(budget=64)
    merge = _input_agent("M", ("reviews",))
    engine = _engine(
        _bundle(
            [_leaf("P"), merge],
            [TemplateEdge(from_op="P", to_op="M", to_port="reviews")],
            (_decl("out:M", "M"),),
        ),
        granted=frozenset({"model"}),
    )
    engine.on_succeeded("P")
    _write_result(runtime, "P", {"taskType": "agent", "value": "x" * 5000})
    runtime._resolve_agent_inputs_locked(engine, Advance())
    wi = engine.work_item("M")
    assert wi.status in (WorkItemStatus.SETTLED, WorkItemStatus.CANCELLED)
    assert wi.outcome is PublicationOutcome.DECLARED_FAILURE
