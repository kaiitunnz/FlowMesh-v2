"""Versioned v2 plan-time representations and the compatibility gate."""

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from server.task.v2 import (
    AgentOperator,
    BoundaryEventKind,
    ExecutionMode,
    FrontendWorkflowSource,
    LeafOperator,
    LogicalWorkflowTemplate,
    PersistedV2Workflow,
    PortKind,
    ResultDeclaration,
    VersionId,
    project_acyclic,
)

DAG_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata:
  name: dag
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source: { type: huggingface, identifier: org/model-a, revision: r1 }
          data: { type: list, items: ["hi"] }
      - name: b
        dependsOn: [a]
        spec:
          taskType: echo
          data: { type: list, items: ["x"] }
"""

DAG_V1 = DAG_V2.replace("flowmesh/v2", "flowmesh/v1")

AGENT_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata:
  name: agent
spec:
  taskType: agent
  agent: {}
  data: { type: list, items: ["go"] }
"""


class _CapturingRegistry:
    """Records what runtime.register persists, without a real Redis."""

    def __init__(self) -> None:
        self.v2: dict[str, PersistedV2Workflow | None] = {}

    async def register_workflow_async(
        self, workflow_id: str, tasks: list[Any], v2: PersistedV2Workflow | None = None
    ) -> None:
        self.v2[workflow_id] = v2

    async def save_task_states_async(self, items: list[Any]) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        return None


class _WorkerRegistryStub:
    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(id=worker_id, node_id="nde-1")

    def publish_interrupt(self, *args: Any) -> int:
        return 0


def _runtime(registry: _CapturingRegistry) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("v2-test"),
    )


def _project(payload: str) -> PersistedV2Workflow:
    parsed = parse_workflow(payload, "native")
    source = FrontendWorkflowSource.capture(payload, "native", name="wf")
    template, plan = project_acyclic("wfl-test", parsed, source)
    return PersistedV2Workflow(source=source, template=template, plan=plan)


# --------------------------------------------------------------------------- #
# Mode gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("api_version", "expected"),
    [
        ("flowmesh/v2", ExecutionMode.V2),
        ("flowmesh/v1", ExecutionMode.V1),
        ("mloc/v1", ExecutionMode.V1),
        (None, ExecutionMode.V1),
        ("", ExecutionMode.V1),
    ],
)
def test_execution_mode_from_api_version(
    api_version: str | None, expected: ExecutionMode
) -> None:
    assert ExecutionMode.from_api_version(api_version) is expected


@pytest.mark.anyio
async def test_v2_gate_off_by_default() -> None:
    registry = _CapturingRegistry()
    runtime = _runtime(registry)
    workflow_id, _ = await runtime.register("owner", "org", DAG_V1, format="native")
    assert registry.v2[workflow_id] is None


@pytest.mark.anyio
async def test_v2_gate_builds_representations() -> None:
    registry = _CapturingRegistry()
    runtime = _runtime(registry)
    workflow_id, _ = await runtime.register("owner", "org", DAG_V2, format="native")
    bundle = registry.v2[workflow_id]
    assert bundle is not None
    assert bundle.source.raw_payload == DAG_V2
    assert len(bundle.template.operators) == 2
    assert bundle.plan.template_version == bundle.template.version


# --------------------------------------------------------------------------- #
# Versioning
# --------------------------------------------------------------------------- #


def test_version_id_is_frozen() -> None:
    version = VersionId(lineage="wfl:template", content_digest="abc")
    with pytest.raises(Exception):
        version.content_digest = "mutated"  # type: ignore[misc]


def test_version_successor_rules() -> None:
    v1 = VersionId(lineage="wfl:template", revision=1, content_digest="a")
    v2 = v1.next_revision("b")
    assert v2.revision == 2
    assert v1.is_compatible_successor(v2)
    assert not v2.is_compatible_successor(v1)
    other = VersionId(lineage="other:template", revision=2, content_digest="b")
    assert not v1.is_compatible_successor(other)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def test_project_acyclic_dag_shape() -> None:
    bundle = _project(DAG_V2)
    kinds = [op.kind.value for op in bundle.template.operators]
    assert kinds == ["leaf", "leaf"]
    assert len(bundle.template.edges) == 1
    assert len(bundle.template.result_declarations) == 2
    assert len(bundle.template.legacy_projection) == 2
    assert len(bundle.plan.nodes) == 2
    # A model-carrying inference leaf references its ModelRef on a typed port.
    leaf_a = next(
        op
        for op in bundle.template.operators
        if isinstance(op, LeafOperator) and op.profile.binding.task_type == "inference"
    )
    model_ports = [p for p in leaf_a.inputs if p.kind is PortKind.MODEL_REF]
    assert model_ports and model_ports[0].model_ref is not None
    assert model_ports[0].model_ref.architecture == "org/model-a"
    assert model_ports[0].model_ref.version == "r1"


def test_project_agent_carries_authority_and_boundary() -> None:
    bundle = _project(AGENT_V2)
    (agent,) = bundle.template.operators
    assert isinstance(agent, AgentOperator)
    assert BoundaryEventKind.SPAWN in agent.boundary.events
    assert BoundaryEventKind.INVOCATION in agent.boundary.events
    assert agent.authority.invoke == ()
    # Agent is result-owning: it induces one logical output slot.
    assert len(bundle.template.legacy_projection) == 1


def test_serve_is_residency_not_result_owning() -> None:
    payload = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: { name: serve }
spec:
  taskType: serve
  resources:
    hardware: { gpu: { type: any, count: 1 } }
  model:
    source: { type: huggingface, identifier: org/served }
"""
    bundle = _project(payload)
    (leaf,) = bundle.template.operators
    assert isinstance(leaf, LeafOperator) and leaf.residency_only
    # No induced result slot for a residency binding.
    assert bundle.template.result_declarations == ()
    assert bundle.template.legacy_projection == ()
    # The physical node carries the residency intent hook.
    (node,) = bundle.plan.nodes
    assert node.residency_intent is not None
    assert node.service_family_requirement is not None


def test_conditional_recorded_as_guard() -> None:
    payload = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: { name: cond }
spec:
  graph:
    nodes:
      - name: up
        spec: { taskType: echo, data: { type: list, items: ["skip-me"] } }
      - name: gated
        dependsOn: [up]
        spec:
          taskType: echo
          condition: { node: up, field: items.0.output, equals: run-me }
          data: { type: list, items: ["x"] }
"""
    bundle = _project(payload)
    guarded = [
        op
        for op in bundle.template.operators
        if isinstance(op, LeafOperator) and op.guard is not None
    ]
    assert len(guarded) == 1
    assert guarded[0].guard is not None
    assert guarded[0].guard.equals == "run-me"


# --------------------------------------------------------------------------- #
# Symbolic invariant + round-trip + ownership
# --------------------------------------------------------------------------- #


def test_representations_are_symbolic() -> None:
    bundle = _project(DAG_V2)
    for blob in (
        bundle.template.model_dump_json(),
        bundle.plan.model_dump_json(),
    ):
        lowered = blob.lower()
        for forbidden in ("worker_id", "replica", "endpoint", "activation", "attempt"):
            assert forbidden not in lowered


def test_persisted_bundle_round_trips() -> None:
    bundle = _project(DAG_V2)
    restored = PersistedV2Workflow.model_validate_json(bundle.model_dump_json())
    assert restored == bundle
    assert restored.template.version == bundle.template.version
    assert restored.source.digest == bundle.source.digest


def test_invalid_ownership_link_rejected() -> None:
    version = VersionId(lineage="wfl:template", content_digest="a")
    dangling = ResultDeclaration(output_id="o1", source_ref="missing-op")
    with pytest.raises(ValueError, match="unknown operator"):
        LogicalWorkflowTemplate(version=version, result_declarations=(dangling,))
