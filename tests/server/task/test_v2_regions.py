import json
import logging
from typing import Any, cast

import pytest

from server.config import AgentBindingConfig, OrchestrationConfig
from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from server.task.v2 import (
    CompileError,
    FrontendWorkflowSource,
    PersistedV2Workflow,
    build_inspection,
    compile_workflow,
)
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _live_runtime,
    _NoopSecretVault,
)

REGIONS_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata:
  name: autoresearch
spec:
  graph:
    nodes:
      - name: plan
        spec:
          taskType: agent
          task: draft a research plan
          v2:
            authority: {invoke: [search/v1], delegate: []}
            tools: [{name: web_search, interface: search/v1}]
            boundary: [invocation, external_effect, yield]
      - name: deep
        dependsOn: [plan]
        spec: {taskType: agent, task: deep dive}
      - name: quick
        dependsOn: [plan]
        spec: {taskType: echo, data: {type: list, items: [quick]}}
      - name: brief
        dependsOn: [deep, quick]
        region: {kind: merge, combination: concat}
      - name: verify_child
        spec: {taskType: echo, data: {type: list, items: [verdict]}}
      - name: search_child
        spec: {taskType: echo, data: {type: list, items: [hit]}}
      - name: verify
        dependsOn: [brief]
        region: {kind: call, child: verify_child, returns: [verdict]}
      - name: fanout
        dependsOn: [verify]
        region:
          kind: spawn
          child: search_child
          authority: {invoke: [search/v1], delegate: [search/v1]}
      - name: collect
        dependsOn: [fanout]
        region: {kind: join, completion: all_settled, residual: cancel}
      - name: report
        dependsOn: [collect]
        spec:
          taskType: echo
          data: {type: list, items: [done]}
          v2: {result: {visibility: published, cardinality: singleton}}
"""


# A deployment default harness so a bare agent in a fixture resolves rather than
# failing binding validation; these tests exercise region lowering, not bindings.
_BINDINGS = AgentBindingDefaults(default_backend="codex")


def _compile(text: str) -> Any:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, _ = compile_workflow("wfl-test", parsed, source, bindings=_BINDINGS)
    return template


def test_inspection_fixture_demonstrates_all_shapes() -> None:
    template = _compile(REGIONS_WF)
    kinds = {op.kind.value for op in template.operators}
    for shape in ("merge", "spawn", "join", "agent", "leaf"):
        assert shape in kinds, shape


def test_call_normalizes_to_spawn_then_join() -> None:
    template = _compile(REGIONS_WF)
    ids = {op.operator_id: op for op in template.operators}
    assert ids["verify"].kind.value == "spawn"
    assert ids["verify:join"].kind.value == "join"
    assert any(
        e.from_op == "verify" and e.to_op == "verify:join" for e in template.edges
    )


def test_tool_interface_and_published_result() -> None:
    template = _compile(REGIONS_WF)
    assert any(t.name == "web_search" for t in template.tool_declarations)
    published = [
        d for d in template.result_declarations if d.visibility.value == "published"
    ]
    assert len(published) == 1


def test_inspection_report_renders_without_executing() -> None:
    parsed = parse_workflow(REGIONS_WF, "native")
    source = FrontendWorkflowSource.capture(REGIONS_WF, "native", name="wf")
    report = build_inspection("wfl-x", parsed, source, bindings=_BINDINGS)
    assert report.ok
    assert report.region_bearing
    text = report.render_text()
    assert "merge" in text and "spawn" in text


# --------------------------------------------------------------------------- #
# Submit vs inspect + v1 parser guard
# --------------------------------------------------------------------------- #


class _CapturingRegistry:
    def __init__(self) -> None:
        self.v2: dict[str, PersistedV2Workflow | None] = {}

    async def register_workflow_async(
        self,
        workflow_id: str,
        tasks: list[Any],
        v2: Any = None,
        submitted_at: str | None = None,
    ) -> None:
        self.v2[workflow_id] = v2

    async def save_task_states_async(self, items: list[Any]) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        return None

    def save_ledger_snapshot(self, workflow_id: str, snapshot: Any) -> None:
        return None


def _runtime() -> TaskRuntime:
    from types import SimpleNamespace

    worker_stub = SimpleNamespace(
        get_worker=lambda wid: SimpleNamespace(id=wid, node_id="nde-1"),
        publish_interrupt=lambda *a: 0,
    )
    return TaskRuntime(
        cast(Any, _CapturingRegistry()),
        cast(Any, worker_stub),
        OrchestrationConfig(agent_binding=AgentBindingConfig(default_backend="codex")),
        make_result_reader(),
        logging.getLogger("v2-regions-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


@pytest.mark.anyio
async def test_region_bearing_submit_is_admitted() -> None:
    # Region-bearing v2 workflows now run: the submit path admits them and builds the
    # orchestration engine rather than rejecting them as inspect-only.
    runtime = _runtime()
    workflow_id, _ = await runtime.register("owner", "org", REGIONS_WF, format="native")
    assert runtime.is_v2_workflow(workflow_id)
    assert runtime.orchestration_engine(workflow_id) is not None


def test_region_bearing_inspect_succeeds() -> None:
    runtime = _runtime()
    report = runtime.inspect_v2(REGIONS_WF, format="native")
    assert report is not None and report.ok


def test_region_under_v1_rejected_by_parser() -> None:
    v1 = REGIONS_WF.replace("flowmesh/v2", "flowmesh/v1")
    with pytest.raises(ValueError, match="flowmesh/v2"):
        parse_workflow(v1, "native")


_SPAWN_ONLY = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: c
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: only
        region: {kind: spawn, child: c, authority: {invoke: []}}
"""


@pytest.mark.anyio
async def test_spawn_bearing_workflow_is_admitted_as_v2() -> None:
    # A region-bearing workflow is classified v2 from the root apiVersion and admitted.
    runtime = _runtime()
    workflow_id, _ = await runtime.register(
        "owner", "org", _SPAWN_ONLY, format="native"
    )
    assert runtime.is_v2_workflow(workflow_id)
    report = runtime.inspect_v2(_SPAWN_ONLY, format="native")
    assert report is not None and report.region_bearing


def test_region_to_task_edge_is_preserved() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: route
        dependsOn: [a]
        region: {kind: merge}
      - name: after
        dependsOn: [route]
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, _ = compile_workflow("wfl", parsed, source, bindings=_BINDINGS)
    after_id = next(t.task_id for t in parsed.tasks if t.graph_node_name == "after")
    assert ("route", after_id) in {(e.from_op, e.to_op) for e in template.edges}


def test_duplicate_operator_id_is_a_compile_error() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: verify
        region: {kind: call, child: c}
      - name: 'verify:join'
        region: {kind: merge}
"""
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    with pytest.raises(CompileError):
        build_inspection("wfl", parsed, source, bindings=_BINDINGS)


def test_non_string_authority_list_is_rejected() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: sp
        region: {kind: spawn, child: c, authority: {invoke: 42}}
"""
    with pytest.raises(CompileError) as exc:
        _compile(text)
    assert "v2.not-string-list" in {d.code for d in exc.value.diagnostics}


def test_spec_v2_under_v1_rejected_by_parser() -> None:
    text = """
apiVersion: flowmesh/v1
kind: InferenceTask
metadata: {name: t}
spec:
  taskType: echo
  data: {type: list, items: [x]}
  v2: {provenance: live}
"""
    with pytest.raises(ValueError, match="flowmesh/v2"):
        parse_workflow(text, "native")


_MERGE_ONLY = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: route
        dependsOn: [a]
        region: {kind: merge}
"""


@pytest.mark.anyio
async def test_a_stored_branch_bearing_workflow_still_rehydrates() -> None:
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    workflow_id, _ = await runtime.register(
        "owner", "org", _MERGE_ONLY, format="native"
    )
    # A bundle stored before branch regions were refused at compile.
    bundle = json.loads(registry.v2_blobs[workflow_id])
    route = next(
        op for op in bundle["template"]["operators"] if op["operator_id"] == "route"
    )
    route.pop("combination", None)
    route.update(kind="branch", selection="s", outputs=[{"name": "p"}])
    registry.v2_blobs[workflow_id] = json.dumps(bundle)

    restored = _live_runtime(registry, "restored")
    assert await restored.rehydrate() == 1
    engine = restored.orchestration_engine(workflow_id)
    assert engine is not None
    assert any(op.kind.value == "branch" for op in engine._bundle.template.operators)
