import logging
import pathlib
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from server.task.v2 import (
    CompileError,
    EffectClass,
    FrontendWorkflowSource,
    PersistedV2Workflow,
    compile_workflow,
)
from server.task.v2.compiler.bindings import BindingClass, binding_class
from shared.tasks import TaskType

_EXAMPLES = pathlib.Path(__file__).parents[3] / "examples" / "templates"
_TEMPLATE_FILES = sorted(_EXAMPLES.glob("*.yaml"))
_FORBIDDEN = ("worker_id", "replica", "endpoint", "activation", "attempt")


def _compile(text: str) -> tuple[Any, Any, Any]:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, plan = compile_workflow("wfl-test", parsed, source)
    return parsed, template, plan


@pytest.mark.parametrize("path", _TEMPLATE_FILES, ids=lambda p: p.name)
def test_examples_compile_to_acyclic_templates(path: pathlib.Path) -> None:
    """Every current example compiles to an equivalent acyclic template."""
    v2_text = path.read_text().replace("flowmesh/v1", "flowmesh/v2")
    parsed, template, _ = _compile(v2_text)

    # One operator per task; dependsOn preserved as edges.
    assert len(template.operators) == len(parsed.tasks)
    task_ids = {t.task_id for t in parsed.tasks}
    forward_edges = {(e.from_op, e.to_op) for e in template.edges if not e.feedback}
    expected_edges = {
        (dep, task.task_id)
        for task in parsed.tasks
        for dep in task.depends_on
        if dep in task_ids
    }
    assert expected_edges <= forward_edges


@pytest.mark.parametrize("path", _TEMPLATE_FILES, ids=lambda p: p.name)
def test_examples_source_map_is_complete(path: pathlib.Path) -> None:
    v2_text = path.read_text().replace("flowmesh/v1", "flowmesh/v2")
    _, template, plan = _compile(v2_text)
    mapped = {entry.logical_ref for entry in template.source_map}
    assert mapped == template.operator_ids
    # Every physical node maps back to a known operator.
    for node in plan.nodes:
        assert node.logical_ref in template.operator_ids


@pytest.mark.parametrize("path", _TEMPLATE_FILES, ids=lambda p: p.name)
def test_examples_are_symbolic(path: pathlib.Path) -> None:
    v2_text = path.read_text().replace("flowmesh/v1", "flowmesh/v2")
    _, template, plan = _compile(v2_text)
    blob = template.model_dump_json() + plan.model_dump_json()
    assert not any(token in blob for token in _FORBIDDEN)


@pytest.mark.parametrize("path", _TEMPLATE_FILES, ids=lambda p: p.name)
def test_physical_plan_one_boundary_per_task(path: pathlib.Path) -> None:
    v2_text = path.read_text().replace("flowmesh/v1", "flowmesh/v2")
    parsed, _, plan = _compile(v2_text)
    # One physical boundary per legacy task/executor boundary.
    assert len(plan.nodes) == len(parsed.tasks)
    assert {n.logical_ref for n in plan.nodes} == {t.task_id for t in parsed.tasks}


def test_binding_registry_is_an_adapter() -> None:
    assert binding_class(TaskType.AGENT) is BindingClass.AGENT
    assert binding_class(TaskType.SERVE) is BindingClass.RESIDENCY
    assert binding_class(TaskType.INFERENCE) is BindingClass.LEAF
    assert binding_class(TaskType.SFT) is BindingClass.LEAF


def test_effect_override_to_external_induces_boundary() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: n
        spec:
          taskType: echo
          data: {type: list, items: [x]}
          v2: {effect: external_effect, recovery: record}
"""
    _, template, _ = _compile(text)
    (op,) = template.operators
    assert op.profile.effect is EffectClass.EXTERNAL_EFFECT
    # The boundary is induced from the overridden profile, so validation passes.
    assert {b.source_ref for b in template.effect_boundaries} == {op.operator_id}


def test_effect_override_to_pure_drops_boundary() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: n
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
          v2: {effect: pure}
"""
    _, template, _ = _compile(text)
    (op,) = template.operators
    assert op.profile.effect is EffectClass.PURE
    # No orphan boundary survives an override away from external effect.
    assert template.effect_boundaries == ()


def test_readable_source_location_in_error() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
          v2: {recovery: recompute}
"""
    with pytest.raises(CompileError) as exc:
        _compile(text)
    rendered = "; ".join(d.render() for d in exc.value.diagnostics)
    assert "graph node 'caller'" in rendered


# --------------------------------------------------------------------------- #
# Compile-before-mutate + old-parser regression
# --------------------------------------------------------------------------- #


class _CapturingRegistry:
    def __init__(self) -> None:
        self.v2: dict[str, PersistedV2Workflow | None] = {}

    async def register_workflow_async(
        self, workflow_id: str, tasks: list[Any], v2: Any = None
    ) -> None:
        self.v2[workflow_id] = v2

    async def save_task_states_async(self, items: list[Any]) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
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
        OrchestrationConfig(),
        logging.getLogger("v2-compiler-test"),
    )


_BAD_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
          v2: {recovery: recompute}
"""


@pytest.mark.anyio
async def test_compile_error_leaves_no_orphan_state() -> None:
    runtime = _runtime()
    with pytest.raises(CompileError):
        await runtime.register("owner", "org", _BAD_V2, format="native")
    # The failed submission mutated no in-memory scheduler state.
    assert runtime.list_tasks() == []


def test_old_parser_path_still_selectable() -> None:
    # Every example still parses through the untouched v1 parser path.
    for path in _TEMPLATE_FILES:
        parsed = parse_workflow(path.read_text(), "native")
        assert parsed.tasks
        assert parsed.regions == []
