import pytest

from server.task.parser import parse_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.diagnostics import CompileError
from server.task.v2.compiler.episodes import lower_to_episodes
from server.task.v2.compiler.pipeline import compile_workflow
from server.task.v2.representations.operators import LeafOperator, ServiceInterface
from server.task.v2.representations.plan import EpisodeBoundaryKind
from server.task.v2.representations.source import FrontendWorkflowSource


def _compile(text):
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    return compile_workflow("wfl-t", parsed, source, bindings=AgentBindingDefaults())


def _resident_leaf(template) -> LeafOperator:
    return next(
        op
        for op in template.operators
        if isinstance(op, LeafOperator) and op.service_dependency is not None
    )


def _resident_inference(service_body: str, task_type: str = "inference") -> str:
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: t}}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: {task_type}
          model: {{source: {{identifier: Qwen/Qwen3-4B}}}}
          service: {service_body}
"""


def test_resident_inference_leaf_pins_a_service_family_and_intent():
    template, plan = _compile(_resident_inference("{mode: resident}"))
    leaf = _resident_leaf(template)
    assert leaf.service_dependency is not None
    assert leaf.service_dependency.service_ref == "Qwen/Qwen3-4B"
    assert leaf.service_dependency.interface is ServiceInterface.CHAT

    resident = [n for n in plan.nodes if n.service_family_requirement is not None]
    assert len(resident) == 1
    requirement = resident[0].service_family_requirement
    assert requirement.family == "Qwen/Qwen3-4B|chat"
    assert resident[0].residency_intent.required is True


def test_resident_leaf_carries_the_adapter_and_its_source():
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source: {identifier: Qwen/Qwen3-4B}
            adapters: [{type: lora, name: my-lora, path: hf/my-lora}]
          service: {mode: resident}
"""
    template, plan = _compile(text)
    dep = _resident_leaf(template).service_dependency
    assert dep is not None
    assert dep.adapter == "my-lora"
    assert dep.adapter_source == "hf/my-lora"
    # The adapter does not fork a family: the family key stays base+interface.
    resident = [n for n in plan.nodes if n.service_family_requirement is not None]
    assert resident[0].service_family_requirement.family == "Qwen/Qwen3-4B|chat"


def test_resident_embedding_leaf_with_an_adapter_is_rejected():
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: embedding
          model:
            source: {identifier: BAAI/bge-small-en-v1.5}
            adapters: [{type: lora, name: my-lora, path: hf/my-lora}]
          service: {mode: resident}
"""
    with pytest.raises(CompileError, match="embedding leaf cannot declare an adapter"):
        _compile(text)


def test_resident_leaf_lowers_to_a_service_issue_episode():
    template, plan = _compile(_resident_inference("{mode: resident}"))
    leaf_id = _resident_leaf(template).operator_id
    episodes = lower_to_episodes(template, plan.nodes)
    node = next(n for n in episodes if n.logical_ref == leaf_id)
    assert node.episode is not None
    assert node.episode.boundary is EpisodeBoundaryKind.SERVICE_ISSUE


def test_embedding_and_chat_of_one_model_pin_distinct_families():
    _, chat_plan = _compile(_resident_inference("{mode: resident}", "inference"))
    _, embed_plan = _compile(_resident_inference("{mode: resident}", "embedding"))
    chat = next(
        n.service_family_requirement.family
        for n in chat_plan.nodes
        if n.service_family_requirement is not None
    )
    embed = next(
        n.service_family_requirement.family
        for n in embed_plan.nodes
        if n.service_family_requirement is not None
    )
    assert chat != embed


def test_isolation_domain_pins_a_distinct_family():
    _, plain = _compile(_resident_inference("{mode: resident}"))
    _, isolated = _compile(_resident_inference("{mode: resident, isolation: tenant-a}"))
    plain_family = next(
        n.service_family_requirement.family
        for n in plain.nodes
        if n.service_family_requirement is not None
    )
    isolated_family = next(
        n.service_family_requirement.family
        for n in isolated.nodes
        if n.service_family_requirement is not None
    )
    assert plain_family != isolated_family


def test_explicit_service_model_ref_overrides_the_task_model():
    template, _ = _compile(
        _resident_inference("{mode: resident, service_model_ref: served/alias}")
    )
    leaf = _resident_leaf(template)
    assert leaf.service_dependency.service_ref == "served/alias"


def test_resident_binding_without_a_resolvable_ref_is_rejected():
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          service: {mode: resident}
"""
    with pytest.raises(CompileError, match="service.missing-ref"):
        _compile(text)
