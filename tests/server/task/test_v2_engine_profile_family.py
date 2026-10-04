"""A resident leaf shares a family only with leaves declaring its engine profile."""

from server.task.parser import parse_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.pipeline import compile_workflow
from server.task.v2.representations.source import FrontendWorkflowSource


def _family(vllm: str, revision: str = "") -> tuple[str, str | None]:
    revision_line = f", revision: {revision}" if revision else ""
    text = f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: t}}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source: {{identifier: Qwen/Qwen3-4B{revision_line}}}
            vllm: {vllm}
          resources: {{hardware: {{gpu: {{count: 1}}}}}}
          data: {{type: list, items: [hi]}}
"""
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    _, plan = compile_workflow("wfl-t", parsed, source, bindings=AgentBindingDefaults())
    [node] = plan.nodes
    assert node.embodiment_menu is not None
    requirement = next(
        candidate.service_family_requirement
        for candidate in node.embodiment_menu.candidates
        if candidate.service_family_requirement is not None
    )
    return requirement.family, requirement.engine_batch_key


def test_a_leaf_declaring_no_profile_keeps_the_shared_family() -> None:
    family = ("Qwen/Qwen3-4B|chat", "Qwen/Qwen3-4B|chat")
    assert _family("{gpu_memory_utilization: 0.5}") == family
    assert _family("{enforce_eager: true, seed: 3}", revision="main") == family


def test_leaves_declaring_different_profiles_land_on_different_families() -> None:
    short = _family("{max_model_len: 1024}")
    long = _family("{max_model_len: 4096}")
    plain = _family("{gpu_memory_utilization: 0.5}")

    assert len({short, long, plain}) == 3
    assert _family("{max_model_len: 1024, gpu_memory_utilization: 0.3}") == short
    assert _family("{gpu_memory_utilization: 0.5}", revision="v1") != plain
