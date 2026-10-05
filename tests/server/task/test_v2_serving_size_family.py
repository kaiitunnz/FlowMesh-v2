"""A resident leaf shares a family only with leaves declaring its serving size."""

import pytest

from server.task.parser import parse_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.pipeline import compile_workflow
from server.task.v2.policy.lowering import screen_service_family
from server.task.v2.representations.admission import ResidentAdmissionBinding
from server.task.v2.representations.operators import (
    LeafOperator,
    ServiceDependency,
)
from server.task.v2.representations.plan import ServiceFamilyRequirement
from server.task.v2.representations.serving_size import (
    DEFAULT_SERVING_SIZE,
    ServingSize,
)
from server.task.v2.representations.source import FrontendWorkflowSource


def _compile(leaf: str) -> tuple[ServiceDependency, ServiceFamilyRequirement]:
    text = f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: t}}
spec:
  graph:
    nodes:
      - name: a
        spec:
{leaf}
"""
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, plan = compile_workflow(
        "wfl-t", parsed, source, bindings=AgentBindingDefaults()
    )
    [op] = [op for op in template.operators if isinstance(op, LeafOperator)]
    assert op.service_dependency is not None
    [node] = plan.nodes
    requirement = node.service_family_requirement
    if node.embodiment_menu is not None:
        [requirement] = [
            candidate.service_family_requirement
            for candidate in node.embodiment_menu.candidates
            if candidate.service_family_requirement is not None
        ]
    assert requirement is not None
    return op.service_dependency, requirement


def _chat(hardware: str, vllm: str = "{max_model_len: 1024}", service: str = "") -> str:
    service_line = f"\n          service: {service}" if service else ""
    return f"""          taskType: inference
          model:
            source: {{identifier: Qwen/Qwen3-4B}}
            vllm: {vllm}
          resources: {{hardware: {hardware}}}
          data: {{type: list, items: [hi]}}{service_line}"""


def _embedding(hardware: str, vllm: str = "{}") -> str:
    return f"""          taskType: embedding
          model:
            source: {{identifier: Qwen/Qwen3-Embedding-0.6B}}
            vllm: {vllm}
          resources: {{hardware: {hardware}}}
          service: {{mode: resident}}"""


def _family(leaf: str) -> tuple[str, str | None]:
    _dependency, requirement = _compile(leaf)
    return requirement.family, requirement.engine_batch_key


_DEFAULT = ("Qwen/Qwen3-4B|chat|profile=", "Qwen/Qwen3-4B|chat|profile=")


@pytest.mark.parametrize(
    "hardware",
    [
        "{gpu: {count: 1}}",
        "{cpu: 2, memory: 4Gi, gpu: {count: 1, type: any}}",
        "{cpu: 2, memory: 4096Mi, gpu: {count: 1, type: '*'}}",
    ],
)
def test_a_default_size_leaf_keeps_the_family_it_had_before_sizes(
    hardware: str,
) -> None:
    family, batch_key = _family(_chat(hardware))
    assert family.startswith(_DEFAULT[0]) and "size=" not in family
    assert batch_key is not None and "size=" not in batch_key
    assert family == _family(_chat("{gpu: {count: 1}}"))[0]


def test_equivalent_spellings_of_a_size_compile_to_one_family() -> None:
    sized = _family(_chat("{cpu: 8, memory: 4Gi, gpu: {count: 2, type: H100}}"))
    assert sized == _family(
        _chat("{cpu: 8, memory: 4096Mi, gpu: {count: 2, type: h100}}")
    )
    assert sized == _family(
        _chat(
            "{cpu: 8, memory: 4294967296, gpu: {count: 2, type: ' H100 '}}",
            vllm="{max_model_len: 1024, tensor_parallel_size: 2}",
        )
    )
    assert sized[0].endswith("|size=cpu8,mem4Gi,tp2,gpu2xh100")
    # A count alone and a tensor-parallel size alone are one size.
    assert _family(_embedding("{gpu: {count: 2}}")) == _family(
        _embedding("{}", vllm="{tensor_parallel_size: 2}")
    )


@pytest.mark.parametrize(
    ("hardware", "vllm"),
    [
        ("{cpu: 4, gpu: {count: 1}}", ""),
        ("{memory: 8Gi, gpu: {count: 1}}", ""),
        ("{gpu: {count: 1, type: a100}}", ""),
        ("{gpu: {count: 1, memory: 40Gi}}", ""),
        ("{gpu: {count: 2}}", ""),
        ("{gpu: {count: 2}}", "tensor_parallel_size: 1, "),
    ],
)
def test_each_distinct_size_is_its_own_family(hardware: str, vllm: str) -> None:
    family, batch_key = _family(_chat(hardware, vllm=f"{{{vllm}max_model_len: 1024}}"))
    default_family, default_key = _family(_chat("{gpu: {count: 1}}"))
    assert family != default_family
    assert batch_key != default_key


def test_a_menu_and_a_pinned_leaf_of_one_size_share_a_family() -> None:
    hardware = "{cpu: 8, gpu: {count: 2}}"
    menu_dependency, menu = _compile(_chat(hardware))
    pinned_dependency, pinned = _compile(_chat(hardware, service="{mode: resident}"))

    assert menu.family == pinned.family
    assert menu == menu_dependency.family_requirement()
    assert pinned == pinned_dependency.family_requirement()
    assert pinned.serving_size == ServingSize(
        cpu=8, gpu_count=2, tensor_parallel_size=2
    )
    binding = ResidentAdmissionBinding(
        workflow_id="wfl-t", dependency=pinned_dependency, requirement=pinned
    )
    assert binding.compatible()
    other = pinned.model_copy(update={"serving_size": DEFAULT_SERVING_SIZE})
    assert not binding.model_copy(update={"requirement": other}).compatible()


def test_a_policy_cannot_move_a_requirement_to_another_size() -> None:
    _dependency, derived = _compile(
        _chat("{gpu: {count: 2}}", service="{mode: resident}")
    )
    resized = derived.model_copy(update={"serving_size": DEFAULT_SERVING_SIZE})
    renamed = derived.model_copy(update={"family": "pool-a"})

    assert screen_service_family(derived, resized) == derived
    assert screen_service_family(derived, renamed) == renamed


def test_a_dependency_stored_without_a_size_reads_as_the_default() -> None:
    dependency, _requirement = _compile(_chat("{gpu: {count: 1}}"))
    stored = dependency.model_dump(mode="json")
    stored.pop("serving_size")

    restored = ServiceDependency.model_validate(stored)

    assert restored.serving_size == DEFAULT_SERVING_SIZE
    assert restored.service_family == dependency.service_family
    assert restored.engine_batch_key == dependency.engine_batch_key
