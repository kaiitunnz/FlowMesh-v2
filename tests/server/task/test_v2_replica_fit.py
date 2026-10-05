"""A menu offers resident serving only when a replica can run the leaf as declared.

A replica runs with the deployment's model access, serves its base model, and is
chosen before upstream values render. A pinned resident leaf runs on the replica's
terms, unless the replica would serve another model than the leaf's.
"""

import json

import pytest

from server.task.parser import parse_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.diagnostics import CompileError
from server.task.v2.compiler.pipeline import compile_workflow
from server.task.v2.representations.plan import PhysicalNode
from server.task.v2.representations.source import FrontendWorkflowSource


def _leaf(
    vllm: str = "{max_model_len: 1024}",
    service: str = "",
    gpus: int = 1,
    gpu_type: str = "",
    revision: str = "",
    extra: str = "",
) -> PhysicalNode:
    service_line = f"\n          service: {service}" if service else ""
    revision_line = f", revision: '{revision}'" if revision else ""
    type_line = f", type: '{gpu_type}'" if gpu_type else ""
    text = f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: t}}
spec:
  taskType: echo
  graph:
    nodes:
      - name: u
        spec: {{taskType: echo}}
      - name: a
        dependsOn: [u]
        spec:
          taskType: inference
          model:
            source: {{identifier: Qwen/Qwen3-4B{revision_line}}}
            vllm: {vllm}
          resources: {{hardware: {{gpu: {{count: {gpus}{type_line}}}}}}}
          data: {{type: list, items: [hi]}}{service_line}{extra}
"""
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    _, plan = compile_workflow("wfl-t", parsed, source, bindings=AgentBindingDefaults())
    [leaf] = [task.task_id for task in parsed.tasks if task.graph_node_name == "a"]
    return next(node for node in plan.nodes if node.source_ref == leaf)


_UNFIT = {
    "hf_token": {"vllm": "{hf_token: hf_x}"},
    "credential_env": {
        "vllm": "{env_vars: {HF_TOKEN: hf_x, VLLM_LOGGING_LEVEL: INFO}}"
    },
    "templated_tensor_parallel": {"vllm": "{tensor_parallel_size: '${u.output}'}"},
    "templated_profile_key": {"vllm": "{max_model_len: '${u.output}'}"},
    "templated_revision": {"revision": "${u.output}"},
    "checkpoint": {"extra": "\n          checkpoint: {load: {type: local, path: /c}}"},
}


@pytest.mark.parametrize("case", _UNFIT.values(), ids=_UNFIT.keys())
def test_a_leaf_a_replica_cannot_run_as_declared_gets_no_resident_candidate(
    case: dict[str, object],
) -> None:
    node = _leaf(**case)  # type: ignore[arg-type]
    assert node.embodiment_menu is None
    assert node.service_family_requirement is None


@pytest.mark.parametrize("case", _UNFIT.values(), ids=_UNFIT.keys())
def test_an_explicit_menu_a_replica_cannot_run_is_refused(
    case: dict[str, object],
) -> None:
    with pytest.raises(CompileError, match="not-contract-equivalent|one proven"):
        _leaf(service="{mode: local_eligible}", **case)  # type: ignore[arg-type]


_MULTI_GPU = {
    "two_gpus": {"gpus": 2},
    # A vLLM inference leaf declares its count, which caps the size it shards over.
    "tensor_parallel": {"vllm": "{tensor_parallel_size: 2}", "gpus": 2},
    "tensor_parallel_string": {"vllm": "{tensor_parallel_size: '2'}", "gpus": 2},
    "tensor_parallel_float": {"vllm": "{tensor_parallel_size: 2.0}", "gpus": 2},
}


@pytest.mark.parametrize("case", _MULTI_GPU.values(), ids=_MULTI_GPU.keys())
def test_a_multi_gpu_leaf_gets_a_resident_candidate_of_its_size(
    case: dict[str, object],
) -> None:
    node = _leaf(**case)  # type: ignore[arg-type]
    assert node.embodiment_menu is not None
    [requirement] = [
        candidate.service_family_requirement
        for candidate in node.embodiment_menu.candidates
        if candidate.service_family_requirement is not None
    ]
    assert requirement.serving_size.gpu_count == 2
    assert requirement.serving_size.tensor_parallel_size == 2
    assert requirement.family.endswith("|size=cpu2,mem4Gi,tp2,gpu2xany")


_PINNED = {k: v for k, v in (_UNFIT | _MULTI_GPU).items() if k != "checkpoint"}


@pytest.mark.parametrize("case", _PINNED.values(), ids=_PINNED.keys())
def test_a_pinned_resident_leaf_runs_on_the_replicas_terms(
    case: dict[str, object],
) -> None:
    node = _leaf(service="{mode: resident}", **case)  # type: ignore[arg-type]
    assert node.service_family_requirement is not None


def test_a_pinned_leaf_profile_carries_no_credential_or_templated_value() -> None:
    node = _leaf(
        vllm="{env_vars: {HF_TOKEN: hf_x, A: '1'}, max_model_len: '${u.output}'}",
        service="{mode: resident}",
    )
    requirement = node.service_family_requirement
    assert requirement is not None
    plain = _leaf(vllm="{env_vars: {A: '1'}}", service="{mode: resident}")
    assert plain.service_family_requirement is not None
    assert requirement.family == plain.service_family_requirement.family
    assert "hf_x" not in json.dumps(node.model_dump(mode="json"))


def test_a_pinned_resident_leaf_loading_a_checkpoint_is_refused() -> None:
    with pytest.raises(CompileError, match="checkpoint"):
        _leaf(service="{mode: resident}", **_UNFIT["checkpoint"])  # type: ignore[arg-type]


def test_a_leaf_served_by_another_model_lends_it_no_profile_or_size() -> None:
    node = _leaf(
        vllm="{max_model_len: 1024, tensor_parallel_size: 2}",
        revision="refs/pr/7",
        gpus=2,
        service="{mode: resident, service_model_ref: meta-llama/Llama-3.1-8B}",
    )
    requirement = node.service_family_requirement
    assert requirement is not None
    assert requirement.serving_size.is_default
    assert requirement.family == "meta-llama/Llama-3.1-8B|chat"
    assert requirement.engine_batch_key == "meta-llama/Llama-3.1-8B|chat"


@pytest.mark.parametrize("service", ["{mode: resident}", ""])
def test_a_templated_gpu_type_sizes_a_placeable_replica(service: str) -> None:
    node = _leaf(gpu_type="${u.output}", service=service)
    requirement = node.service_family_requirement or next(
        candidate.service_family_requirement
        for candidate in (
            node.embodiment_menu.candidates if node.embodiment_menu else ()
        )
        if candidate.service_family_requirement is not None
    )
    assert requirement is not None
    assert requirement.serving_size.gpu_type == "any"
    assert "${" not in requirement.family
    assert requirement.serving_size.hardware()["gpu"]["type"] == "any"


def test_an_enforce_cpu_rendered_from_upstream_gets_no_menu() -> None:
    node = _leaf(extra="\n          enforce_cpu: '${u.output}'")
    assert node.embodiment_menu is None
    assert node.service_family_requirement is None
