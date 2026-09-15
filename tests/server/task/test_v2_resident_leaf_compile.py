import pytest

from server.task.parser import parse_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.diagnostics import CompileError
from server.task.v2.compiler.episodes import lower_to_episodes
from server.task.v2.compiler.pipeline import compile_workflow
from server.task.v2.representations.operators import (
    InferenceEmbodimentEligibility,
    LeafOperator,
    ServiceInterface,
)
from server.task.v2.representations.plan import EpisodeBoundaryKind
from server.task.v2.representations.source import FrontendWorkflowSource
from shared.tasks.specs import InferenceEmbodimentKind


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


def _resident_inference(
    service_body: str, task_type: str = "inference", data: str | None = None
) -> str:
    data_line = f"\n          data: {data}" if data else ""
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
          model: {{source: {{identifier: Qwen/Qwen3-4B}}}}{data_line}
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


def _inference_leaf(template) -> LeafOperator:
    return next(op for op in template.operators if isinstance(op, LeafOperator))


def test_a_resident_binding_pins_a_resident_required_embodiment():
    template, _plan = _compile(_resident_inference("{mode: resident}"))
    embodiment = _inference_leaf(template).embodiment
    assert embodiment.eligibility is InferenceEmbodimentEligibility.RESIDENT_REQUIRED
    assert embodiment.primary is None


def test_no_service_binding_pins_a_self_contained_required_embodiment():
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
          model: {source: {identifier: Qwen/Qwen3-4B}}
"""
    template, _plan = _compile(text)
    embodiment = _inference_leaf(template).embodiment
    assert (
        embodiment.eligibility is InferenceEmbodimentEligibility.SELF_CONTAINED_REQUIRED
    )
    assert embodiment.primary is None


def _undeclared_binding(**overrides: str) -> str:
    """A leaf that declares no service binding at all, so its default decides."""
    body = {
        "model": (
            "{source: {identifier: Qwen/Qwen3-4B}, "
            "vllm: {gpu_memory_utilization: 0.9}}"
        ),
        "data": '{type: list, items: ["hello"]}',
        "resources": "{hardware: {gpu: {count: 1}}}",
        **overrides,
    }
    fields = "\n".join(f"          {k}: {v}" for k, v in body.items())
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
          taskType: inference
{fields}
"""


def test_an_undeclared_binding_compiles_to_a_menu_with_a_derived_primary():
    template, plan = _compile(_undeclared_binding())
    embodiment = _inference_leaf(template).embodiment
    assert embodiment.eligibility is InferenceEmbodimentEligibility.LOCAL_ELIGIBLE
    assert embodiment.primary is None
    menu = _menu_node(plan).embodiment_menu
    assert menu.candidate(menu.primary).kind is InferenceEmbodimentKind.RESIDENT_SERVED


def test_an_undeclared_binding_that_is_unprovable_keeps_one_embodiment():
    # The proof, not the default, bounds the menu: a leaf that pins no vLLM engine
    # keeps exactly the embodiment it had before, and names no service.
    template, plan = _compile(_undeclared_binding(model="{source: {identifier: q}}"))
    leaf = _inference_leaf(template)
    assert (
        leaf.embodiment.eligibility
        is InferenceEmbodimentEligibility.SELF_CONTAINED_REQUIRED
    )
    assert leaf.service_dependency is None
    assert all(n.embodiment_menu is None for n in plan.nodes)


def test_a_leaf_whose_inference_settings_do_not_relay_keeps_one_embodiment():
    # Guided decoding from a declared template is applied by a local generation and is
    # not carried to a replica, so the two would produce structurally different output.
    template, plan = _compile(
        _undeclared_binding(inference='{templates: {answer: "{a}"}}')
    )
    leaf = _inference_leaf(template)
    assert (
        leaf.embodiment.eligibility
        is InferenceEmbodimentEligibility.SELF_CONTAINED_REQUIRED
    )
    assert all(n.embodiment_menu is None for n in plan.nodes)


def test_declared_sampling_still_admits_a_menu():
    # Sampling IS carried, so declaring it must not cost the leaf its menu.
    _template, plan = _compile(_undeclared_binding(inference="{max_tokens: 10}"))
    assert any(n.embodiment_menu is not None for n in plan.nodes)


def test_an_undeclared_mode_with_a_provable_contract_compiles_to_a_menu():
    # A binding that names a served model but no mode still declares one contract, so
    # it admits both embodiments; only an explicit mode pins a single one.
    template, plan = _compile(_local_eligible(service="{isolation: tenant-a}"))
    leaf = _inference_leaf(template)
    assert leaf.embodiment.eligibility is InferenceEmbodimentEligibility.LOCAL_ELIGIBLE
    menu = _menu_node(plan).embodiment_menu
    assert menu.candidate(menu.primary).kind is InferenceEmbodimentKind.RESIDENT_SERVED


def test_an_undeclared_mode_with_an_unprovable_contract_stays_resident():
    # An adapter is rejected by the proof and is exactly the resident serving case, so
    # the fallback keeps the resident embodiment the binding named rather than
    # stripping it down to a self-contained one.
    template, _plan = _compile(
        _local_eligible(
            service="{isolation: tenant-a}",
            model=(
                "{source: {identifier: Qwen/Qwen3-4B}, "
                "vllm: {gpu_memory_utilization: 0.9}, "
                "adapters: [{type: lora, name: a, path: hf/a}]}"
            ),
        )
    )
    leaf = _inference_leaf(template)
    assert (
        leaf.embodiment.eligibility is InferenceEmbodimentEligibility.RESIDENT_REQUIRED
    )
    assert leaf.service_dependency is not None


def _local_eligible(
    primary: str = "resident_served", service: str | None = None, **overrides: str
) -> str:
    body = {
        "model": (
            "{source: {identifier: Qwen/Qwen3-4B}, "
            "vllm: {gpu_memory_utilization: 0.9}}"
        ),
        "data": '{type: list, items: ["hello"]}',
        "resources": "{hardware: {gpu: {count: 1}}}",
        **overrides,
    }
    binding = service or f"{{mode: local_eligible, primary: {primary}}}"
    fields = "\n".join(f"          {k}: {v}" for k, v in body.items())
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
          taskType: inference
{fields}
          service: {binding}
"""


def test_a_local_eligible_binding_pins_both_embodiments_and_its_primary():
    template, _plan = _compile(_local_eligible(primary="self_contained"))
    embodiment = _inference_leaf(template).embodiment
    assert embodiment.eligibility is InferenceEmbodimentEligibility.LOCAL_ELIGIBLE
    assert embodiment.primary is InferenceEmbodimentKind.SELF_CONTAINED


def _menu_node(plan):
    return next(n for n in plan.nodes if n.embodiment_menu is not None)


def test_a_local_eligible_leaf_compiles_to_a_two_candidate_menu():
    _template, plan = _compile(_local_eligible())
    node = _menu_node(plan)
    menu = node.embodiment_menu

    kinds = {c.kind for c in menu.candidates}
    assert kinds == {
        InferenceEmbodimentKind.RESIDENT_SERVED,
        InferenceEmbodimentKind.SELF_CONTAINED,
    }
    assert menu.contract_fingerprint
    assert menu.candidate(menu.primary).kind is InferenceEmbodimentKind.RESIDENT_SERVED

    # One logical leaf, one source map, one physical node.
    assert node.logical_ref == node.source_ref
    assert len([n for n in plan.nodes if n.logical_ref == node.logical_ref]) == 1


def test_an_unresolved_menu_registers_no_residency_demand():
    _template, plan = _compile(_local_eligible())
    node = _menu_node(plan)
    assert node.service_family_requirement is None
    assert node.residency_intent is None
    assert node.episode is None

    resident = next(
        c
        for c in node.embodiment_menu.candidates
        if c.kind is InferenceEmbodimentKind.RESIDENT_SERVED
    )
    assert resident.residency_intent.conditional is True
    assert resident.residency_intent.required is False
    assert resident.service_family_requirement.family == "Qwen/Qwen3-4B|chat"


def test_each_candidate_carries_its_own_episode_and_envelope():
    _template, plan = _compile(_local_eligible())
    by_kind = {c.kind: c for c in _menu_node(plan).embodiment_menu.candidates}

    resident = by_kind[InferenceEmbodimentKind.RESIDENT_SERVED]
    assert resident.episode.boundary is EpisodeBoundaryKind.SERVICE_ISSUE
    assert resident.local is None

    local = by_kind[InferenceEmbodimentKind.SELF_CONTAINED]
    assert local.episode.boundary is EpisodeBoundaryKind.TASK
    assert local.local.executor_key == "vllm"
    assert local.local.gpu_count == 1


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"data": "{type: dataset, url: squad}"}, "not projectable"),
        ({"data": "{type: list, expr: upstream.items}"}, "max_items"),
        (
            {"model": "{source: {identifier: Qwen/Qwen3-4B}}"},
            "does not pin the vLLM engine",
        ),
        (
            {"postprocess": "{jsonl_export: {path: out.jsonl, fields: {a: b}}}"},
            "postprocessing",
        ),
    ],
)
def test_an_unproven_contract_compiles_to_no_menu(overrides, reason):
    with pytest.raises(CompileError, match=reason):
        _compile(_local_eligible(**overrides))


def test_a_leaf_declaring_several_prompts_compiles_to_a_menu():
    # A batch declares one contract and is served as one invocation, so it admits the
    # same two embodiments a single-prompt leaf does.
    _template, plan = _compile(
        _local_eligible(data='{type: list, items: ["a", "b", "c"]}')
    )
    menu = _menu_node(plan).embodiment_menu

    assert {c.kind for c in menu.candidates} == {
        InferenceEmbodimentKind.RESIDENT_SERVED,
        InferenceEmbodimentKind.SELF_CONTAINED,
    }
    assert menu.candidate(menu.primary) is not None
    # The number the claim's credit is sized from. A menu that lost it would admit the
    # whole batch on one slot.
    assert menu.max_batch_size == 3


def test_a_leafs_dependency_carries_the_batch_its_claim_reserves_for():
    template, _plan = _compile(
        _local_eligible(data='{type: list, items: ["a", "b", "c"]}')
    )
    assert _inference_leaf(template).service_dependency.batch_size == 3


def test_a_single_prompt_leafs_dependency_reserves_one():
    template, _plan = _compile(_local_eligible())
    assert _inference_leaf(template).service_dependency.batch_size == 1


def test_an_embedding_leafs_dependency_reserves_one():
    # An embedding request carries its whole input list, so it is one engine sequence.
    template, _plan = _compile(
        _resident_inference(
            "{mode: resident}",
            task_type="embedding",
            data='{type: list, items: ["a", "b"]}',
        )
    )
    assert _resident_leaf(template).service_dependency.batch_size == 1


def test_a_resident_required_leaf_compiles_to_a_single_embodiment():
    _template, plan = _compile(_resident_inference("{mode: resident}"))
    assert all(n.embodiment_menu is None for n in plan.nodes)
    resident = [n for n in plan.nodes if n.service_family_requirement is not None]
    assert resident[0].residency_intent.required is True
    assert resident[0].residency_intent.conditional is False


def test_a_pinned_resident_leaf_serves_the_batch_its_contract_projects():
    # A pin asks for a replica, and a replica serves a projectable batch under one
    # claim, so the author gets the batch they pinned rather than a refusal.
    template, plan = _compile(
        _local_eligible(
            service="{mode: resident}", data='{type: list, items: ["a", "b"]}'
        )
    )
    leaf = _inference_leaf(template)
    assert (
        leaf.embodiment.eligibility is InferenceEmbodimentEligibility.RESIDENT_REQUIRED
    )
    assert leaf.service_dependency.batch_size == 2
    # The pin forbids the self-contained embodiment, so there is nothing to choose.
    assert all(n.embodiment_menu is None for n in plan.nodes)


def test_a_pinned_resident_batch_with_no_projectable_request_is_rejected():
    # A leaf with no contract to project has no batch to serve, so serving its first
    # prompt alone would lose the rest silently.
    with pytest.raises(CompileError, match="serves several prompts as the batch"):
        _compile(
            _resident_inference(
                "{mode: resident}", data='{type: list, items: ["a", "b"]}'
            )
        )


def test_an_unprovable_resident_leaf_declaring_several_prompts_is_rejected():
    # The same holds for a leaf that falls back to resident because its embodiments
    # are not provably equivalent: there is no proven contract to project either way.
    with pytest.raises(CompileError, match="serves several prompts as the batch"):
        _compile(
            _resident_inference(
                "{isolation: tenant-a}",
                data='{type: list, items: ["a", "b"]}',
            )
        )


def test_a_pinned_resident_leaf_declaring_one_prompt_still_compiles():
    _template, plan = _compile(
        _resident_inference("{mode: resident}", data='{type: list, items: ["a"]}')
    )
    assert all(n.embodiment_menu is None for n in plan.nodes)


def test_a_resident_embedding_leaf_embeds_several_inputs():
    # An embedding request carries its whole input list, so it is not a batch of
    # conversations and the rejection does not apply to it.
    _template, plan = _compile(
        _resident_inference(
            "{mode: resident}",
            task_type="embedding",
            data='{type: list, items: ["a", "b"]}',
        )
    )
    assert any(n.service_family_requirement is not None for n in plan.nodes)


def test_a_local_eligible_embedding_leaf_is_rejected():
    with pytest.raises(ValueError, match="local_eligible is available for inference"):
        _compile(
            _resident_inference(
                "{mode: local_eligible, primary: self_contained}",
                task_type="embedding",
            )
        )
