"""The fixed demonstration policies, and the reach each one has."""

import asyncio
import logging
import pathlib
import tempfile
from types import SimpleNamespace
from typing import Any, cast

from server.config import OrchestrationConfig, PolicySurfaceConfig
from server.task.parser import ParsedWorkflow, parse_workflow
from server.task.runtime import TaskRuntime
from server.task.v2 import FrontendWorkflowSource, compile_workflow
from server.task.v2.compiler.inspect import build_inspection
from server.task.v2.mode import LoweringStrategy
from server.task.v2.policy import LoweringPolicy
from server.task.v2.policy.demo import DemoPolicy, FusionVetoPolicy, WarmthPolicy
from server.task.v2.policy.surface import build_policy_surface
from server.task.v2.representations.plan import (
    LoweringProvenance,
    PhysicalExecutionPlan,
    PhysicalNode,
    ResidencyIntent,
)
from server.task.v2.representations.template import LogicalWorkflowTemplate

# Two fusible pure leaves feeding a resident model boundary: the pair the
# conservative episode-cut lowering folds into one episode.
_PRELUDE = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: prelude}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: c
        dependsOn: [b]
        spec:
          taskType: inference
          model: {source: {identifier: Qwen/Qwen3-4B}}
          data: {type: list, items: ["hello"]}
          service: {mode: resident}
"""

_SERVE = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: serve}
spec:
  taskType: serve
  resources: {hardware: {gpu: {type: any, count: 1}}}
  model: {source: {type: huggingface, identifier: org/served}}
"""

_MENU = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: menu}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source: {identifier: Qwen/Qwen3-4B}
            vllm: {gpu_memory_utilization: 0.9}
          data: {type: list, items: ["hello"]}
          resources: {hardware: {gpu: {count: 1}}}
          service: {mode: local_eligible, primary: resident_served}
"""


class _Workflow:
    """One parse, compiled under several policies so operator ids line up."""

    def __init__(self, text: str) -> None:
        self.parsed: ParsedWorkflow = parse_workflow(text, "native")
        self.source = FrontendWorkflowSource.capture(text, "native", name="wf")
        self.names = {
            task.task_id: task.graph_node_name
            for task in self.parsed.tasks
            if task.graph_node_name
        }

    def named(self, plan: PhysicalExecutionPlan) -> dict[str, tuple[str, ...]]:
        """Each episode's fused operators, keyed by the source's own node names."""
        return {
            self.names[node.logical_ref]: tuple(
                sorted(self.names[ref] for ref in node.episode.fused_refs)
            )
            for node in plan.nodes
            if node.episode is not None and node.logical_ref in self.names
        }

    def compile(
        self,
        policy: LoweringPolicy | None = None,
        strategy: LoweringStrategy = LoweringStrategy.EPISODE_CUT,
    ) -> tuple[LogicalWorkflowTemplate, PhysicalExecutionPlan]:
        return compile_workflow(
            "wfl-d", self.parsed, self.source, strategy=strategy, policy=policy
        )

    def plan(
        self,
        policy: LoweringPolicy | None = None,
        strategy: LoweringStrategy = LoweringStrategy.EPISODE_CUT,
    ) -> PhysicalExecutionPlan:
        return self.compile(policy, strategy)[1]


def _resident_node(plan: PhysicalExecutionPlan) -> PhysicalNode:
    return next(
        node for node in plan.nodes if node.service_family_requirement is not None
    )


def _intent(plan: PhysicalExecutionPlan) -> ResidencyIntent:
    intent = _resident_node(plan).residency_intent
    assert intent is not None
    return intent


def _lowering(plan: PhysicalExecutionPlan) -> LoweringProvenance:
    assert plan.lowering is not None
    return plan.lowering


def test_the_fusion_veto_cuts_the_boundary_adjacent_pure_leaf() -> None:
    prelude = _Workflow(_PRELUDE)
    assert prelude.named(prelude.plan()) == {"a": ("b",), "c": ()}
    refined = prelude.named(prelude.plan(FusionVetoPolicy()))
    assert refined == {"a": (), "b": (), "c": ()}


def test_the_fusion_veto_preserves_the_logical_contract() -> None:
    prelude = _Workflow(_PRELUDE)
    baseline, base_plan = prelude.compile()
    refined, refined_plan = prelude.compile(FusionVetoPolicy())
    assert refined.operators == baseline.operators
    assert refined.edges == baseline.edges
    assert refined.source_map == baseline.source_map
    assert refined.result_declarations == baseline.result_declarations
    assert refined.effect_boundaries == baseline.effect_boundaries
    assert (
        _resident_node(refined_plan).service_family_requirement
        == _resident_node(base_plan).service_family_requirement
    )


def test_the_fusion_veto_is_inert_under_the_transparent_strategy() -> None:
    prelude = _Workflow(_PRELUDE)
    refined = prelude.plan(FusionVetoPolicy(), LoweringStrategy.TRANSPARENT)
    baseline = prelude.plan(strategy=LoweringStrategy.TRANSPARENT)
    assert refined.nodes == baseline.nodes


def test_the_warmth_policy_stamps_a_required_ordinary_dependency() -> None:
    prelude = _Workflow(_PRELUDE)
    assert _intent(prelude.plan()).warmth is None
    intent = _intent(prelude.plan(WarmthPolicy()))
    assert intent.warmth == "warm"
    assert intent.required is True


def test_the_warmth_policy_leaves_the_rest_of_the_intent_alone() -> None:
    prelude = _Workflow(_PRELUDE)
    baseline = _intent(prelude.plan())
    refined = _intent(prelude.plan(WarmthPolicy()))
    assert refined == baseline.model_copy(update={"warmth": "warm"})


def test_the_warmth_policy_does_not_reach_a_serve_node() -> None:
    serve = _Workflow(_SERVE)
    # A serve node declares its own standing residency and never consults the hook,
    # so its intent is the same with and without the policy.
    assert (
        _resident_node(serve.plan(WarmthPolicy())).residency_intent
        == _resident_node(serve.plan()).residency_intent
    )


def test_the_warmth_policy_does_not_reach_an_unresolved_menu() -> None:
    menu = _Workflow(_MENU)
    (node,) = [n for n in menu.plan(WarmthPolicy()).nodes if n.embodiment_menu]
    assert node.residency_intent is None and node.embodiment_menu is not None
    intents = [
        candidate.residency_intent
        for candidate in node.embodiment_menu.candidates
        if candidate.residency_intent is not None
    ]
    assert intents and all(
        intent.conditional and intent.warmth is None for intent in intents
    )


def test_the_demo_policy_applies_both_refinements() -> None:
    prelude = _Workflow(_PRELUDE)
    plan = prelude.plan(DemoPolicy())
    assert prelude.named(plan) == prelude.named(prelude.plan(FusionVetoPolicy()))
    assert _intent(plan).warmth == "warm"


def test_a_plan_records_the_lowering_that_produced_it() -> None:
    prelude = _Workflow(_PRELUDE)
    lowering = _lowering(prelude.plan(DemoPolicy()))
    assert lowering.strategy == LoweringStrategy.EPISODE_CUT.value
    assert lowering.policy == "d30-demo"


def test_a_deployment_running_no_policy_records_the_effective_one() -> None:
    lowering = _lowering(
        _Workflow(_PRELUDE).plan(strategy=LoweringStrategy.TRANSPARENT)
    )
    assert lowering.strategy == LoweringStrategy.TRANSPARENT.value
    assert lowering.policy == "conservative"


def test_the_lowering_separates_two_otherwise_equal_plan_versions() -> None:
    prelude = _Workflow(_PRELUDE)
    baseline = prelude.plan(strategy=LoweringStrategy.TRANSPARENT)
    refined = prelude.plan(FusionVetoPolicy(), LoweringStrategy.TRANSPARENT)
    # Nothing but the recorded lowering differs, and the version still separates them.
    assert refined.nodes == baseline.nodes
    assert refined.plan_version != baseline.plan_version


def test_an_inspection_reports_the_plan_the_same_lowering_produces() -> None:
    prelude = _Workflow(_PRELUDE)
    report = build_inspection(
        "wfl-d",
        prelude.parsed,
        prelude.source,
        strategy=LoweringStrategy.EPISODE_CUT,
        policy=DemoPolicy(),
    )
    assert report.plan == prelude.plan(DemoPolicy(), LoweringStrategy.EPISODE_CUT)
    assert "policy=d30-demo" in report.render_text()


def test_each_demo_policy_is_selectable_by_name() -> None:
    for name in ("conservative", "d30-fusion", "d30-warmth", "d30-demo"):
        surface = build_policy_surface(PolicySurfaceConfig(enabled=True, lowering=name))
        assert surface is not None and surface.lowering.name == name


class _CapturingRegistry:
    async def register_workflow_async(
        self, workflow_id: str, tasks: list, v2=None
    ) -> None:
        return None

    async def save_task_states_async(self, items: list) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        return None

    async def save_ledger_snapshot_async(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoopSecretVault:
    async def put(self, *args, **kwargs) -> None:
        return None


def _runtime(lowering: str) -> TaskRuntime:
    config = PolicySurfaceConfig(enabled=True, lowering=lowering)
    worker_stub = SimpleNamespace(
        get_worker=lambda wid: SimpleNamespace(id=wid, node_id="nde-1"),
        publish_interrupt=lambda *a: 0,
    )
    return TaskRuntime(
        cast(Any, _CapturingRegistry()),
        cast(Any, worker_stub),
        OrchestrationConfig(policy=config),
        pathlib.Path(tempfile.gettempdir()),
        logging.getLogger("lowering-policy-demo-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
        policy=build_policy_surface(config),
    )


def _resident_binding(lowering: str):
    runtime = _runtime(lowering)
    _workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _PRELUDE, format="native")
    )
    inference = next(r for r in results if r.graph_node_name == "c")
    return runtime.resolve_service_dependency(inference.task_id)


def test_the_configured_policy_reaches_the_resolved_admission_binding() -> None:
    binding = _resident_binding("d30-warmth")
    assert binding is not None
    assert binding.warmth == "warm"
    assert binding.compatible()
    assert binding.dependency.service_ref == "Qwen/Qwen3-4B"


def test_a_conservative_deployment_resolves_an_unstyled_binding() -> None:
    binding = _resident_binding("conservative")
    assert binding is not None and binding.warmth is None


def test_a_dry_run_inspection_matches_what_the_runtime_would_register() -> None:
    runtime = _runtime("d30-demo")
    report = runtime.inspect_v2(_PRELUDE, format="native")
    assert report is not None
    lowering = _lowering(report.plan)
    assert lowering.policy == "d30-demo"
    assert lowering.strategy == LoweringStrategy.TRANSPARENT.value
