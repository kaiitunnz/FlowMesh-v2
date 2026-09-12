"""The advisory lowering policy and the screens that keep its answers legal."""

from server.policy import EpisodeAnnotation, LoweringPolicy
from server.task.parser import parse_workflow
from server.task.v2 import FrontendWorkflowSource, compile_workflow
from server.task.v2.mode import LoweringStrategy
from server.task.v2.representations.operators import LogicalOperator
from server.task.v2.representations.plan import (
    EpisodeBoundaryKind,
    PhysicalExecutionPlan,
    ResidencyIntent,
    ServiceFamilyRequirement,
)

_RESIDENT = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: resident}
spec:
  taskType: echo
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model: {source: {identifier: Qwen/Qwen3-4B}}
          service: {mode: resident}
"""

_CHAIN = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: chain}
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
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""


class _NoFusion(LoweringPolicy):
    def fuse(self, predecessor: LogicalOperator, candidate: LogicalOperator) -> bool:
        return False


class _AlwaysCheckpoint(LoweringPolicy):
    def checkpoint(self, op: LogicalOperator) -> bool:
        return True


class _Annotating(LoweringPolicy):
    def annotate(
        self, op: LogicalOperator, boundary: EpisodeBoundaryKind
    ) -> EpisodeAnnotation:
        return EpisodeAnnotation(liveness_key="lane-a", speculative_eligible=True)


class _FamilySwap(LoweringPolicy):
    def __init__(self, *, compatible: bool) -> None:
        self._compatible = compatible

    def service_family(
        self, requirement: ServiceFamilyRequirement
    ) -> ServiceFamilyRequirement:
        update = {"family": "other-family"}
        if not self._compatible:
            update["engine_batch_key"] = "other-key"
        return requirement.model_copy(update=update)

    def residency(self, intent: ResidencyIntent) -> ResidencyIntent:
        return intent.model_copy(
            update={"service_family": "smuggled", "required": False, "warmth": "warm"}
        )


class _Workflow:
    """One parse, compiled under several policies so operator ids line up."""

    def __init__(self, text: str) -> None:
        self._parsed = parse_workflow(text, "native")
        self._source = FrontendWorkflowSource.capture(text, "native", name="wf")

    def plan(
        self,
        policy: LoweringPolicy | None = None,
        strategy: LoweringStrategy = LoweringStrategy.EPISODE_CUT,
    ) -> PhysicalExecutionPlan:
        _, plan = compile_workflow(
            "wfl-p", self._parsed, self._source, strategy=strategy, policy=policy
        )
        return plan


def _episodes(plan: PhysicalExecutionPlan) -> list:
    return [node.episode for node in plan.nodes if node.episode is not None]


def _resident_node(plan: PhysicalExecutionPlan):
    return next(
        node for node in plan.nodes if node.service_family_requirement is not None
    )


def test_conservative_policy_lowers_identically_to_no_policy() -> None:
    chain = _Workflow(_CHAIN)
    assert chain.plan(LoweringPolicy()).nodes == chain.plan().nodes


def test_a_vetoed_fusion_leaves_each_operator_its_own_episode() -> None:
    chain = _Workflow(_CHAIN)
    assert any(episode.fused_refs for episode in _episodes(chain.plan()))
    unfused = _episodes(chain.plan(_NoFusion()))
    assert len(unfused) == 3
    assert all(episode.fused_refs == () for episode in unfused)


def test_a_vetoed_fusion_keeps_every_operator_in_the_plan() -> None:
    chain = _Workflow(_CHAIN)
    refs = {node.logical_ref for node in chain.plan(_NoFusion()).nodes}
    transparent = {
        node.logical_ref
        for node in chain.plan(strategy=LoweringStrategy.TRANSPARENT).nodes
    }
    assert refs == transparent


def test_a_task_boundary_can_be_refined_to_a_durable_checkpoint() -> None:
    chain = _Workflow(_CHAIN)
    assert EpisodeBoundaryKind.TASK in {
        episode.boundary for episode in _episodes(chain.plan())
    }
    refined = {
        episode.boundary for episode in _episodes(chain.plan(_AlwaysCheckpoint()))
    }
    assert refined == {EpisodeBoundaryKind.DURABLE_CHECKPOINT}


def test_annotations_ride_on_the_episode() -> None:
    for episode in _episodes(_Workflow(_CHAIN).plan(_Annotating())):
        assert episode.liveness_key == "lane-a"
        assert episode.speculative_eligible is True


def test_episodes_carry_no_annotation_by_default() -> None:
    for episode in _episodes(_Workflow(_CHAIN).plan()):
        assert episode.liveness_key is None
        assert episode.speculative_eligible is False


def test_lowering_policy_is_inert_under_the_transparent_strategy() -> None:
    chain = _Workflow(_CHAIN)
    with_policy = chain.plan(_AlwaysCheckpoint(), LoweringStrategy.TRANSPARENT)
    assert with_policy.nodes == chain.plan(strategy=LoweringStrategy.TRANSPARENT).nodes


def test_a_compatible_family_refinement_is_honored() -> None:
    node = _resident_node(_Workflow(_RESIDENT).plan(_FamilySwap(compatible=True)))
    assert node.service_family_requirement.family == "other-family"
    assert node.residency_intent.service_family == "other-family"


def test_an_incompatible_family_refinement_is_discarded() -> None:
    resident = _Workflow(_RESIDENT)
    derived = _resident_node(resident.plan())
    refined = _resident_node(resident.plan(_FamilySwap(compatible=False)))
    assert refined.service_family_requirement == derived.service_family_requirement


def test_a_policy_never_unpins_a_required_residency() -> None:
    node = _resident_node(_Workflow(_RESIDENT).plan(_FamilySwap(compatible=True)))
    assert node.residency_intent.required is True
    assert node.residency_intent.service_family != "smuggled"
    assert node.residency_intent.warmth == "warm"
