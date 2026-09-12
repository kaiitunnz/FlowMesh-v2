import pytest

from server.config import PolicySurfaceConfig
from server.policy import (
    InstanceStateLocality,
    LoweringPolicy,
    PlacementContext,
    PlacementPolicy,
    RecencyWarmth,
    SealedGenerationEvidence,
    StateControlDecision,
    StateControlPolicy,
    StateControlVerb,
    build_policy_surface,
    eligible_verbs,
    screen,
    screen_residency,
    screen_service_family,
)
from server.task.v2.representations.plan import (
    ResidencyIntent,
    ServiceFamilyRequirement,
)
from shared.private_state import BundleProfile, SealedComponent, StateComponentKind


def _evidence(**overrides) -> SealedGenerationEvidence:
    base = dict(
        reference_id="aps-1",
        instance_id="wfl-1",
        activation_id="act-1",
        owner_id="user-1",
        org_id="org-1",
        tenant=None,
        profile=BundleProfile.AGENT_HARNESS,
        generation=2,
        owner_worker_id="worker-a",
        owner_incarnation=1,
        components=(
            SealedComponent(
                kind=StateComponentKind.WORKSPACE_FS,
                schema_version=1,
                content_digest="d",
                size_bytes=1,
                entry_count=1,
            ),
        ),
        attached=False,
        resumable=False,
        exportable=False,
        sealed_at="2026-01-01T00:00:00Z",
    )
    return SealedGenerationEvidence(**{**base, **overrides})


def test_surface_is_none_while_disabled() -> None:
    assert build_policy_surface(PolicySurfaceConfig()) is None


def test_surface_builds_the_configured_facets() -> None:
    surface = build_policy_surface(PolicySurfaceConfig(enabled=True))
    assert surface is not None
    assert isinstance(surface.placement, InstanceStateLocality)
    assert isinstance(surface.state_control, RecencyWarmth)
    assert surface.lowering.name == LoweringPolicy.name


def test_unknown_policy_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="ORCHESTRATOR_PLACEMENT_POLICY"):
        build_policy_surface(PolicySurfaceConfig(enabled=True, placement="nope"))


def test_default_policies_express_no_preference() -> None:
    context = PlacementContext(
        task_id="tsk-1",
        candidates=("worker-a", "worker-b"),
        instance_state_holders=frozenset({"worker-a"}),
    )
    assert PlacementPolicy().prefer(context) == frozenset()
    assert InstanceStateLocality().prefer(context) == frozenset({"worker-a"})


def test_default_state_control_retains_every_generation() -> None:
    decisions = StateControlPolicy().decide([_evidence()])
    assert decisions["aps-1"].verb is StateControlVerb.RETAIN


def test_recency_warmth_evicts_beyond_the_warm_count() -> None:
    warm = _evidence(reference_id="aps-warm", sealed_at="2026-01-02T00:00:00Z")
    cold = _evidence(reference_id="aps-cold", sealed_at="2026-01-01T00:00:00Z")
    decisions = RecencyWarmth(1).decide([cold, warm])
    assert decisions["aps-warm"].verb is StateControlVerb.RETAIN
    assert decisions["aps-cold"].verb is StateControlVerb.EVICT


def test_local_only_components_admit_no_copy_verb() -> None:
    verbs = eligible_verbs(_evidence())
    assert verbs == {StateControlVerb.RETAIN, StateControlVerb.EVICT}
    assert eligible_verbs(_evidence(exportable=True)) >= {
        StateControlVerb.REPLICATE,
        StateControlVerb.PLACE,
        StateControlVerb.PREFETCH,
    }


@pytest.mark.parametrize("live", ["attached", "resumable"])
def test_a_live_generation_is_never_evicted(live: str) -> None:
    evidence = _evidence(**{live: True})
    assert StateControlVerb.EVICT not in eligible_verbs(evidence)
    decision = screen(
        evidence, StateControlDecision(verb=StateControlVerb.EVICT, reason="cold")
    )
    assert decision.verb is StateControlVerb.RETAIN


def test_screen_keeps_an_admissible_decision() -> None:
    decision = StateControlDecision(verb=StateControlVerb.EVICT, reason="cold")
    assert screen(_evidence(), decision) is decision


def test_service_family_refinement_holds_the_compatibility_key() -> None:
    derived = ServiceFamilyRequirement(
        family="fam-a", engine_batch_key="vllm:qwen", isolation="tenant"
    )
    compatible = derived.model_copy(update={"family": "fam-b"})
    assert screen_service_family(derived, compatible) is compatible
    incompatible = derived.model_copy(update={"engine_batch_key": "vllm:other"})
    assert screen_service_family(derived, incompatible) is derived


def test_residency_refinement_holds_the_pinned_family() -> None:
    derived = ResidencyIntent(service_family="fam-a", required=True)
    refined = ResidencyIntent(
        service_family="fam-b", required=False, warmth="warm", affinity="node"
    )
    screened = screen_residency(derived, refined)
    assert screened.service_family == "fam-a"
    assert screened.required is True
    assert (screened.warmth, screened.affinity) == ("warm", "node")
