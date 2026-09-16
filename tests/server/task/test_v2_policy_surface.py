"""Assembly of the deployment's advisory policy surface and its screens."""

import pytest

from server.config import PolicySurfaceConfig
from server.task.v2.policy import (
    FusionPolicy,
    PolicySurface,
    ResidencyPolicy,
    ServiceFamilyPolicy,
    build_policy_surface,
    screen_residency,
    screen_service_family,
)
from server.task.v2.policy.demo import FusionVetoPolicy
from server.task.v2.representations.plan import (
    ResidencyIntent,
    ServiceFamilyRequirement,
)


def test_an_unconfigured_deployment_runs_the_conservative_policy_at_every_hook() -> (
    None
):
    surface = build_policy_surface(PolicySurfaceConfig())

    assert surface.fusion.name == FusionPolicy.name
    assert surface.residency.name == ResidencyPolicy.name
    assert surface.service_family.name == ServiceFamilyPolicy.name


def test_the_default_surface_is_the_conservative_one() -> None:
    default = PolicySurface()

    assert (default.fusion.name, default.residency.name) == (
        FusionPolicy.name,
        ResidencyPolicy.name,
    )


def test_each_hook_reads_its_own_selector_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_FUSION_POLICY", "custom-fusion")
    monkeypatch.setenv("ORCHESTRATOR_RESIDENCY_POLICY", "custom-residency")
    monkeypatch.setenv("ORCHESTRATOR_SERVICE_FAMILY_POLICY", "custom-family")

    config = PolicySurfaceConfig.from_env()

    assert config.fusion == "custom-fusion"
    assert config.residency == "custom-residency"
    assert config.service_family == "custom-family"


def test_an_unset_selector_falls_back_to_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for knob in (
        "ORCHESTRATOR_FUSION_POLICY",
        "ORCHESTRATOR_RESIDENCY_POLICY",
        "ORCHESTRATOR_SERVICE_FAMILY_POLICY",
    ):
        monkeypatch.delenv(knob, raising=False)

    config = PolicySurfaceConfig.from_env()

    assert (config.fusion, config.residency, config.service_family) == (
        FusionPolicy.name,
        ResidencyPolicy.name,
        ServiceFamilyPolicy.name,
    )


def test_an_unknown_name_is_rejected_against_its_own_hook() -> None:
    with pytest.raises(ValueError, match="ORCHESTRATOR_FUSION_POLICY"):
        build_policy_surface(PolicySurfaceConfig(fusion="nope"))

    with pytest.raises(ValueError, match="ORCHESTRATOR_RESIDENCY_POLICY"):
        build_policy_surface(PolicySurfaceConfig(residency="nope"))

    with pytest.raises(ValueError, match="ORCHESTRATOR_SERVICE_FAMILY_POLICY"):
        build_policy_surface(PolicySurfaceConfig(service_family="nope"))


def test_a_fusion_name_is_not_selectable_at_the_residency_hook() -> None:
    """Each hook admits only the policies that answer it."""
    with pytest.raises(ValueError, match="ORCHESTRATOR_RESIDENCY_POLICY"):
        build_policy_surface(PolicySurfaceConfig(residency=FusionVetoPolicy.name))


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
    refined = derived.model_copy(
        update={"service_family": "smuggled", "required": False, "warmth": "warm"}
    )

    screened = screen_residency(derived, refined)

    assert screened.service_family == "fam-a"
    assert screened.required is True
    assert screened.warmth == "warm"
