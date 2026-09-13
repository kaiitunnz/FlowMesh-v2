"""Assembly of the deployment's advisory policy surface and its screens."""

import pytest

from server.config import PolicySurfaceConfig
from server.task.v2.policy import (
    LoweringPolicy,
    build_policy_surface,
    screen_residency,
    screen_service_family,
)
from server.task.v2.representations.plan import (
    ResidencyIntent,
    ServiceFamilyRequirement,
)


def test_surface_is_none_while_disabled() -> None:
    assert build_policy_surface(PolicySurfaceConfig()) is None


def test_surface_builds_the_configured_facet() -> None:
    surface = build_policy_surface(PolicySurfaceConfig(enabled=True))

    assert surface is not None
    assert surface.lowering.name == LoweringPolicy.name


def test_the_enable_flag_reads_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_POLICY_SURFACE_ENABLED", "true")
    assert PolicySurfaceConfig.from_env().enabled is True

    monkeypatch.delenv("ORCHESTRATOR_POLICY_SURFACE_ENABLED")
    assert PolicySurfaceConfig.from_env().enabled is False


def test_the_lowering_selector_reads_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_LOWERING_POLICY", "custom")
    assert PolicySurfaceConfig.from_env().lowering == "custom"

    monkeypatch.delenv("ORCHESTRATOR_LOWERING_POLICY")
    assert PolicySurfaceConfig.from_env().lowering == "conservative"


def test_unknown_policy_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="ORCHESTRATOR_LOWERING_POLICY"):
        build_policy_surface(PolicySurfaceConfig(enabled=True, lowering="nope"))


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
