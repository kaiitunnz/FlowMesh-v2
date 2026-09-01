"""The materialization policy gate produces typed allow/deny outcomes.

These prove an unlisted model, an exhausted replica quota, and a saturated cold-start
limit each yield a distinct typed denial, while an allowed model within limits is
admitted — none of which itself creates a replica or a credit.
"""

from server.resident import (
    ProvisioningDenialReason,
    ResidentPolicyLimits,
    decide_materialization,
)


def test_allows_within_limits():
    decision = decide_materialization(
        model_ref="m",
        limits=ResidentPolicyLimits(),
        active_replicas=0,
        materializing_replicas=0,
    )
    assert decision.allowed and decision.reason is None


def test_empty_allowlist_permits_any_model():
    decision = decide_materialization(
        model_ref="anything",
        limits=ResidentPolicyLimits(allowed_models=frozenset()),
        active_replicas=0,
        materializing_replicas=0,
    )
    assert decision.allowed


def test_denies_unlisted_model():
    decision = decide_materialization(
        model_ref="secret-model",
        limits=ResidentPolicyLimits(allowed_models=frozenset({"allowed-model"})),
        active_replicas=0,
        materializing_replicas=0,
    )
    assert not decision.allowed
    assert decision.reason is ProvisioningDenialReason.MODEL_NOT_ALLOWED


def test_denies_over_family_quota():
    decision = decide_materialization(
        model_ref="m",
        limits=ResidentPolicyLimits(max_replicas_per_family=1),
        active_replicas=1,
        materializing_replicas=0,
    )
    assert not decision.allowed
    assert decision.reason is ProvisioningDenialReason.QUOTA_EXCEEDED


def test_denies_over_cold_start_limit():
    decision = decide_materialization(
        model_ref="m",
        limits=ResidentPolicyLimits(
            max_replicas_per_family=4, max_concurrent_cold_starts=1
        ),
        active_replicas=0,
        materializing_replicas=1,
    )
    assert not decision.allowed
    assert decision.reason is ProvisioningDenialReason.COLD_START_LIMIT
