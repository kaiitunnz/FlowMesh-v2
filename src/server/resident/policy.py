"""The submission/admission policy gate for auto-materialization.

Before the Lifecycle & scale manager creates or joins an allocation for a plan-derived
family, policy checks the allowed model/engine catalog, per-family replica quota, and
the bounded concurrent cold-start limit. A refusal is a typed, durable outcome that
creates no allocation, route authorization, or credit and never silently substitutes an
arbitrary provider. Thresholds here are the conservative baseline.
"""

from dataclasses import dataclass, field

from .state import ProvisioningDenialReason


@dataclass(frozen=True)
class ResidentPolicyLimits:
    """Deployment-resolved caps for resident auto-provisioning.

    ``allowed_models`` empty means the deployment allows any plan-derived model; a non-
    empty set enforces an explicit catalog. The manager receives resolved values from
    the config edge rather than reading the environment.
    """

    allowed_models: frozenset[str] = field(default_factory=frozenset)
    max_replicas_per_family: int = 1
    max_concurrent_cold_starts: int = 1
    cold_start_deadline_sec: float = 300.0


@dataclass(frozen=True)
class ProvisioningDecision:
    """The result of the materialization policy gate."""

    allowed: bool
    reason: ProvisioningDenialReason | None = None
    detail: str | None = None

    @classmethod
    def allow(cls) -> "ProvisioningDecision":
        return cls(allowed=True)

    @classmethod
    def deny(
        cls, reason: ProvisioningDenialReason, detail: str
    ) -> "ProvisioningDecision":
        return cls(allowed=False, reason=reason, detail=detail)


def decide_materialization(
    *,
    model_ref: str,
    limits: ResidentPolicyLimits,
    active_replicas: int,
    materializing_replicas: int,
) -> ProvisioningDecision:
    """Whether a family may materialize one more replica now.

    ``active_replicas`` counts live incarnations (warm, busy, materializing, draining);
    ``materializing_replicas`` counts in-flight cold starts against the concurrency cap.
    """
    if limits.allowed_models and model_ref not in limits.allowed_models:
        return ProvisioningDecision.deny(
            ProvisioningDenialReason.MODEL_NOT_ALLOWED,
            f"model {model_ref!r} is not in the allowed catalog",
        )
    if active_replicas >= limits.max_replicas_per_family:
        return ProvisioningDecision.deny(
            ProvisioningDenialReason.QUOTA_EXCEEDED,
            f"family already at its replica quota of {limits.max_replicas_per_family}",
        )
    if materializing_replicas >= limits.max_concurrent_cold_starts:
        return ProvisioningDecision.deny(
            ProvisioningDenialReason.COLD_START_LIMIT,
            f"at the cold-start limit of {limits.max_concurrent_cold_starts}",
        )
    return ProvisioningDecision.allow()
