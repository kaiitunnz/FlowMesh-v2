"""Advisory hooks over the physical lowering of a compiled template.

A lowering policy refines choices the compiler has already found legal: it may veto a
fusion, ask for an extra durable checkpoint, annotate an episode, steer a service
dependency to a compatible family, and express residency preference. The compiler
screens every answer, so a policy widens nothing: fusion stays within the pure,
deterministic, local set, a boundary only ever cuts more often, and a family
refinement holds the dependency's engine-batch key and isolation.
"""

from dataclasses import dataclass

from ..task.v2.representations.operators import LogicalOperator
from ..task.v2.representations.plan import (
    EpisodeBoundaryKind,
    ResidencyIntent,
    ServiceFamilyRequirement,
)


@dataclass(frozen=True)
class EpisodeAnnotation:
    """The lightweight annotations a policy attaches to an episode."""

    liveness_key: str | None = None
    speculative_eligible: bool = False


class LoweringPolicy:
    """The lowerer's advisory hook surface, with conservative defaults.

    Each method answers for one operator of the template being compiled. The defaults
    reproduce the compiler's own choices, so an unconfigured deployment and a policy
    that overrides nothing lower identically.
    """

    name = "conservative"

    def fuse(self, head: LogicalOperator, candidate: LogicalOperator) -> bool:
        """Whether a fusible successor joins the head operator's episode."""
        return True

    def checkpoint(self, op: LogicalOperator) -> bool:
        """Whether an operator that closes on its task boundary takes a durable one."""
        return False

    def annotate(
        self, op: LogicalOperator, boundary: EpisodeBoundaryKind
    ) -> EpisodeAnnotation:
        """The liveness and speculative-eligibility annotations for an episode."""
        return EpisodeAnnotation()

    def service_family(
        self, requirement: ServiceFamilyRequirement
    ) -> ServiceFamilyRequirement:
        """The family a service dependency binds, among the compatible ones."""
        return requirement

    def residency(self, intent: ResidencyIntent) -> ResidencyIntent:
        """The warmth, reuse, affinity, and preemption preference for a dependency."""
        return intent


def screen_service_family(
    derived: ServiceFamilyRequirement, refined: ServiceFamilyRequirement
) -> ServiceFamilyRequirement:
    """The refined requirement when it is compatible with the derived one.

    Engine-batch key and isolation are the dependency's compatibility key: a refinement
    that moves either would serve the invocation from an incompatible family, so only
    the family name is a policy's to choose.
    """
    if (
        refined.engine_batch_key != derived.engine_batch_key
        or refined.isolation != derived.isolation
    ):
        return derived
    return refined


def screen_residency(
    derived: ResidencyIntent, refined: ResidencyIntent
) -> ResidencyIntent:
    """The refined intent under the compiler's own family and requiredness.

    A required resident dependency is pinned by the template's binding; a policy carries
    preference only, so warmth, reuse domain, affinity, and preemption are taken from
    the refinement and the rest from the plan.
    """
    return derived.model_copy(
        update={
            "warmth": refined.warmth,
            "reuse_domain": refined.reuse_domain,
            "affinity": refined.affinity,
            "preemption": refined.preemption,
        }
    )
