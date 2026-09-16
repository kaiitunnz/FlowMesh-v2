"""Advisory hooks over the physical lowering of a compiled template.

A policy refines choices the compiler has already found legal: it may keep a fusible
operator out of its predecessor's episode, steer a service dependency to a compatible
family, and express residency preference. The compiler screens every answer, so a policy
only narrows: fusion is bounded to the pure, deterministic, local set, and a family
refinement holds the dependency's engine-batch key and isolation.

There is one policy per hook, so a deployment composes the facets it wants
independently. Each hook's default reproduces the compiler's own choice, so a surface
that overrides nothing lowers identically to the compiler alone.

Choosing a worker, reserving capacity, minting a claim or attachment, and replacing a
pinned resident binding belong to the fabric; a policy picks among the alternatives the
compiler already proved legal.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..representations.plan import CONSERVATIVE_POLICY, known_warmth

if TYPE_CHECKING:
    from ..representations.operators import LogicalOperator
    from ..representations.plan import (
        ResidencyIntent,
        ServiceFamilyRequirement,
    )


class FusionPolicy:
    """Whether an operator joins the episode of the operator before it."""

    name = CONSERVATIVE_POLICY

    def fuse(
        self, predecessor: "LogicalOperator", candidate: "LogicalOperator"
    ) -> bool:
        """Whether a fusible operator joins its predecessor's episode.

        The pair is asked one predecessor and one candidate at a time, in template
        order, so an answer can rest on the two operators it names and never on what
        follows them.
        """
        return True


class ServiceFamilyPolicy:
    """Which of the families compatible with a dependency it binds."""

    name = CONSERVATIVE_POLICY

    def service_family(
        self, requirement: "ServiceFamilyRequirement"
    ) -> "ServiceFamilyRequirement":
        """The family a service dependency binds, among the compatible ones."""
        return requirement


class ResidencyPolicy:
    """The residency preference a plan-derived resident dependency carries."""

    name = CONSERVATIVE_POLICY

    def residency(self, intent: "ResidencyIntent") -> "ResidencyIntent":
        """The warmth, reuse, affinity, and preemption preference for a dependency."""
        return intent


@dataclass(frozen=True)
class PolicySurface:
    """The advisory policy a deployment runs at each lowering hook.

    Its default is the conservative policy at every hook, which the compiler consults
    like any other and which answers with the compiler's own choice.
    """

    fusion: FusionPolicy = field(default_factory=FusionPolicy)
    residency: ResidencyPolicy = field(default_factory=ResidencyPolicy)
    service_family: ServiceFamilyPolicy = field(default_factory=ServiceFamilyPolicy)


def screen_service_family(
    derived: "ServiceFamilyRequirement", refined: "ServiceFamilyRequirement"
) -> "ServiceFamilyRequirement":
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
    derived: "ResidencyIntent", refined: "ResidencyIntent"
) -> "ResidencyIntent":
    """The refined intent under the compiler's own family and requiredness.

    A required resident dependency is pinned by the template's binding; a policy carries
    preference only, so warmth, reuse domain, affinity, and preemption are taken from
    the refinement and the rest from the plan. A warmth outside the vocabulary the
    fabric expresses is dropped here, so no later reader has to interpret one.
    """
    return derived.model_copy(
        update={
            "warmth": known_warmth(refined.warmth),
            "reuse_domain": refined.reuse_domain,
            "affinity": refined.affinity,
            "preemption": refined.preemption,
        }
    )
