"""The deployment's advisory policy surface over lowering, placement, and sealed state.

Every hook refines a choice the fabric has already found legal. Capacity admission,
claim and attachment minting, and fence relaxation belong to the fabric; a policy
chooses among the alternatives it already allows, and each answer is screened at its
call site. So an enabled policy changes which legal alternative runs, never what a
workflow declares, the effects it has, or how it recovers.
"""

from .lowering import (
    EpisodeAnnotation,
    LoweringPolicy,
    screen_residency,
    screen_service_family,
)
from .placement import InstanceStateLocality, PlacementContext, PlacementPolicy
from .state_control import (
    RecencyWarmth,
    SealedGenerationEvidence,
    StateControlDecision,
    StateControlPolicy,
    StateControlVerb,
    eligible_verbs,
    screen,
)
from .surface import PolicySurface, build_policy_surface

__all__ = [
    "EpisodeAnnotation",
    "InstanceStateLocality",
    "LoweringPolicy",
    "PlacementContext",
    "PlacementPolicy",
    "PolicySurface",
    "RecencyWarmth",
    "SealedGenerationEvidence",
    "StateControlDecision",
    "StateControlPolicy",
    "StateControlVerb",
    "build_policy_surface",
    "eligible_verbs",
    "screen",
    "screen_residency",
    "screen_service_family",
]
