"""The deployment's advisory policy surface over physical lowering.

Every hook refines a choice the compiler has already found legal, and each answer is
screened at its call site. An enabled policy changes which legal lowering runs, never
what a workflow declares, the effects it has, or how it recovers.
"""

from .lowering import (
    LoweringPolicy,
    screen_residency,
    screen_service_family,
)
from .surface import PolicySurface, build_policy_surface

__all__ = [
    "LoweringPolicy",
    "PolicySurface",
    "build_policy_surface",
    "screen_residency",
    "screen_service_family",
]
