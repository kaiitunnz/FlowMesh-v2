"""The deployment's advisory policy surface over physical lowering.

Every hook refines a choice the compiler has already found legal, and each answer is
screened at its call site. A policy changes which legal lowering runs, never what a
workflow declares, the effects it has, or how it recovers.
"""

from .lowering import (
    FusionPolicy,
    PolicySurface,
    ResidencyPolicy,
    ServiceFamilyPolicy,
    screen_residency,
    screen_service_family,
)
from .surface import build_policy_surface

__all__ = [
    "FusionPolicy",
    "PolicySurface",
    "ResidencyPolicy",
    "ServiceFamilyPolicy",
    "build_policy_surface",
    "screen_residency",
    "screen_service_family",
]
