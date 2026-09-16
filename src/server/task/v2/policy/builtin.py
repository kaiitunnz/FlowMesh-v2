"""The advisory lowering policies the fabric ships.

Each refines one choice the compiler has already found legal, so the physical
realization changes while the logical template and its declared contract do not. A
deployment selects one per hook by name; the conservative policy is the default at
every hook.
"""

from ..representations.operators import (
    LeafOperator,
    LogicalOperator,
    RecoveryClass,
)
from ..representations.plan import ResidencyIntent, ResidencyWarmth
from .lowering import FusionPolicy, ResidencyPolicy


def _recomputes(op: LogicalOperator) -> bool:
    return isinstance(op, LeafOperator) and op.profile.recovery is (
        RecoveryClass.RECOMPUTE
    )


class RecomputeOnlyFusion(FusionPolicy):
    """Fuses two operators only where both recover by recomputation.

    An episode recovers as a unit, so an operator that recovers from a recorded output
    would be recomputed along with the segment it is folded into. Keeping it in its own
    episode holds the recovery granularity its profile declares, at the cost of one more
    boundary; an operator that recomputes either way loses nothing by fusing.
    """

    name = "recompute-only"

    def fuse(self, predecessor: LogicalOperator, candidate: LogicalOperator) -> bool:
        return _recomputes(predecessor) and _recomputes(candidate)


class WarmRetention(ResidencyPolicy):
    """Prefers a warm resident family for a required, unconditional dependency.

    Warmth is a retention preference a family definition carries; it allocates no
    capacity and changes no claim, admission, route, or credit.
    """

    name = "warm-retention"

    def residency(self, intent: ResidencyIntent) -> ResidencyIntent:
        if not intent.required or intent.conditional:
            return intent
        return intent.model_copy(update={"warmth": ResidencyWarmth.WARM})
