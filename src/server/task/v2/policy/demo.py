"""Fixed demonstration policies for the advisory lowering surface.

Each refines one choice the compiler has already found legal, so the physical
realization changes while the logical template and its declared contract do not.
A deployment selects one per hook by name; none is the default.
"""

from shared.tasks import TaskType

from ..representations.operators import LeafOperator, LogicalOperator
from ..representations.plan import WARM, ResidencyIntent
from .lowering import FusionPolicy, ResidencyPolicy


def _is_echo(op: LogicalOperator) -> bool:
    return (
        isinstance(op, LeafOperator) and op.profile.binding.task_type is TaskType.ECHO
    )


class FusionVetoPolicy(FusionPolicy):
    """Keeps a fusible ``echo`` leaf out of its predecessor's episode.

    The compiler asks only about pairs it has already proved pure, deterministic,
    and local, so the veto cuts one more episode within the same contract.
    """

    name = "demo-fusion"

    def fuse(self, predecessor: LogicalOperator, candidate: LogicalOperator) -> bool:
        return not _is_echo(candidate)


class WarmthPolicy(ResidencyPolicy):
    """Prefers a warm resident family for a required, unconditional dependency.

    Warmth is a retention preference a family definition carries; it allocates no
    capacity and changes no claim, admission, route, or credit.
    """

    name = "demo-warmth"

    def residency(self, intent: ResidencyIntent) -> ResidencyIntent:
        if not intent.required or intent.conditional:
            return intent
        return intent.model_copy(update={"warmth": WARM})
