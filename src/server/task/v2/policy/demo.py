"""Fixed demonstration policies for the advisory lowering surface.

Each refines one choice the compiler has already found legal, so the physical
realization changes while the logical template and its declared contract do not.
A deployment selects one by name; none is the default.
"""

from shared.tasks import TaskType

from ..representations.operators import LeafOperator, LogicalOperator
from ..representations.plan import WARM, ResidencyIntent
from .lowering import LoweringPolicy


def _is_echo(op: LogicalOperator) -> bool:
    return (
        isinstance(op, LeafOperator) and op.profile.binding.task_type is TaskType.ECHO
    )


class FusionVetoPolicy(LoweringPolicy):
    """Keeps a fusible ``echo`` leaf out of its predecessor's episode.

    The compiler asks only about pairs it has already proved pure, deterministic,
    and local, so the veto cuts one more episode within the same contract.
    """

    name = "d30-fusion"

    def fuse(self, predecessor: LogicalOperator, candidate: LogicalOperator) -> bool:
        return not _is_echo(candidate)


class WarmthPolicy(LoweringPolicy):
    """Prefers a warm resident family for a required, unconditional dependency.

    Warmth is a retention preference a family definition carries; it allocates no
    capacity and changes no claim, admission, route, or credit.
    """

    name = "d30-warmth"

    def residency(self, intent: ResidencyIntent) -> ResidencyIntent:
        if not intent.required or intent.conditional:
            return intent
        return intent.model_copy(update={"warmth": WARM})


class DemoPolicy(FusionVetoPolicy, WarmthPolicy):
    """Both fixed refinements under one selectable name."""

    name = "d30-demo"
