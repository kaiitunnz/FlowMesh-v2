"""Advisory hooks over where a ready episode is placed.

A placement policy names workers it prefers among the candidates a dispatch has already
narrowed to. The dispatcher intersects that preference with the surviving pool, so a
policy can only ever narrow: a worker excluded by owner affinity, by a selected-worker
hint, or by having failed the task is unreachable through this surface, and an empty
preference leaves the pool as it was.
"""

from dataclasses import dataclass, field

from ..task.v2.representations.plan import EpisodeSpec


@dataclass(frozen=True)
class PlacementContext:
    """What a placement policy reads about one dispatch.

    ``candidates`` are the workers still eligible after the dispatch's hard filters.
    ``state_generation`` and ``state_owner`` describe the task's own private-state
    binding — an owner-bound generation leaves one candidate, so locality is a choice
    only while a lineage has yet to seal. ``instance_state_holders`` are the workers
    holding sealed state of the same workflow instance.
    """

    task_id: str
    candidates: tuple[str, ...]
    episode: EpisodeSpec | None = None
    state_generation: int | None = None
    state_owner: str | None = None
    instance_state_holders: frozenset[str] = field(default_factory=frozenset)


class PlacementPolicy:
    """The dispatcher's advisory placement hook, expressing no preference."""

    name = "none"

    def prefer(self, context: PlacementContext) -> frozenset[str]:
        """The worker ids this policy prefers, empty for no preference."""
        return frozenset()


class InstanceStateLocality(PlacementPolicy):
    """Prefers workers that already hold sealed private state of the same instance.

    Keeping an instance's activations on the hosts that hold its state narrows where
    that instance's lineages come to live, which is a choice worth making while a
    lineage has yet to seal and an owner fence has yet to bind it.
    """

    name = "instance_state_locality"

    def prefer(self, context: PlacementContext) -> frozenset[str]:
        return context.instance_state_holders
