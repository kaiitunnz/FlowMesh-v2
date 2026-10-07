"""The runtime-visible effect of an engine transition, and the errors it raises."""

from dataclasses import dataclass, field
from typing import Self


def dependency_failed(task_id: str) -> str:
    """The reason a task fails for when a failure it depends on cascades into it."""
    return f"Dependency {task_id} failed"


class RegionError(ValueError):
    """Raised when a structured-region operation is invalid, e.g. a child after seal."""


@dataclass
class Advance:
    """Runtime-visible effect of an engine transition, in legacy task ids.

    ``ready`` work items become admissible for a new attempt, ``failed`` ones settle
    terminally and cascade, and ``retry`` reissues an existing work item as a fresh
    attempt under its stable identity; ``cancelled`` lists the children a residual
    policy cancelled. Control settlement and dynamic child materialization are internal
    and never appear here.
    """

    ready: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    retry: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    def extend(self, other: "Advance") -> Self:
        self.ready.extend(other.ready)
        self.failed.extend(other.failed)
        self.retry.extend(other.retry)
        self.cancelled.extend(other.cancelled)
        return self
