"""The runtime-visible effect of an engine transition, and the errors it raises."""

from dataclasses import dataclass, field
from typing import Self


def dependency_failed(task_id: str) -> str:
    """The reason a task fails for when a failure it depends on cascades into it."""
    return f"Dependency {task_id} failed"


def legacy_control_unsupported(operator_id: str, missing: str) -> str:
    """The reason a branch or loop stored without its runnable contract fails for."""
    return f"LegacyControlRegionUnsupported: {operator_id} was stored without {missing}"


class RegionError(ValueError):
    """Raised when a structured-region operation is invalid, e.g. a child after seal."""


@dataclass(frozen=True)
class Materialization:
    """A work item inside a region definition that needs a task to run on.

    Names the task id it runs under, the operator whose blueprint the task follows,
    and the context and time it runs at.
    """

    task_id: str
    work_item_id: str
    operator_id: str
    occurrence: str
    context_id: str
    time: tuple[tuple[str, int], ...] = ()


@dataclass
class Advance:
    """Runtime-visible effect of an engine transition, in legacy task ids.

    ``ready`` work items become admissible for a new attempt, ``failed`` ones settle
    terminally and cascade, and ``retry`` reissues an existing work item as a fresh
    attempt under its stable identity; ``cancelled`` lists the children a residual
    policy cancelled. ``materialized`` names the region-definition work items that need
    a task before they run, and ``skipped`` the existing tasks a dead route settles
    without running. Control settlement is internal and never appears here.
    """

    ready: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    retry: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    materialized: list[Materialization] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def extend(self, other: "Advance") -> Self:
        self.ready.extend(other.ready)
        self.failed.extend(other.failed)
        self.retry.extend(other.retry)
        self.cancelled.extend(other.cancelled)
        self.materialized.extend(other.materialized)
        self.skipped.extend(other.skipped)
        return self
