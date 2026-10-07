"""The static dependency graph of a runtime's tasks."""

from collections import defaultdict


class StaticDag:
    """Holds each task's unmet dependencies and the tasks waiting on each one."""

    def __init__(self) -> None:
        self.pending_deps: dict[str, set[str]] = {}
        self.dependents: dict[str, set[str]] = defaultdict(set)
