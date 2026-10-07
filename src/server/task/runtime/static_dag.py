"""The static dependency graph of a runtime's tasks."""

from collections import defaultdict


class StaticDag:
    """Holds each task's unmet dependencies and the tasks waiting on each one."""

    def __init__(self) -> None:
        self.pending_deps: dict[str, set[str]] = {}
        self.dependents: dict[str, set[str]] = defaultdict(set)

    def forget_pending(self, task_id: str) -> None:
        self.pending_deps.pop(task_id, None)

    def take_dependents(self, task_id: str) -> set[str]:
        return self.dependents.pop(task_id, set())

    def discard_dependency(self, task_id: str, dependency: str) -> None:
        self.pending_deps[task_id].discard(dependency)
