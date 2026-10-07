"""Declared-failure facts of one workflow instance."""

_DECLARED_FAILURE_REASON = "declared-failure obligation"


_DECLARED_FAILURE_REASON = "declared-failure obligation"


_DECLARED_FAILURE_REASON = "declared-failure obligation"


_DECLARED_FAILURE_REASON = "declared-failure obligation"


_DECLARED_FAILURE_REASON = "declared-failure obligation"


_DECLARED_FAILURE_REASON = "declared-failure obligation"


_DECLARED_FAILURE_REASON = "declared-failure obligation"


class FailureLedger:
    """Holds which control regions and child-init scopes settled as a declared failure,
    and why each task failed."""

    def __init__(self) -> None:
        # Control operators settled as a declared failure; a late record from another
        # input never fires one.
        self.failed_regions: set[str] = set()
        # Child-init scopes a failed agent opened and that had not released: each
        # one's join never releases.
        self.failed_scopes: set[str] = set()
        # Why each task settled as a declared failure: its own reason, or the failure
        # it depends on.
        self.failure_reasons: dict[str, str] = {}

    def mark_region_failed(self, operator_id: str) -> None:
        self.failed_regions.add(operator_id)

    def region_failed(self, operator_id: str) -> bool:
        return operator_id in self.failed_regions

    def mark_scope_failed(self, scope_id: str) -> None:
        self.failed_scopes.add(scope_id)

    def scope_failed(self, scope_id: str) -> bool:
        return scope_id in self.failed_scopes

    def name_failure(self, task_id: str, reason: str) -> None:
        """Name why a task failed, unless an earlier failure named it."""
        self.failure_reasons.setdefault(task_id, reason)

    def name_failures(self, failed: list[str], reason: str) -> None:
        for task_id in failed:
            self.failure_reasons.setdefault(task_id, reason)

    def failure_reason(self, task_id: str) -> str | None:
        """Why a task settled as a declared failure, or None for one that has not."""
        return self.failure_reasons.get(task_id)

    def declared_failures(self) -> dict[str, str]:
        """Every task settled as a declared failure, with why."""
        return self.failure_reasons
