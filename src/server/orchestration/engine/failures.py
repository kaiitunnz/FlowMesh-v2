"""Declared-failure facts of one workflow instance."""

from ..journal import LedgerJournal, TrackedDict, TrackedSet

DECLARED_FAILURE_REASON = "declared-failure obligation"


class FailureLedger:
    """Holds which control regions and child-init scopes settled as a declared failure,
    and why each task failed."""

    def __init__(self, journal: LedgerJournal) -> None:
        # Control operators settled as a declared failure; a late record from another
        # input never fires one.
        self.failed_regions: TrackedSet[str] = TrackedSet(journal, "failed_regions")
        # Child-init scopes a failed agent opened and that had not released: each
        # one's join never releases.
        self.failed_scopes: TrackedSet[str] = TrackedSet(journal, "failed_scopes")
        # Why each task settled as a declared failure: its own reason, or the failure
        # it depends on.
        self.failure_reasons: TrackedDict[str, str] = TrackedDict(
            journal, "failure_reasons"
        )
        # The declared failures not yet applied to their task records.
        self.unapplied: dict[str, None] = {}
        # Why the whole instance failed, once it has.
        self.instance_failure: str | None = None
        # The first fault of a control occurrence's own, once one has faulted.
        self.control_failure: str | None = None
        # Whether the whole instance was cancelled.
        self.instance_cancelled = False

    def note_control_failure(self, reason: str) -> None:
        self.control_failure = self.control_failure or reason

    def mark_region_failed(self, operator_id: str) -> None:
        self.failed_regions.add(operator_id)

    def region_failed(self, operator_id: str) -> bool:
        return operator_id in self.failed_regions

    def mark_scope_failed(self, scope_id: str) -> None:
        self.failed_scopes.add(scope_id)

    def scope_failed(self, scope_id: str) -> bool:
        return scope_id in self.failed_scopes

    def name_failures(self, failed: list[str], reason: str) -> None:
        for task_id in failed:
            if task_id not in self.failure_reasons:
                self.failure_reasons[task_id] = reason
                self.unapplied[task_id] = None

    def failure_reason(self, task_id: str) -> str | None:
        """Why a task settled as a declared failure, or None for one that has not."""
        return self.failure_reasons.get(task_id)

    def unapplied_failures(self) -> list[tuple[str, str]]:
        """Return each task settled as a declared failure whose record is owed that
        failure, with why."""
        return [(task_id, self.failure_reasons[task_id]) for task_id in self.unapplied]

    def mark_applied(self, task_id: str) -> None:
        self.unapplied.pop(task_id, None)
