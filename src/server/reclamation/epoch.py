"""The root epoch: which root incarnation may still commit asserted bindings.

A starting root takes a new epoch before it reconciles anything an earlier root left.
A guarded commit watches the epoch key on the connection that executes it, so a
transaction an earlier root issued lands only if no later root has started, and the
later root's reconciliation then sees it.
"""

from dataclasses import dataclass


class EpochFenced(RuntimeError):
    """A later root took the epoch: this root may no longer commit asserted state."""


class GuardInterrupted(ConnectionError):
    """A guarded commit lost its watch without the epoch changing; it may be retried."""


@dataclass(frozen=True)
class CommitGuard:
    """Commit only while the root epoch is still ``epoch``."""

    epoch: int


__all__ = ["CommitGuard", "EpochFenced", "GuardInterrupted"]
