"""The worker-local rendezvous for a held model turn's one-use egress permit.

A synchronous-turn-only facade holds a Codex turn across an in-turn model call: it
registers a waiter for the held occurrence, proposes the request digest, and blocks here
until the control plane relays the one-use permit — or a denial — over the mediated
attachment. Registering the waiter before the propose is emitted closes the race where a
fast permit would arrive before the waiter exists. The wait is bounded and cancellable,
so a permit that never arrives, a denial, or a cancelled turn ends it rather than hang.
"""

import queue
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from shared.tools.contract import MediatedOperationPermit

_BoundaryKey = tuple[str, str]

# How long a held occurrence stays recognizable after its waiter is registered, so a
# permit that arrives once the held turn has already timed out is still identified as a
# stale held-turn permit rather than a durable-yield one.
_HELD_KEY_TTL_SEC = 300.0


@dataclass(frozen=True)
class PermitDenied:
    """A control-plane denial of a held model turn: terminal and non-retryable."""

    reason: str


PermitDelivery = MediatedOperationPermit | PermitDenied


class ModelTurnRendezvous:
    """Hand a held model turn's permit from the attachment to its waiting facade."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiters: dict[_BoundaryKey, queue.Queue[PermitDelivery]] = {}
        # Occurrences a held waiter was registered for, retained past the waiter's exit
        # so a late permit is still known as a held-turn permit; expiry epoch per key.
        self._held: dict[_BoundaryKey, float] = {}

    def register(self, agent_task_id: str, call_correlation: str) -> "PermitWaiter":
        """Arm a waiter for one held occurrence before its propose is emitted."""
        key = (agent_task_id, call_correlation)
        box: queue.Queue[PermitDelivery] = queue.Queue(maxsize=1)
        with self._lock:
            self._prune_held_locked()
            self._waiters[key] = box
            self._held[key] = time.monotonic() + _HELD_KEY_TTL_SEC
        return PermitWaiter(self, key, box)

    def has_waiter(self, agent_task_id: str, call_correlation: str) -> bool:
        """Whether a held facade is waiting on this occurrence's permit."""
        with self._lock:
            return (agent_task_id, call_correlation) in self._waiters

    def was_held(self, agent_task_id: str, call_correlation: str) -> bool:
        """Whether a held waiter was recently registered for this occurrence.

        A held-turn model permit routes through the rendezvous; a durable-yield model
        permit never arms a waiter and drives the async sidecar. This distinguishes a
        stale held-turn permit (its waiter timed out) from a durable-yield one so only
        the former is dropped rather than egressed on the async lane.
        """
        with self._lock:
            self._prune_held_locked()
            return (agent_task_id, call_correlation) in self._held

    def _prune_held_locked(self) -> None:
        now = time.monotonic()
        for key in [k for k, exp in self._held.items() if exp <= now]:
            self._held.pop(key, None)

    def deliver_permit(self, permit: MediatedOperationPermit) -> bool:
        """Wake the held facade with its permit; False if no waiter is armed."""
        return self._deliver((permit.agent_task_id, permit.call_correlation), permit)

    def deliver_deny(
        self, agent_task_id: str, call_correlation: str, reason: str
    ) -> bool:
        """Wake the held facade with a terminal denial; False if no waiter is armed."""
        return self._deliver(
            (agent_task_id, call_correlation), PermitDenied(reason=reason)
        )

    def _deliver(self, key: _BoundaryKey, delivery: PermitDelivery) -> bool:
        with self._lock:
            box = self._waiters.get(key)
        if box is None:
            return False
        try:
            box.put_nowait(delivery)
        except queue.Full:
            return False
        return True

    def _discard(self, key: _BoundaryKey) -> None:
        with self._lock:
            self._waiters.pop(key, None)


class PermitWaiter:
    """A one-shot handle a held facade blocks on for its permit or denial."""

    def __init__(
        self,
        rendezvous: ModelTurnRendezvous,
        key: _BoundaryKey,
        box: "queue.Queue[PermitDelivery]",
    ) -> None:
        self._rendezvous = rendezvous
        self._key = key
        self._box = box

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._rendezvous._discard(self._key)

    def await_permit(self, timeout: float) -> PermitDelivery | None:
        """Block for the permit or denial; None on timeout. Waiter clears on exit."""
        try:
            return self._box.get(timeout=timeout)
        except queue.Empty:
            return None
