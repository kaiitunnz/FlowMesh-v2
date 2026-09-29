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
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from shared.tools.contract import MediatedOperationPermit

_BoundaryKey = tuple[str, str]

# How long a held occurrence stays recognizable after its waiter is registered, so a
# permit that arrives once the held turn has already timed out is still identified as a
# stale held-turn permit rather than a durable-yield one.
_HELD_KEY_TTL_SEC = 300.0
# How many given-up episodes stay refused; one given up this long ago has no harness
# left to make a call.
_MAX_REFUSED_EPISODES = 1024


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
        # Episodes refused until they reopen, so a held turn's later round cannot arm a
        # waiter and propose once its episode is being given up; a released one's
        # waiters are armed already denied with the reason recorded here.
        self._refused: dict[str, str | None] = {}
        # The registration each open episode's held turns run under; a turn of any
        # other registration of the task is stale.
        self._open: dict[str, str] = {}

    def register(
        self,
        agent_task_id: str,
        call_correlation: str,
        episode: str,
        on_armed: Callable[[], None],
    ) -> "PermitWaiter":
        """Arm a waiter for one held occurrence of the ``episode`` registration
        before its propose is emitted, running ``on_armed`` as it is armed.

        A released episode's turn gets a waiter already denied, and a turn of a
        registration no longer open gets a stale one; neither arms anything, so it can
        neither take a live turn's waiter nor stash over its request. A refused
        episode's waiter is armed refused, without running ``on_armed``.
        """
        key = (agent_task_id, call_correlation)
        box: queue.Queue[PermitDelivery] = queue.Queue(maxsize=1)
        with self._lock:
            if (reason := self._refused.get(agent_task_id)) is not None:
                box.put_nowait(PermitDenied(reason=reason))
                return PermitWaiter(self, key, box, refused=True)
            if self._open.get(agent_task_id) != episode:
                return PermitWaiter(self, key, box, stale=True)
            self._prune_held_locked()
            self._waiters[key] = box
            self._held[key] = time.monotonic() + _HELD_KEY_TTL_SEC
            refused = agent_task_id in self._refused
            if not refused:
                on_armed()
        return PermitWaiter(self, key, box, refused=refused)

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
        """Wake the held facade with its permit; False if no waiter is armed or the
        episode is refused, which egresses nothing once it is being given up."""
        return self._deliver(
            (permit.agent_task_id, permit.call_correlation), permit, refusable=True
        )

    def deliver_deny(
        self, agent_task_id: str, call_correlation: str, reason: str
    ) -> bool:
        """Wake the held facade with a terminal denial; False if no waiter is armed."""
        return self._deliver(
            (agent_task_id, call_correlation), PermitDenied(reason=reason)
        )

    def refuse(self, agent_task_id: str) -> None:
        """Arm the episode's later waiters refused until it reopens."""
        with self._lock:
            self._refuse_locked(agent_task_id, self._refused.get(agent_task_id))

    def release(self, agent_task_id: str, reason: str) -> None:
        """Wake every held waiter of one episode with a terminal denial, and deny the
        waiters it arms until it reopens."""
        with self._lock:
            self._refuse_locked(agent_task_id, reason)
            boxes = [
                box for key, box in self._waiters.items() if key[0] == agent_task_id
            ]
        for box in boxes:
            try:
                box.put_nowait(PermitDenied(reason=reason))
            except queue.Full:
                pass

    def reopen(self, agent_task_id: str, episode: str) -> None:
        """Open the ``episode`` registration of a task, whose held turns alone arm
        waiters from now on."""
        with self._lock:
            self._open[agent_task_id] = episode
            self._refused.pop(agent_task_id, None)

    def close(self, agent_task_id: str, episode: str) -> None:
        """Close the ``episode`` registration, if it is still the task's open one."""
        with self._lock:
            if self._open.get(agent_task_id) == episode:
                del self._open[agent_task_id]

    def _refuse_locked(self, agent_task_id: str, reason: str | None) -> None:
        self._refused.pop(agent_task_id, None)
        self._refused[agent_task_id] = reason
        while len(self._refused) > _MAX_REFUSED_EPISODES:
            del self._refused[next(iter(self._refused))]

    def _deliver(
        self, key: _BoundaryKey, delivery: PermitDelivery, refusable: bool = False
    ) -> bool:
        with self._lock:
            if refusable and key[0] in self._refused:
                return False
            box = self._waiters.get(key)
        if box is None:
            return False
        try:
            box.put_nowait(delivery)
        except queue.Full:
            return False
        return True

    def _discard(self, key: _BoundaryKey, box: "queue.Queue[PermitDelivery]") -> None:
        # A later waiter for the same occurrence may have replaced this one.
        with self._lock:
            if self._waiters.get(key) is box:
                del self._waiters[key]


class PermitWaiter:
    """A one-shot handle a held facade blocks on for its permit or denial."""

    def __init__(
        self,
        rendezvous: ModelTurnRendezvous,
        key: _BoundaryKey,
        box: "queue.Queue[PermitDelivery]",
        refused: bool = False,
        stale: bool = False,
    ) -> None:
        self._rendezvous = rendezvous
        self._key = key
        self._box = box
        self.refused = refused
        self.stale = stale

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._rendezvous._discard(self._key, self._box)

    def await_permit(self, timeout: float) -> PermitDelivery | None:
        """Block for the permit or denial; None on timeout. Waiter clears on exit."""
        try:
            return self._box.get(timeout=timeout)
        except queue.Empty:
            return None
