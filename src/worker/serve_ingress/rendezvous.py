"""The ingress's wait for control's admission decision on one external request.

A forward ingress authenticates nothing itself: it hands the presented credential and
the frozen request's descriptor to control over its authenticated attachment and waits
here for the decision. Control authenticates the principal, checks task-read access,
resolves the live binding, and either admits the request — returning the claim-bound
handoff the ingress relays to the replica's sidecar — or denies it.

The waiter is armed before the request goes up, so a decision that returns immediately
cannot arrive before anything is listening for it.
"""

import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from shared.resident.carriage import ResidentCarriagePlan
from shared.resident.contracts import AdmissionHandoff


@dataclass(frozen=True)
class ServeIngressAdmission:
    """Control admitted the request: the fence and session to relay it under.

    ``task_id`` and ``call_correlation`` are the invocation's worker-lane correlation
    that control assigned; the ingress lane echoes them on the ack and outcome it sends
    back so control matches every report to the same in-flight attempt.
    ``carriage_plan`` names the transport control selected for the attempt.
    """

    session_id: str
    task_id: str
    call_correlation: str
    handoff: AdmissionHandoff
    carriage_plan: ResidentCarriagePlan


@dataclass(frozen=True)
class ServeIngressDenied:
    """Control refused the request, with the status the client should receive."""

    status: int
    detail: str


ServeIngressDecision = ServeIngressAdmission | ServeIngressDenied


class ServeIngressWaiter:
    """One in-flight request's slot for control's decision."""

    def __init__(self) -> None:
        self._slot: queue.Queue[ServeIngressDecision] = queue.Queue(maxsize=1)

    def deliver(self, decision: ServeIngressDecision) -> None:
        try:
            self._slot.put_nowait(decision)
        except queue.Full:
            pass

    def await_decision(self, timeout: float) -> ServeIngressDecision | None:
        """The decision, or None when control did not answer in time."""
        try:
            return self._slot.get(timeout=timeout)
        except queue.Empty:
            return None


class ServeIngressRendezvous:
    """Matches control's admission decisions to the requests waiting for them."""

    def __init__(self) -> None:
        self._waiters: dict[str, ServeIngressWaiter] = {}
        self._lock = threading.Lock()

    @contextmanager
    def register(self, request_id: str) -> Iterator[ServeIngressWaiter]:
        """Arm a waiter for one request, dropping it however the request ends."""
        waiter = ServeIngressWaiter()
        with self._lock:
            self._waiters[request_id] = waiter
        try:
            yield waiter
        finally:
            with self._lock:
                self._waiters.pop(request_id, None)

    def deliver(self, request_id: str, decision: ServeIngressDecision) -> bool:
        """Hand one decision to its waiting request; False when none is waiting."""
        with self._lock:
            waiter = self._waiters.get(request_id)
        if waiter is None:
            return False
        waiter.deliver(decision)
        return True
