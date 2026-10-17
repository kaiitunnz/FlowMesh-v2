"""Registering this worker's multipart uploads with control before any part is sent.

Each registration goes up as a worker event and waits for control's verdict on that
exact UploadId, which control relays back over the worker's attachment. A verdict that
does not come is asked for again, at a pace that backs off to a cap, since control
answers a repeated registration with the one it already recorded. The write holds its
source and keeps asking through a control outage of any length, so an outage delays a
write rather than failing it; only the write's cancellation, or the registration it
was asked under ending, gives it up.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from shared.content import WriteCancelled
from shared.content.multipart import (
    MultipartDoomed,
    MultipartUnowned,
    MultipartVerdict,
    current_finalization,
)

# How often a write waiting on a verdict checks whether it was cancelled.
_CANCEL_POLL_SEC = 0.5

type Waiter = Callable[[threading.Event, float], bool]
"""Wait up to a timeout for an event; returns whether it is set."""


def _wait(event: threading.Event, timeout: float) -> bool:
    return event.wait(timeout)


class _Pending:
    def __init__(self) -> None:
        self.answered = threading.Event()
        self.verdict: MultipartVerdict | None = None


class MultipartRegistrar:
    """Registers uploads and hands each control verdict to the write waiting on it.

    ``send`` carries one task's registration payload to control.
    """

    def __init__(
        self,
        send: Callable[[str, dict[str, Any]], None],
        *,
        resend_sec: float = 5.0,
        max_resend_sec: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        wait: Waiter = _wait,
        logger: logging.Logger | None = None,
    ) -> None:
        self._send = send
        self._resend_sec = resend_sec
        self._max_resend_sec = max_resend_sec
        self._clock = clock
        self._wait = wait
        self._logger = logger or logging.getLogger("content-multipart")
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    def register(
        self,
        task_id: str,
        scope: str,
        key: str,
        upload_id: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Register an upload a task is about to send parts of under ``scope``.

        Returns once control acknowledges it, asking again for as long as no verdict
        comes. Raises ``MultipartDoomed`` for an UploadId control is aborting,
        ``MultipartUnowned`` when the task owns no upload in that scope, and
        ``WriteCancelled`` once ``cancelled`` holds or the registration is abandoned.
        """
        payload: dict[str, Any] = {
            "task_id": task_id,
            "scope": scope,
            "key": key,
            "upload_id": upload_id,
        }
        if (idempotency_key := current_finalization()) is not None:
            payload["idempotency_key"] = idempotency_key
        pending = _Pending()
        with self._lock:
            self._pending[upload_id] = pending
        try:
            self._await_verdict(task_id, upload_id, payload, pending, cancelled)
        finally:
            with self._lock:
                self._pending.pop(upload_id, None)
        match pending.verdict:
            case MultipartVerdict.ACK:
                return
            case MultipartVerdict.DOOMED:
                raise MultipartDoomed(f"upload {upload_id} of {key} is being aborted")
            case MultipartVerdict.UNOWNED:
                raise MultipartUnowned(
                    f"task {task_id} owns no multipart upload in scope {scope}"
                )
        raise WriteCancelled(f"registration of upload {upload_id} of {key} ended")

    def _await_verdict(
        self,
        task_id: str,
        upload_id: str,
        payload: dict[str, Any],
        pending: _Pending,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        delay = self._resend_sec
        asked = 0
        while not pending.answered.is_set():
            if cancelled is not None and cancelled():
                return
            if asked:
                self._logger.warning(
                    "no verdict on upload %s after %d registrations; asking again",
                    upload_id,
                    asked,
                )
            try:
                self._send(task_id, payload)
            except Exception:
                self._logger.warning(
                    "could not send the registration of upload %s",
                    upload_id,
                    exc_info=True,
                )
            asked += 1
            deadline = self._clock() + delay
            while not pending.answered.is_set():
                if cancelled is not None and cancelled():
                    return
                if (remaining := deadline - self._clock()) <= 0:
                    break
                self._wait(pending.answered, min(remaining, _CANCEL_POLL_SEC))
            delay = min(delay * 2, self._max_resend_sec)

    def deliver(self, frame: dict[str, Any]) -> None:
        """Hand control's verdict on one upload to the write waiting on it."""
        try:
            verdict = MultipartVerdict(str(frame.get("verdict")))
        except ValueError:
            self._logger.warning("ignoring an unknown multipart verdict: %s", frame)
            return
        with self._lock:
            pending = self._pending.get(str(frame.get("upload_id") or ""))
            if pending is not None and not pending.answered.is_set():
                pending.verdict = verdict
                pending.answered.set()

    def abandon(self) -> None:
        """End every registration waiting now, as the registration of this worker that
        asked for them has ended; each write waiting on one is cancelled."""
        with self._lock:
            for pending in self._pending.values():
                pending.answered.set()


__all__ = ["MultipartRegistrar"]
