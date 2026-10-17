"""Registering this worker's multipart uploads with control before any part is sent.

Each registration goes up as a worker event and waits for control's verdict on that
exact UploadId, which control relays back over the worker's attachment. A verdict that
does not come is asked for again, since control answers a repeated registration with
the one it already recorded; one that never comes leaves the upload unregistered.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from shared.content.multipart import (
    MultipartDoomed,
    MultipartUnowned,
    MultipartUnregistered,
    MultipartVerdict,
    current_finalization,
)


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
        attempts: int = 3,
        wait_sec: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._send = send
        self._attempts = attempts
        self._wait_sec = wait_sec
        self._logger = logger or logging.getLogger("content-multipart")
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    def register(self, task_id: str, scope: str, key: str, upload_id: str) -> None:
        """Register an upload a task is about to send parts of under ``scope``.

        Returns once control acknowledges it; raises ``MultipartDoomed`` for an
        UploadId control is aborting, ``MultipartUnowned`` when the task owns no upload
        in that scope, and ``MultipartUnregistered`` when no verdict came.
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
            for _ in range(self._attempts):
                try:
                    self._send(task_id, payload)
                except Exception:
                    self._logger.warning(
                        "could not send the registration of upload %s",
                        upload_id,
                        exc_info=True,
                    )
                if pending.answered.wait(self._wait_sec):
                    break
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
        raise MultipartUnregistered(f"upload {upload_id} of {key} was not registered")

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


__all__ = ["MultipartRegistrar"]
