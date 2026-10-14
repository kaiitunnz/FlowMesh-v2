"""Publishing a training run's checkpoints as it saves them.

A trainer's save hands each closed checkpoint to a spool; one uploader per task
publishes each snapshot as the ``checkpoints`` artifact at the revision of its step,
one at a time, and reports each publication on the task's event stream. The task's
success lists every revision the uploader published, so a report lost on the way still
binds when the task settles.

A publication that fails is retried with backoff, since every write is conditional and
repeats settle on what landed. One that still fails marks the spool failed, which stops
the trainer at its next save, and fails the task when the uploader drains.
"""

import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from shared.content import FabricObjectStore
from shared.tasks.artifact_binding import ArtifactBinding, step_revision

from .artifacts import (
    CHECKPOINTS_DIR,
    ArtifactPublisher,
    fail_spool,
    open_spool,
    spooled_steps,
)

_POLL_SEC = 1.0
_ATTEMPTS = 5
_BACKOFF_SEC = 1.0
# How long an abort waits for an upload it interrupted to stop.
_ABORT_WAIT_SEC = 10.0


class CheckpointPublishFailed(RuntimeError):
    """A checkpoint the trainer saved could not be published."""


class CheckpointUploader:
    """Publishes the snapshots appearing in one task's checkpoint spool."""

    def __init__(
        self,
        store: FabricObjectStore,
        scope: str,
        spool_dir: Path,
        report: Callable[[ArtifactBinding], None],
        logger: logging.Logger,
        cancelled: Callable[[], bool] = lambda: False,
        poll_sec: float = _POLL_SEC,
        backoff_sec: float = _BACKOFF_SEC,
    ) -> None:
        self._task_cancelled = cancelled
        self._publisher = ArtifactPublisher(store, scope, cancelled=self._cancelled)
        self._spool_dir = spool_dir
        self._report = report
        self._logger = logger
        self._poll_sec = poll_sec
        self._backoff_sec = backoff_sec
        self._published: dict[int, ArtifactBinding] = {}
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._aborted = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="checkpoint-uploader", daemon=True
        )

    def start(self) -> None:
        open_spool(self._spool_dir)
        self._thread.start()

    def drain(self) -> list[ArtifactBinding]:
        """Publish what is still spooled and return every revision published.

        Raises ``CheckpointPublishFailed`` when a checkpoint could not be published.
        """
        self._stop.set()
        self._thread.join()
        if self._error is None:
            self._sweep()
        error = self._error
        self._clear()
        if error is not None:
            raise CheckpointPublishFailed(str(error)) from error
        return [self._published[step] for step in sorted(self._published)]

    def abort(self) -> None:
        """Stop without publishing what is still spooled, interrupting an upload."""
        self._aborted.set()
        self._stop.set()
        self._thread.join(_ABORT_WAIT_SEC)
        self._clear()

    def _cancelled(self) -> bool:
        return self._aborted.is_set() or self._task_cancelled()

    def _interrupted_within(self, delay: float) -> bool:
        deadline = time.monotonic() + delay
        while not self._cancelled():
            if (left := deadline - time.monotonic()) <= 0:
                return False
            self._aborted.wait(min(left, self._poll_sec))
        return True

    def _run(self) -> None:
        while not self._stop.wait(self._poll_sec):
            self._sweep()
            if self._error is not None:
                return

    def _sweep(self) -> None:
        for step, snapshot in spooled_steps(self._spool_dir):
            if step in self._published:
                continue
            if (
                self._aborted.is_set()
                or (binding := self._publish(step, snapshot)) is None
            ):
                return
            self._published[step] = binding
            shutil.rmtree(snapshot, ignore_errors=True)
            try:
                self._report(binding)
            except Exception:
                self._logger.exception("Reporting checkpoint %d failed", step)

    def _publish(self, step: int, snapshot: Path) -> ArtifactBinding | None:
        delay = self._backoff_sec
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                binding, _ = self._publisher.publish_tree(
                    CHECKPOINTS_DIR, snapshot, step_revision(step)
                )
                return binding
            except Exception as exc:
                if self._cancelled():
                    self._error = exc
                    return None
                self._logger.warning(
                    "Publishing checkpoint %d failed (attempt %d of %d): %s",
                    step,
                    attempt,
                    _ATTEMPTS,
                    exc,
                )
                if attempt == _ATTEMPTS:
                    self._error = exc
                    fail_spool(self._spool_dir, str(exc))
                    return None
            if self._interrupted_within(delay):
                return None
            delay *= 2
        return None

    def _clear(self) -> None:
        shutil.rmtree(self._spool_dir, ignore_errors=True)


__all__ = ["CheckpointPublishFailed", "CheckpointUploader"]
