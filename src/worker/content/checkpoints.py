"""Publishing a training run's checkpoints as it saves them.

A trainer's save hands each closed checkpoint to a spool; one uploader per task
publishes each snapshot as the ``checkpoints`` artifact at the revision of its step,
one at a time, and reports each publication on the task's event stream. The task's
success lists every revision the uploader published, so a report lost on the way still
binds when the task settles.

A publication that fails is retried with backoff, since every write is conditional and
repeats settle on what landed. One that still fails marks the spool failed, which stops
the trainer at its next save, and fails the task when the uploader drains. However the
uploader stops, it closes the spool, so a save never waits on it; a spool it could not
close goes stale ``SPOOL_STALE_SEC`` after the uploader last refreshed it.
"""

import logging
import shutil
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

from shared.content import FabricObjectStore
from shared.tasks.artifact_binding import ArtifactBinding, step_revision
from shared.utils.nofollow import private_dir, remove_below

from .artifacts import (
    CHECKPOINTS_DIR,
    ArtifactPublisher,
    close_spool,
    fail_spool,
    open_spool,
    refresh_spool,
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
        self._held = ExitStack()
        self._spool = spool_dir
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
        self._held_lock = threading.Lock()
        self._cleared = False
        self._heartbeat = threading.Thread(
            target=self._keep_open, name="checkpoint-spool-heartbeat", daemon=True
        )

    def start(self) -> None:
        self._spool = self._held.enter_context(private_dir(self._spool_dir))
        open_spool(self._spool)
        self._thread.start()
        self._heartbeat.start()

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
        try:
            while not self._stop.wait(self._poll_sec):
                self._sweep()
                if self._error is not None:
                    return
        finally:
            with self._held_lock:
                try:
                    if not self._cleared:
                        close_spool(self._spool)
                except OSError:
                    self._logger.exception("Closing the checkpoint spool failed")

    def _keep_open(self) -> None:
        """Refresh the spool's open mark while the uploader runs, through publications
        however long, so the mark goes stale only once it stops."""
        while self._thread.is_alive():
            with self._held_lock:
                if self._cleared:
                    return
                try:
                    refresh_spool(self._spool)
                except FileNotFoundError:
                    return
                except OSError:
                    self._logger.debug("Refreshing the checkpoint spool failed")
            self._thread.join(self._poll_sec)

    def _sweep(self) -> None:
        for step, snapshot in spooled_steps(self._spool):
            if step in self._published:
                continue
            if (
                self._aborted.is_set()
                or (binding := self._publish(step, snapshot)) is None
            ):
                return
            self._published[step] = binding
            # The snapshot is reached through the spool's descriptor, which a cleared
            # uploader has closed and the process may have reused.
            with self._held_lock:
                if self._cleared:
                    return
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
                    with self._held_lock:
                        if not self._cleared:
                            fail_spool(self._spool, str(exc))
                    return None
            if self._interrupted_within(delay):
                return None
            delay *= 2
        return None

    def _clear(self) -> None:
        with self._held_lock:
            self._cleared = True
            self._held.close()
        remove_below(self._spool_dir)


__all__ = ["CheckpointPublishFailed", "CheckpointUploader"]
