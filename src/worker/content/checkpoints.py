"""Publishing a training run's checkpoints as it saves them.

A trainer's save hands each closed checkpoint to a spool; one uploader per task
publishes each snapshot as the ``checkpoints`` artifact at the revision of its step,
one at a time, and reports each publication on the task's event stream. The task's
success lists every revision the uploader published, so a report lost on the way still
binds when the task settles.
"""

import logging
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from shared.tasks.artifact_binding import ArtifactBinding, step_revision

from .artifacts import CHECKPOINTS_DIR, ArtifactPublisher, spooled_steps

_POLL_SEC = 1.0


class CheckpointPublishFailed(RuntimeError):
    """A checkpoint the trainer saved could not be published."""


class CheckpointUploader:
    """Publishes the snapshots appearing in one task's checkpoint spool."""

    def __init__(
        self,
        publisher: ArtifactPublisher,
        spool_dir: Path,
        report: Callable[[ArtifactBinding], None],
        logger: logging.Logger,
        poll_sec: float = _POLL_SEC,
    ) -> None:
        self._publisher = publisher
        self._spool_dir = spool_dir
        self._report = report
        self._logger = logger
        self._poll_sec = poll_sec
        self._published: dict[int, ArtifactBinding] = {}
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="checkpoint-uploader", daemon=True
        )

    def start(self) -> None:
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
        """Stop without publishing what is still spooled."""
        self._stop.set()
        self._thread.join()
        self._clear()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_sec):
            self._sweep()
            if self._error is not None:
                return

    def _sweep(self) -> None:
        for step, snapshot in spooled_steps(self._spool_dir):
            if step in self._published:
                continue
            try:
                binding, _ = self._publisher.publish_tree(
                    CHECKPOINTS_DIR, snapshot, step_revision(step)
                )
            except Exception as exc:
                self._logger.warning("Publishing checkpoint %d failed: %s", step, exc)
                self._error = exc
                return
            self._published[step] = binding
            shutil.rmtree(snapshot, ignore_errors=True)
            try:
                self._report(binding)
            except Exception:
                self._logger.exception("Reporting checkpoint %d failed", step)

    def _clear(self) -> None:
        shutil.rmtree(self._spool_dir, ignore_errors=True)


__all__ = ["CheckpointPublishFailed", "CheckpointUploader"]
