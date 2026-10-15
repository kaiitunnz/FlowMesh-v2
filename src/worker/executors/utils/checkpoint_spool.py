"""The trainer callback that hands each saved checkpoint to the task's spool."""

import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from shared.utils.nofollow import PathRefused, private_dir

from ...content.artifacts import (
    spool_checkpoint,
    spool_failed,
    spool_is_open,
    spooled_steps,
)

# The snapshots a save finds waiting before it waits for the uploader to catch up.
_MAX_PENDING = 2
_POLL_SEC = 0.5


class CheckpointSpool(TrainerCallback):
    """Snapshots each checkpoint once every rank has written it.

    Before a save's barrier every rank waits while ``max_pending`` snapshots are
    publishing, which bounds the spool's disk outside any collective; once a checkpoint
    could not be published, every rank stops at the same save.
    """

    def __init__(
        self,
        spool_dir: Path,
        max_pending: int = _MAX_PENDING,
        poll_sec: float = _POLL_SEC,
    ) -> None:
        self._spool_dir = spool_dir
        self._max_pending = max_pending
        self._poll_sec = poll_sec
        self._stopped = False

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        distributed = torch.distributed.is_available() and (
            torch.distributed.is_initialized()
        )
        self._wait_for_uploads()
        if distributed:
            torch.distributed.barrier()
        if state.is_world_process_zero and args.output_dir is not None:
            step = state.global_step
            self._spool(Path(args.output_dir) / f"checkpoint-{step}", step)
        if distributed:
            torch.distributed.barrier()
        failed = self.failed()
        if distributed:
            decided = [failed]
            torch.distributed.broadcast_object_list(decided, src=0)
            failed = bool(decided[0])
        if failed:
            self._stopped = True
            control.should_training_stop = True

    @property
    def stopped(self) -> bool:
        """Whether a save decided to stop training, alike on every rank."""
        return self._stopped

    def failed(self) -> bool:
        """Return whether a checkpoint this spool took could not be published."""
        try:
            with private_dir(self._spool_dir, create=False) as spool:
                return spool_failed(spool)
        except (FileNotFoundError, PathRefused):
            return False

    def _spool(self, checkpoint: Path, step: int) -> None:
        try:
            with private_dir(self._spool_dir, create=False) as spool:
                if not spool_is_open(spool) or spool_failed(spool):
                    return
                if checkpoint.is_dir():
                    spool_checkpoint(checkpoint, spool, step)
        except FileNotFoundError:
            return

    def _wait_for_uploads(self) -> None:
        try:
            with private_dir(self._spool_dir, create=False) as spool:
                while (
                    spool_is_open(spool)
                    and not spool_failed(spool)
                    and len(spooled_steps(spool)) >= self._max_pending
                ):
                    time.sleep(self._poll_sec)
        except (FileNotFoundError, PathRefused):
            return
