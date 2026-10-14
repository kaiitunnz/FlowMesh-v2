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

# The snapshots a save leaves waiting before it waits for the uploader to catch up.
_MAX_PENDING = 2
_POLL_SEC = 0.5


class CheckpointSpool(TrainerCallback):
    """Snapshots each checkpoint once every rank has written it.

    Every rank waits at the save's barrier, so a distributed save is whole before the
    first rank snapshots it, and only that rank does. A save waits while more than
    ``max_pending`` snapshots wait to publish, so the spool's disk stays bounded, and
    once a checkpoint could not be published every rank stops training.
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
        if distributed:
            torch.distributed.barrier()
        if state.is_world_process_zero and args.output_dir is not None:
            step = state.global_step
            self._spool(Path(args.output_dir) / f"checkpoint-{step}", step)
        if distributed:
            torch.distributed.barrier()
        try:
            with private_dir(self._spool_dir, create=False) as spool:
                failed = spool_failed(spool)
        except (FileNotFoundError, PathRefused):
            failed = False
        if failed:
            control.should_training_stop = True

    def _spool(self, checkpoint: Path, step: int) -> None:
        try:
            with private_dir(self._spool_dir, create=False) as spool:
                if not spool_is_open(spool) or spool_failed(spool):
                    return
                if checkpoint.is_dir():
                    spool_checkpoint(checkpoint, spool, step)
                self._wait_for_uploads(spool)
        except FileNotFoundError:
            return

    def _wait_for_uploads(self, spool: Path) -> None:
        while (
            spool_is_open(spool)
            and not spool_failed(spool)
            and len(spooled_steps(spool)) > self._max_pending
        ):
            time.sleep(self._poll_sec)
