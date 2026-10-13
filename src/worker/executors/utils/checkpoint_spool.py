"""The trainer callback that hands each saved checkpoint to the task's spool."""

from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from ...content.artifacts import spool_checkpoint


class CheckpointSpool(TrainerCallback):
    """Snapshots each checkpoint once every rank has written it.

    Every rank waits at the save's barrier, so a distributed save is whole before the
    first rank snapshots it, and only that rank does.
    """

    def __init__(self, spool_dir: Path) -> None:
        self._spool_dir = spool_dir

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if state.is_world_process_zero and args.output_dir is not None:
            step = state.global_step
            checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
            if checkpoint.is_dir():
                self._spool_dir.mkdir(parents=True, exist_ok=True)
                spool_checkpoint(checkpoint, self._spool_dir, step)
