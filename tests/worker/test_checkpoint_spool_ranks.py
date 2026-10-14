"""A distributed save neither holds its ranks in a collective while the uploader
catches up nor lets them disagree on whether to stop."""

import multiprocessing
import os
import shutil
import socket
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch.distributed as dist
from transformers import TrainerControl, TrainerState, TrainingArguments

from worker.content.artifacts import open_spool
from worker.executors.utils.checkpoint_spool import CheckpointSpool

_COLLECTIVE_TIMEOUT_SEC = 3


class _ReadsFailedOnRankZero(CheckpointSpool):
    """A spool whose failed marker lands between the two ranks' readings."""

    def failed(self) -> bool:
        return dist.get_rank() == 0


def _save(rank: int, port: int, root: str, diverging: bool, out: Any) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=_COLLECTIVE_TIMEOUT_SEC),
    )
    base = Path(root)
    state = TrainerState(global_step=4)
    state.is_world_process_zero = rank == 0
    control = TrainerControl()
    spool_type = _ReadsFailedOnRankZero if diverging else CheckpointSpool
    try:
        spool_type(base / "spool", poll_sec=0.05).on_save(
            TrainingArguments(output_dir=(base / "out").as_posix(), report_to=[]),
            state,
            control,
        )
        out.put((rank, "stopped" if control.should_training_stop else "goes on"))
    except Exception as exc:
        out.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_ranks(root: Path, diverging: bool, during: Any = None) -> dict[int, str]:
    context = multiprocessing.get_context("spawn")
    out = context.Queue()
    port = _free_port()
    ranks = [
        context.Process(
            target=_save, args=(rank, port, root.as_posix(), diverging, out)
        )
        for rank in (0, 1)
    ]
    for process in ranks:
        process.start()
    if during is not None:
        during()
    results = dict(out.get(timeout=60) for _ in ranks)
    for process in ranks:
        process.join(30)
    return results


def _spool_with_backlog(root: Path, pending: int) -> Path:
    spool = root / "spool"
    spool.mkdir(mode=0o700)
    open_spool(spool)
    for step in range(1, pending + 1):
        (spool / f"step-{step}" / f"checkpoint-{step}").mkdir(parents=True)
    (root / "out" / "checkpoint-4").mkdir(parents=True)
    (root / "out" / "checkpoint-4" / "w.bin").write_bytes(b"w")
    return spool


def test_a_slow_upload_holds_no_rank_in_a_collective(tmp_path: Path) -> None:
    spool = _spool_with_backlog(tmp_path, 3)

    def upload_slowly() -> None:
        time.sleep(_COLLECTIVE_TIMEOUT_SEC * 2)
        for step in (1, 2):
            shutil.rmtree(spool / f"step-{step}")

    assert _run_ranks(tmp_path, False, upload_slowly) == {
        0: "goes on",
        1: "goes on",
    }
    assert (spool / "step-4").is_dir()


def test_every_rank_stops_on_the_first_ranks_reading(tmp_path: Path) -> None:
    _spool_with_backlog(tmp_path, 0)

    assert _run_ranks(tmp_path, True) == {0: "stopped", 1: "stopped"}
