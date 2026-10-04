#!/usr/bin/env python3
"""Distributed worker used to execute SFT tasks launched via torchrun.

The ``SFTExecutor`` persists task state to disk, and this module rehydrates the
task inside each distributed rank. It accepts and ignores a ``--local_rank``
flag while forwarding the remaining positional arguments to ``SFTExecutor``.
"""

import argparse
import json
import sys
from pathlib import Path

from shared.tasks.worker_message import WorkerTaskMessage
from worker.config import WorkerConfig

from .sft_executor import SFTExecutor
from .utils.distributed import run_rank


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Distributed SFT worker entrypoint")
    parser.add_argument(
        "task_json", type=Path, help="Path to serialized task specification"
    )
    parser.add_argument(
        "out_dir", type=Path, help="Output directory for training artifacts"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="Rank injected by torchrun",
    )
    args = parser.parse_args(argv[1:])

    task_path = args.task_json
    out_dir = args.out_dir
    with task_path.open("r", encoding="utf-8") as fh:
        task = WorkerTaskMessage.model_validate(json.load(fh))
    ex = SFTExecutor(WorkerConfig.from_env())
    try:
        run_rank(out_dir, lambda: ex.run(task, out_dir))
    finally:
        try:
            ex.cleanup_after_run()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
