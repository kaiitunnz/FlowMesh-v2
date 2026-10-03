"""An executor reporting the CUDA devices its process saw, for spawned children."""

import os
from pathlib import Path

from shared.schemas.result import BaseExecutorResult
from worker.executors.base_executor import Executor

SEEN_AT_IMPORT = os.environ.get("CUDA_VISIBLE_DEVICES")


class SeenDevicesResult(BaseExecutorResult):
    at_import: str | None
    at_run: str | None
    device_order: str | None


class SeenDevicesExecutor(Executor):
    name = "seen_devices"

    def run(self, task, out_dir: Path) -> SeenDevicesResult:
        return SeenDevicesResult(
            at_import=SEEN_AT_IMPORT,
            at_run=os.environ.get("CUDA_VISIBLE_DEVICES"),
            device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        )
