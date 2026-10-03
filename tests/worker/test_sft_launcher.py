"""A multi-GPU SFT run launches its ranks under torchrun on the task's devices."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from shared.tasks import TaskType
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import SFTSpecStrict
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import sft_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.sft_executor import SFTExecutor

_DEEPSPEED = {"zero_optimization": {"stage": 2}, "gradient_accumulation_steps": 1}


def _launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, training: dict[str, Any]
) -> dict[str, Any]:
    """Run SFT on a worker bound to two GPUs, capturing its torchrun launch."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-c")
    monkeypatch.setattr(sft_executor, "_STARTED_ON", "GPU-b,GPU-c")
    monkeypatch.delenv("KV_SFT_DISTRIBUTED", raising=False)
    monkeypatch.setattr(sft_executor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft_executor.torch.cuda, "device_count", lambda: 2)
    # Nothing may reach a real launcher, whichever launch path the code takes.
    monkeypatch.setattr(sft_executor, "deepspeed_available", lambda: False)
    launched: dict[str, Any] = {}

    def fake_torchrun(
        *, nproc_per_node: int, module: str, module_args: list[str], **_: Any
    ) -> None:
        launched["nproc"] = nproc_per_node
        launched["env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        launched["task"] = json.loads(Path(module_args[0]).read_text())

    spec = SFTSpecStrict(
        taskType=TaskType.SFT,
        model=ModelConfig(source=ModelSource(identifier="m")),
        training=training,
    )
    with patch.object(sft_executor, "run_torchrun", side_effect=fake_torchrun):
        SFTExecutor(make_worker_config()).run(
            make_worker_task_message(spec=spec, task_type=TaskType.SFT), tmp_path
        )
    return launched


def test_a_deepspeed_run_launches_under_torchrun_on_the_bound_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched = _launch(
        monkeypatch, tmp_path, {"allow_multi_gpu": True, "deepspeed": _DEEPSPEED}
    )

    assert launched["nproc"] == 2
    assert launched["env"] == "GPU-b,GPU-c"
    training = launched["task"]["data"]["task"]["spec"]["training"]
    assert training["deepspeed"] == _DEEPSPEED


def test_training_devices_are_positions_within_the_bound_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched = _launch(monkeypatch, tmp_path, {"visible_devices": [1, 0]})

    assert launched["env"] == "GPU-c,GPU-b"


def test_a_training_device_outside_the_bound_devices_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ExecutionError, match="not one of the 2 GPU"):
        _launch(monkeypatch, tmp_path, {"visible_devices": [2]})


def test_a_warm_process_maps_each_run_from_the_devices_it_started_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _launch(monkeypatch, tmp_path, {"visible_devices": [1, 0]})
    launched = _launch(monkeypatch, tmp_path, {"visible_devices": [0, 1]})

    assert launched["env"] == "GPU-b,GPU-c"


def test_a_launched_rank_keeps_the_devices_its_launch_chose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sft_executor, "_STARTED_ON", "GPU-b,GPU-c")
    monkeypatch.setattr(sft_executor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft_executor.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-c")
    monkeypatch.delenv("KV_SFT_DISTRIBUTED", raising=False)
    SFTExecutor._configure_devices({"visible_devices": [1, 0]})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c,GPU-b"

    # The rank, started on the parent's narrowed devices, applies the same config.
    monkeypatch.setattr(sft_executor, "_STARTED_ON", "GPU-c,GPU-b")
    monkeypatch.setenv("KV_SFT_DISTRIBUTED", "1")
    SFTExecutor._configure_devices({"visible_devices": [1, 0]})

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c,GPU-b"
