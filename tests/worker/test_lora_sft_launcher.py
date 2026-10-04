"""A LoRA SFT run with a DeepSpeed config trains as a torchrun rank."""

import os
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

import pytest

from shared.schemas.result import LoRAResult
from shared.tasks import TaskType
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import LoRASFTSpecStrict
from shared.utils.manifest import scratch_dir
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import lora_sft_executor, sft_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.lora_sft_executor import LoRASFTExecutor

_DEEPSPEED = {"zero_optimization": {"stage": 2}}


def _train_in_process(*_: Any, **__: Any) -> NoReturn:
    raise RuntimeError("the run trained in-process")


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    training: dict[str, Any],
    devices: str = "GPU-b",
) -> tuple[dict[str, Any], LoRAResult]:
    """Run LoRA SFT on a worker bound to ``devices``, capturing its torchrun launch."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    monkeypatch.setattr(sft_executor, "_STARTED_ON", devices)
    monkeypatch.setattr(
        sft_executor, "_started_device_count", lambda: len(devices.split(","))
    )
    monkeypatch.delenv(sft_executor._SFT_LAUNCHER_FLAG, raising=False)
    monkeypatch.setattr(
        lora_sft_executor.AutoTokenizer, "from_pretrained", _train_in_process
    )
    launched: dict[str, Any] = {}
    rank_result = LoRAResult(model_name="m", output_dir=tmp_path.as_posix())

    def fake_torchrun(
        *, nproc_per_node: int, module: str, module_args: list[str], **_: Any
    ) -> None:
        launched["nproc"] = nproc_per_node
        launched["module"] = module
        launched["env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        (scratch_dir(Path(module_args[1])) / "distributed_result.json").write_text(
            rank_result.model_dump_json()
        )

    spec = LoRASFTSpecStrict(
        taskType=TaskType.LORA_SFT,
        model=ModelConfig(source=ModelSource(identifier="m")),
        training=training,
    )
    with patch.object(lora_sft_executor, "run_torchrun", side_effect=fake_torchrun):
        result = LoRASFTExecutor(make_worker_config()).run(
            make_worker_task_message(spec=spec, task_type=TaskType.LORA_SFT), tmp_path
        )
    assert result == rank_result
    return launched, result


def test_a_deepspeed_lora_run_on_one_gpu_launches_one_torchrun_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched, _ = _run(monkeypatch, tmp_path, {"deepspeed": _DEEPSPEED})

    assert launched == {
        "nproc": 1,
        "module": "worker.executors.lora_sft_dist_entry",
        "env": "GPU-b",
    }


def test_a_deepspeed_lora_run_on_two_gpus_launches_one_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched, _ = _run(monkeypatch, tmp_path, {"deepspeed": _DEEPSPEED}, "GPU-b,GPU-c")

    assert launched["nproc"] == 1


def test_a_lora_run_without_deepspeed_trains_in_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ExecutionError, match="trained in-process"):
        _run(monkeypatch, tmp_path, {})
