"""An SFT run launches its ranks under torchrun on the task's devices."""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

import pytest

from shared.schemas.result import SFTResult
from shared.tasks import TaskType
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import SFTSpecStrict
from shared.utils.manifest import scratch_dir
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import sft_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.sft_executor import SFTExecutor
from worker.executors.utils import distributed

_DEEPSPEED = {"zero_optimization": {"stage": 2}, "gradient_accumulation_steps": 1}


class _InProcess(Exception):
    pass


def _train_in_process(*_: Any, **__: Any) -> NoReturn:
    raise _InProcess("the run trained in-process")


def _launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    training: dict[str, Any],
    devices: str = "GPU-b,GPU-c",
) -> dict[str, Any]:
    """Run SFT on a worker bound to ``devices``, capturing its torchrun launch."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    monkeypatch.setattr(sft_executor, "_STARTED_ON", devices)
    monkeypatch.delenv(sft_executor._SFT_LAUNCHER_FLAG, raising=False)
    monkeypatch.setattr(sft_executor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sft_executor.torch.cuda, "device_count", lambda: len(devices.split(","))
    )
    monkeypatch.setattr(
        sft_executor, "_started_device_count", lambda: len(devices.split(","))
    )
    monkeypatch.setattr(
        sft_executor.AutoTokenizer, "from_pretrained", _train_in_process
    )
    # DeepSpeed reads as installed, and its own launcher refuses to run, so a run
    # handed to it fails rather than reaching a real launcher.
    monkeypatch.setattr(sft_executor, "deepspeed_available", lambda: True)
    runner = types.ModuleType("deepspeed.launcher.runner")

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("the DeepSpeed launcher must not run")

    runner.main = refuse  # type: ignore[attr-defined]
    for name, module in (
        ("deepspeed", types.ModuleType("deepspeed")),
        ("deepspeed.launcher", types.ModuleType("deepspeed.launcher")),
        ("deepspeed.launcher.runner", runner),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    launched: dict[str, Any] = {}

    def fake_torchrun(
        *, nproc_per_node: int, module: str, module_args: list[str], **_: Any
    ) -> None:
        launched["nproc"] = nproc_per_node
        launched["env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        launched["task"] = json.loads(Path(module_args[0]).read_text())
        (scratch_dir(Path(module_args[1])) / "distributed_result.json").write_text(
            SFTResult(model_name="m").model_dump_json()
        )

    spec = SFTSpecStrict(
        taskType=TaskType.SFT,
        model=ModelConfig(source=ModelSource(identifier="m")),
        training=training,
    )
    with patch.object(distributed, "run_torchrun", side_effect=fake_torchrun):
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


def test_a_deepspeed_run_on_one_gpu_launches_one_torchrun_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched = _launch(monkeypatch, tmp_path, {"deepspeed": _DEEPSPEED}, "GPU-b")

    assert launched["nproc"] == 1
    assert launched["env"] == "GPU-b"


def test_a_deepspeed_run_held_to_one_gpu_launches_one_torchrun_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched = _launch(
        monkeypatch, tmp_path, {"allow_multi_gpu": False, "deepspeed": _DEEPSPEED}
    )

    assert launched["nproc"] == 1
    assert launched["env"] == "GPU-b"


def test_a_one_gpu_run_without_deepspeed_trains_in_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ExecutionError, match="trained in-process"):
        _launch(monkeypatch, tmp_path, {}, "GPU-b")


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
    monkeypatch.delenv(sft_executor._SFT_LAUNCHER_FLAG, raising=False)
    SFTExecutor._configure_devices({"visible_devices": [1, 0]})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c,GPU-b"

    # The rank, started on the parent's narrowed devices, applies the same config.
    monkeypatch.setattr(sft_executor, "_STARTED_ON", "GPU-c,GPU-b")
    monkeypatch.setenv(sft_executor._SFT_LAUNCHER_FLAG, "1")
    SFTExecutor._configure_devices({"visible_devices": [1, 0]})

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c,GPU-b"


def test_an_in_process_run_narrows_its_devices_before_cuda_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cuda_started() -> NoReturn:
        raise AssertionError("CUDA read its devices before the run narrowed them")

    monkeypatch.setattr(sft_executor.torch.cuda, "is_available", cuda_started)
    monkeypatch.setattr(sft_executor.torch.cuda, "device_count", cuda_started)
    monkeypatch.setattr(sft_executor, "_STARTED_ON", "GPU-b,GPU-c")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-c")
    monkeypatch.delenv(sft_executor._SFT_LAUNCHER_FLAG, raising=False)

    SFTExecutor._configure_devices({"allow_multi_gpu": False, "primary_gpu": 1})

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c"


def test_counting_the_started_devices_leaves_nvml_initialised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def shut_down() -> NoReturn:
        raise AssertionError("NVML shut down under the worker's other readers")

    monkeypatch.setattr(sft_executor, "_STARTED_ON", None)
    monkeypatch.setattr(sft_executor.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(sft_executor.pynvml, "nvmlDeviceGetCount", lambda: 2)
    monkeypatch.setattr(sft_executor.pynvml, "nvmlShutdown", shut_down)

    assert sft_executor._started_device_count() == 2
