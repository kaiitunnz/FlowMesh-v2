"""A torchrun rank hands its training result, marked as launched, to its executor."""

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

from shared.schemas.result import DPOResult, LoRAResult, PPOResult, SFTResult
from shared.tasks import TaskType
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import SFTSpecStrict
from shared.utils.manifest import scratch_dir
from tests.worker.factories import make_worker_task_message
from worker.executors import (
    dpo_dist_entry,
    lora_sft_dist_entry,
    ppo_dist_entry,
    sft_dist_entry,
)


def _task_file(tmp_path: Path) -> Path:
    spec = SFTSpecStrict(
        taskType=TaskType.SFT, model=ModelConfig(source=ModelSource(identifier="m"))
    )
    path = tmp_path / "task.json"
    path.write_text(
        make_worker_task_message(spec=spec, task_type=TaskType.SFT).model_dump_json(
            by_alias=True
        )
    )
    return path


@pytest.mark.parametrize(
    ("entry", "executor_name", "result"),
    [
        (sft_dist_entry, "SFTExecutor", SFTResult(model_name="m", output_dir="o")),
        (
            lora_sft_dist_entry,
            "LoRASFTExecutor",
            LoRAResult(model_name="m", output_dir="o"),
        ),
        (dpo_dist_entry, "DPOExecutor", DPOResult(model_name="m", output_dir="o")),
        (ppo_dist_entry, "PPOExecutor", PPOResult(model_name="m", output_dir="o")),
    ],
)
def test_a_rank_writes_its_result_for_the_launcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entry: ModuleType,
    executor_name: str,
    result: BaseModel,
) -> None:
    class FakeExecutor:
        def __init__(self, *_: Any) -> None:
            pass

        def run(self, *_: Any) -> BaseModel:
            return result

        def cleanup_after_run(self) -> None:
            pass

    monkeypatch.setattr(entry, executor_name, FakeExecutor)
    monkeypatch.setattr(entry.WorkerConfig, "from_env", lambda: None)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    entry.main(["entry", _task_file(tmp_path).as_posix(), out_dir.as_posix()])

    written = (scratch_dir(out_dir) / "distributed_result.json").read_text()
    expected = (
        result.model_copy(update={"spawned_torchrun": True})
        if "spawned_torchrun" in type(result).model_fields
        else result
    )
    assert type(result).model_validate_json(written) == expected
