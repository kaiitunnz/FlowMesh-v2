"""Tasks loading the same published model share one vLLM engine, and each request
reads its own task's copy of what it loads."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.tasks import TaskType
from shared.tasks.specs import InferenceSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.content.artifact_inputs import artifact_inputs_dir
from worker.executors import vllm_executor, vllm_lora_executor
from worker.executors.vllm_executor import VLLMExecutor
from worker.executors.vllm_lora_executor import VLLMLoRAExecutor

_DIGEST = "ab" * 32


def _hydrated(tmp_path: Path, task_id: str) -> Path:
    path = artifact_inputs_dir(tmp_path / task_id) / _DIGEST
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "adapter_config.json").write_text("{}")
    return path


def _spec(**model: Any) -> InferenceSpecStrict:
    return InferenceSpecStrict.model_validate(
        {
            "taskType": TaskType.INFERENCE.value,
            "model": {
                "source": {"type": "huggingface", "identifier": "org/base"},
                "vllm": {"tensor_parallel_size": 1},
                **model,
            },
            "data": {"type": "list", "items": ["hi"]},
        }
    )


def _load(path: Path) -> InferenceSpecStrict:
    spec = _spec()
    return spec.model_copy(
        update={"checkpoint": {"load": {"type": "http", "url": path.as_posix()}}}
    )


def test_two_tasks_loading_one_model_share_an_engine(tmp_path: Path) -> None:
    executor = VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    llm = MagicMock()
    with patch.object(vllm_executor, "LLM", llm):
        executor._ensure_llm(_load(_hydrated(tmp_path, "tsk-a")))
        executor._ensure_llm(_load(_hydrated(tmp_path, "tsk-b")))

    assert llm.call_count == 1


def test_a_task_loading_another_model_gets_its_own_engine(tmp_path: Path) -> None:
    executor = VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    other = artifact_inputs_dir(tmp_path / "tsk-b") / ("cd" * 32)
    other.mkdir(parents=True)
    llm = MagicMock()
    with (
        patch.object(vllm_executor, "LLM", llm),
        patch.object(executor, "_shutdown_llm"),
    ):
        executor._ensure_llm(_load(_hydrated(tmp_path, "tsk-a")))
        executor._ensure_llm(_load(other))

    assert llm.call_count == 2


def _adapter_spec(path: Path, apply: str = "runtime") -> InferenceSpecStrict:
    return _spec(
        adapters=[{"type": "lora", "name": "a", "url": path.as_posix(), "apply": apply}]
    )


def test_each_lora_request_reads_its_own_tasks_adapter(tmp_path: Path) -> None:
    executor = VLLMLoRAExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    first, second = _hydrated(tmp_path, "tsk-a"), _hydrated(tmp_path, "tsk-b")
    llm = MagicMock()
    with (
        patch.object(vllm_executor, "LLM", llm),
        patch.object(vllm_lora_executor, "LoRARequest", MagicMock()) as request,
    ):
        executor._ensure_llm(_adapter_spec(first))
        executor._build_generate_kwargs(_adapter_spec(first), tmp_path / "tsk-a")
        executor._ensure_llm(_adapter_spec(second))
        executor._build_generate_kwargs(_adapter_spec(second), tmp_path / "tsk-b")

    assert llm.call_count == 1
    assert [call.args[2] for call in request.call_args_list] == [
        first.as_posix(),
        second.as_posix(),
    ]


def test_a_merged_adapter_merges_once_into_a_reused_engine(tmp_path: Path) -> None:
    executor = VLLMLoRAExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    engine = MagicMock()
    with patch.object(vllm_executor, "LLM", MagicMock(return_value=engine)):
        for task in ("tsk-a", "tsk-b"):
            spec = _adapter_spec(_hydrated(tmp_path, task), apply="merge")
            executor._ensure_llm(spec)
            executor._build_generate_kwargs(spec, tmp_path / task)

    assert engine.merge_lora_weights.call_count == 1


def test_a_merge_that_fails_partway_leaves_no_engine_to_reuse(tmp_path: Path) -> None:
    executor = VLLMLoRAExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    broken, fresh = MagicMock(), MagicMock()
    broken.merge_lora_weights.side_effect = [None, RuntimeError("out of memory")]
    adapter = _hydrated(tmp_path, "tsk-a")
    spec = _spec(
        adapters=[
            {"type": "lora", "name": name, "url": adapter.as_posix(), "apply": "merge"}
            for name in ("a", "b")
        ]
    )
    llm = MagicMock(side_effect=[broken, fresh])
    with (
        patch.object(vllm_executor, "LLM", llm),
        patch.object(vllm_executor, "destroy_distributed_environment"),
    ):
        executor._ensure_llm(spec)
        with pytest.raises(RuntimeError):
            executor._build_generate_kwargs(spec, tmp_path / "tsk-a")
        executor._ensure_llm(spec)
        executor._build_generate_kwargs(spec, tmp_path / "tsk-a")

    assert llm.call_count == 2
    assert fresh.merge_lora_weights.call_count == 2
