"""An omni result names the model that produced it."""

import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.omni_text2image_executor import OmniText2ImageExecutor


def _executor(model_name: str | None) -> OmniText2ImageExecutor:
    executor = object.__new__(OmniText2ImageExecutor)
    executor._model_name = model_name
    return executor


def test_the_loaded_model_names_the_result() -> None:
    assert _executor("Qwen/Qwen3-Omni-30B-A3B").model_name == "Qwen/Qwen3-Omni-30B-A3B"


def test_a_result_with_no_loaded_model_fails() -> None:
    with pytest.raises(ExecutionError, match="not initialized"):
        _ = _executor(None).model_name
