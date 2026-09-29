"""An omni result names the model that produced it."""

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from shared.schemas.result import OmniText2ImageResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.omni_text2image_executor import OmniText2ImageExecutor

_SPEC = {
    "taskType": "omni_text2image",
    "model": {"source": {"identifier": "org/omni"}},
    "data": {"type": "list", "items": ["a cat"]},
}


def _run(loaded: str | None, out_dir: Path) -> OmniText2ImageResult:
    executor = OmniText2ImageExecutor(DEFAULT_WORKER_CONFIG)

    def load(_spec: object) -> None:
        executor._model_name = loaded

    msg = make_worker_task_message(_SPEC, task_type=TaskType.OMNI_TEXT2IMAGE)
    image = Image.new("RGB", (1, 1))
    with (
        patch.object(executor, "_ensure_omni", side_effect=load),
        patch.object(executor, "_generate_images", return_value=[image]),
    ):
        result = executor.run(msg, out_dir)
    assert isinstance(result, OmniText2ImageResult)
    return result


def test_the_loaded_model_names_the_result(tmp_path: Path) -> None:
    assert _run("org/omni", tmp_path).model == "org/omni"


def test_a_result_with_no_loaded_model_fails_the_task(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match="not initialized"):
        _run(None, tmp_path)
