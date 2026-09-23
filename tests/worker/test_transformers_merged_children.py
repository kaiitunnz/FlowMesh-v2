"""A merged HF transformers dispatch runs each child as a task of its own."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="torch not installed (needs --extra inference)")

from pydantic import TypeAdapter

from shared.schemas.result import EmbeddingResult, InferenceItem, InferenceResult
from shared.tasks import MergedChildTaskStrict, TaskSpecStrict
from shared.tasks.specs import EmbeddingSpecStrict, InferenceSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.transformers_executor import HFTransformersExecutor

_MODEL = {"source": {"identifier": "org/model"}, "transformers": {"dtype": "auto"}}


def _spec(prompt: str) -> dict[str, Any]:
    return {
        "taskType": "inference",
        "model": _MODEL,
        "data": {"type": "list", "items": [prompt]},
    }


def _child(task_id: str, prompt: str) -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id=task_id,
        owner_id="usr-test",
        workflow_id="wfl-test",
        spec=TypeAdapter(TaskSpecStrict).validate_python(_spec(prompt)),
    )


def _run(
    parent: str, children: list[MergedChildTaskStrict], results_dir: Path
) -> tuple[InferenceResult, dict[str, Path]]:
    """Run a merged dispatch whose generation echoes each prompt and fails on "bad"."""
    executor = HFTransformersExecutor(DEFAULT_WORKER_CONFIG)
    out_dirs: dict[str, Path] = {}

    def _run_inner(
        spec: InferenceSpecStrict | EmbeddingSpecStrict, task_id: str, out_dir: Path
    ) -> InferenceResult | EmbeddingResult:
        assert spec.data is not None
        (prompt,) = spec.data["items"]
        if prompt == "bad":
            raise ExecutionError("generation failed")
        out_dirs[task_id] = out_dir
        return InferenceResult(
            model="org/model",
            items=[InferenceItem(index=0, prompt=prompt, output=f"out-{prompt}")],
        )

    executor._run_inner = _run_inner  # type: ignore[method-assign]
    msg = make_worker_task_message(
        _spec(parent),
        task_type=TaskType.INFERENCE,
        task_id="tsk-a",
        merged_children=children,
    )
    result = executor.run(msg, results_dir / "tsk-a")
    assert isinstance(result, InferenceResult)
    return result, out_dirs


def test_a_merged_dispatch_returns_each_childs_own_result(tmp_path: Path) -> None:
    result, out_dirs = _run(
        "alpha", [_child("tsk-b", "bravo"), _child("tsk-c", "charlie")], tmp_path
    )

    assert [item.output for item in result.items] == ["out-alpha"]
    assert {
        child_id: [item.output for item in child.items]
        for child_id, child in result.children.items()
        if isinstance(child, InferenceResult)
    } == {"tsk-b": ["out-bravo"], "tsk-c": ["out-charlie"]}
    assert out_dirs == {task: tmp_path / task for task in ("tsk-a", "tsk-b", "tsk-c")}


def test_a_child_that_fails_is_left_out_of_the_result(tmp_path: Path) -> None:
    result, _ = _run(
        "alpha", [_child("tsk-bad", "bad"), _child("tsk-c", "charlie")], tmp_path
    )

    assert set(result.children) == {"tsk-c"}


def test_the_parents_own_failure_fails_the_dispatch(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError):
        _run("bad", [_child("tsk-c", "charlie")], tmp_path)
