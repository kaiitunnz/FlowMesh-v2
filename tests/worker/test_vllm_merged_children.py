"""A merged vLLM dispatch returns a result for each child it can run, and only those."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference-gpu)")
pytest.importorskip("torch", reason="torch not installed (needs --extra inference)")

from pydantic import TypeAdapter

from shared.schemas.result import InferenceResult
from shared.tasks import MergedChildTaskStrict, TaskSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.vllm_executor import VLLMExecutor

_MODEL = {"source": {"identifier": "Qwen/Qwen3-4B"}}


def _spec(prompt: str, **fields: Any) -> dict[str, Any]:
    return {
        "taskType": "inference",
        "model": _MODEL,
        "data": {"type": "list", "items": [prompt]},
        **fields,
    }


def _child(task_id: str, spec: dict[str, Any]) -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id=task_id,
        owner_id="usr-test",
        workflow_id="wfl-test",
        spec=TypeAdapter(TaskSpecStrict).validate_python(spec),
    )


def _run(
    parent: dict[str, Any],
    children: list[MergedChildTaskStrict],
    out_dir: Path,
    rejected: str | None = None,
) -> tuple[InferenceResult, MagicMock]:
    """Run a merged dispatch against a stand-in engine, which aborts the whole batch
    when it rejects the ``rejected`` prompt, as vLLM does."""
    executor = VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    llm = MagicMock()
    llm.get_tokenizer.return_value.chat_template = None

    def _generate(prompts: list[Any], **_kwargs: Any) -> list[SimpleNamespace]:
        if rejected in prompts:
            raise ValueError("The decoder prompt is longer than max_model_len")
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=f"out-{prompt}", finish_reason="stop", token_ids=[1]
                    )
                ],
                prompt_token_ids=[1, 2],
            )
            for prompt in prompts
        ]

    llm.generate.side_effect = _generate

    def _ensure(*_args: Any, **_kwargs: Any) -> None:
        executor._llm = llm

    executor._ensure_llm = _ensure  # type: ignore[method-assign]
    msg = make_worker_task_message(
        parent, task_type=TaskType.INFERENCE, merged_children=children
    )
    result = executor.run(msg, out_dir)
    assert isinstance(result, InferenceResult)
    return result, llm


def test_a_merged_dispatch_returns_each_childs_own_result(tmp_path: Path) -> None:
    result, _ = _run(
        _spec("parent"),
        [_child("tsk-b", _spec("bravo")), _child("tsk-c", _spec("charlie"))],
        tmp_path,
    )

    assert [item.prompt for item in result.items] == ["parent"]
    assert {
        child_id: [item.prompt for item in child.items]
        for child_id, child in result.children.items()
        if isinstance(child, InferenceResult)
    } == {"tsk-b": ["bravo"], "tsk-c": ["charlie"]}


def test_a_child_the_batch_cannot_run_is_left_out_of_it(tmp_path: Path) -> None:
    unpreparable = _spec("x") | {
        "data": {"type": "list", "items": ["x"], "metadata": [{}, {}]}
    }
    result, llm = _run(
        _spec("parent"),
        [
            _child("tsk-ok", _spec("ok")),
            _child("tsk-params", _spec("params", inference={"temperature": 0.9})),
            _child("tsk-unpreparable", unpreparable),
        ],
        tmp_path,
    )

    assert set(result.children) == {"tsk-ok"}
    assert llm.generate.call_args.args[0] == ["parent", "ok"]


def test_the_parents_own_input_still_fails_the_dispatch(tmp_path: Path) -> None:
    unpreparable = _spec("x") | {
        "data": {"type": "list", "items": ["x"], "metadata": [{}, {}]}
    }

    with pytest.raises(ExecutionError):
        _run(unpreparable, [_child("tsk-ok", _spec("ok"))], tmp_path)


def test_a_batch_the_engine_rejects_fails_the_dispatch(tmp_path: Path) -> None:
    # The dispatch fails rather than running again on an engine the failure may have
    # left unusable; the root decides whose failure it was.
    with pytest.raises(ValueError):
        _run(
            _spec("parent"),
            [_child("tsk-ok", _spec("ok")), _child("tsk-long", _spec("too-long"))],
            tmp_path,
            rejected="too-long",
        )


def test_each_task_writes_its_own_export_and_lineage(tmp_path: Path) -> None:
    export = {"jsonl_export": {"path": "rows.jsonl", "fields": {"answer": "output"}}}
    _run(
        _spec("parent", postprocess=export),
        [_child("tsk-b", _spec("bravo", postprocess=export))],
        tmp_path / "tsk-test",
    )

    for task, prompt in (("tsk-test", "parent"), ("tsk-b", "bravo")):
        rows = (tmp_path / task / "artifacts" / "rows.jsonl").read_text().splitlines()
        assert [json.loads(row) for row in rows] == [{"answer": f"out-{prompt}"}]
        assets = (tmp_path / task / "logs" / "assets.jsonl").read_text()
        assert [json.loads(row)["data_id"] for row in assets.splitlines()] == [task]
