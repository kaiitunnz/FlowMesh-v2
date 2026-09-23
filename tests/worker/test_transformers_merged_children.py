"""A merged HF transformers dispatch generates once for every task it runs."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch", reason="torch not installed (needs --extra inference)")

import torch
from pydantic import TypeAdapter

from shared.schemas.result import InferenceResult
from shared.tasks import MergedChildTaskStrict, TaskSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.transformers_executor import HFTransformersExecutor

_MODEL = {"source": {"identifier": "org/model"}, "transformers": {"dtype": "auto"}}
_EXPORT = {"jsonl_export": {"path": "rows.jsonl", "fields": {"answer": "output"}}}
_OUTPUT_OFFSET = 1000


class _Tokenizer:
    """Encodes each prompt as one token and decodes a generated token as its answer."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = None

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, prompts: list[str], **_kwargs: Any) -> dict[str, Any]:
        ids = [self.vocab.setdefault(prompt, len(self.vocab) + 1) for prompt in prompts]
        return {
            "input_ids": torch.tensor([[token] for token in ids]),
            "attention_mask": torch.ones(len(ids), 1, dtype=torch.long),
        }

    def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
        prompts = {token: prompt for prompt, token in self.vocab.items()}
        return " ".join(f"out-{prompts[int(t) - _OUTPUT_OFFSET]}" for t in tokens)


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
    parent: dict[str, Any], children: list[MergedChildTaskStrict], results_dir: Path
) -> tuple[InferenceResult, MagicMock]:
    executor = HFTransformersExecutor(DEFAULT_WORKER_CONFIG)
    executor._tok = _Tokenizer()  # type: ignore[assignment]
    model = MagicMock()
    model.generate.side_effect = lambda input_ids, **_kwargs: torch.cat(
        [input_ids, input_ids + _OUTPUT_OFFSET], dim=1
    )
    executor._model = model
    executor._device = "cpu"
    executor._model_name = "org/model"
    msg = make_worker_task_message(
        parent,
        task_type=TaskType.INFERENCE,
        task_id="tsk-a",
        merged_children=children,
    )
    with patch.object(executor, "_ensure_model"):
        result = executor.run(msg, results_dir / "tsk-a")
    assert isinstance(result, InferenceResult)
    return result, model


def _outputs(result: InferenceResult) -> dict[str, list[str]]:
    return {
        child_id: [str(item.output) for item in child.items]
        for child_id, child in result.children.items()
        if isinstance(child, InferenceResult)
    }


def test_a_merged_dispatch_generates_once_for_every_task(tmp_path: Path) -> None:
    result, model = _run(
        _spec("alpha"),
        [_child("tsk-b", _spec("bravo")), _child("tsk-c", _spec("charlie"))],
        tmp_path,
    )

    model.generate.assert_called_once()
    assert len(model.generate.call_args.kwargs["input_ids"]) == 3
    assert [item.output for item in result.items] == ["out-alpha"]
    assert _outputs(result) == {"tsk-b": ["out-bravo"], "tsk-c": ["out-charlie"]}
    for task_result in (result, *result.children.values()):
        assert isinstance(task_result, InferenceResult)
        assert task_result.usage is not None
        assert task_result.usage.num_requests == 1


def test_each_task_writes_its_own_export_and_lineage(tmp_path: Path) -> None:
    _run(
        _spec("alpha", postprocess=_EXPORT),
        [_child("tsk-b", _spec("bravo", postprocess=_EXPORT))],
        tmp_path,
    )

    for task, prompt in (("tsk-a", "alpha"), ("tsk-b", "bravo")):
        rows = (tmp_path / task / "artifacts" / "rows.jsonl").read_text().splitlines()
        assert [json.loads(row) for row in rows] == [{"answer": f"out-{prompt}"}]
        assets = (tmp_path / task / "logs" / "assets.jsonl").read_text()
        assert [json.loads(row)["data_id"] for row in assets.splitlines()] == [task]


def test_a_child_whose_own_input_fails_is_left_out(tmp_path: Path) -> None:
    missing = _spec("x") | {"data": {"type": "list", "items": []}}
    result, model = _run(
        _spec("alpha"),
        [_child("tsk-empty", missing), _child("tsk-c", _spec("charlie"))],
        tmp_path,
    )

    assert _outputs(result) == {"tsk-c": ["out-charlie"]}
    assert len(model.generate.call_args.kwargs["input_ids"]) == 2


def test_a_childs_unexpected_error_fails_the_dispatch(tmp_path: Path) -> None:
    executor_cls = HFTransformersExecutor
    original = executor_cls._collect_prompts_for_spec

    def _collect(self: Any, spec: Any, task_id: str, **kwargs: Any) -> Any:
        if task_id == "tsk-b":
            raise RuntimeError("unexpected")
        return original(self, spec, task_id=task_id, **kwargs)

    with patch.object(executor_cls, "_collect_prompts_for_spec", _collect):
        with pytest.raises(RuntimeError):
            _run(_spec("alpha"), [_child("tsk-b", _spec("bravo"))], tmp_path)


def test_the_parents_own_failure_fails_the_dispatch(tmp_path: Path) -> None:
    empty = _spec("x") | {"data": {"type": "list", "items": []}}

    with pytest.raises(ExecutionError):
        _run(empty, [_child("tsk-c", _spec("charlie"))], tmp_path)
