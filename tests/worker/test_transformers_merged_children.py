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
_PAD, _EOS, _OUTPUT_OFFSET = 0, 99, 1000


class _Tokenizer:
    """One token per word, left-padded, with a pad token distinct from EOS."""

    chat_template: str | None = None
    pad_token_id = _PAD
    eos_token_id = _EOS

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"prompts": list(prompts), **kwargs})
        rows = [
            [self.vocab.setdefault(word, len(self.vocab) + 1) for word in p.split()]
            for p in prompts
        ]
        width = max(len(row) for row in rows)
        return {
            "input_ids": torch.tensor([[_PAD] * (width - len(r)) + r for r in rows]),
            "attention_mask": torch.tensor(
                [[0] * (width - len(r)) + [1] * len(r) for r in rows]
            ),
        }

    def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
        words = {token: word for word, token in self.vocab.items()}
        return " ".join(
            (
                f"out-{words[int(t) - _OUTPUT_OFFSET]}"
                if int(t) > _OUTPUT_OFFSET
                else "<s>"
            )
            for t in tokens
            if not (skip_special_tokens and int(t) in (_PAD, _EOS))
        )


def _generate(input_ids: Any, attention_mask: Any, **_kwargs: Any) -> Any:
    """Answers each word of a prompt, then EOS, padding rows to the longest answer."""
    answers = [
        [int(t) + _OUTPUT_OFFSET for t, m in zip(row, mask, strict=True) if m] + [_EOS]
        for row, mask in zip(input_ids, attention_mask, strict=True)
    ]
    width = max(len(answer) for answer in answers)
    padded = [answer + [_PAD] * (width - len(answer)) for answer in answers]
    return torch.cat([input_ids, torch.tensor(padded)], dim=1)


def _spec(prompt: str, **fields: Any) -> dict[str, Any]:
    return {
        "taskType": "inference",
        "model": _MODEL,
        "data": {"type": "list", "items": [prompt]},
        **fields,
    }


def _child(
    task_id: str, spec: dict[str, Any], owner_id: str = "usr-test"
) -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id=task_id,
        owner_id=owner_id,
        workflow_id="wfl-test",
        spec=TypeAdapter(TaskSpecStrict).validate_python(spec),
    )


def _run(
    parent: dict[str, Any],
    children: list[MergedChildTaskStrict],
    results_dir: Path,
    tokenizer: Any = None,
    task_id: str = "tsk-a",
) -> tuple[InferenceResult, MagicMock]:
    executor = HFTransformersExecutor(DEFAULT_WORKER_CONFIG)
    executor._tok = tokenizer or _Tokenizer()  # type: ignore[assignment]
    model = MagicMock()
    model.generate.side_effect = _generate
    executor._model = model
    executor._device = "cpu"
    executor._model_name = "org/model"
    msg = make_worker_task_message(
        parent,
        task_type=TaskType.INFERENCE,
        task_id=task_id,
        merged_children=children,
    )
    with patch.object(executor, "_ensure_model"):
        result = executor.run(msg, results_dir / task_id)
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


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_a_merged_task_gets_what_it_would_alone(
    tmp_path: Path, skip_special_tokens: bool
) -> None:
    inference = {"max_new_tokens": 8, "skip_special_tokens": skip_special_tokens}
    child_spec = _spec("zulu", inference=inference)
    merged, _ = _run(
        _spec("a b c d e", inference=inference),
        [_child("tsk-b", child_spec)],
        tmp_path / "merged",
    )
    alone, _ = _run(child_spec, [], tmp_path / "alone", task_id="tsk-b")

    merged_child = merged.children["tsk-b"]
    assert isinstance(merged_child, InferenceResult)
    assert merged_child.items == alone.items
    assert merged_child.items[0].finish_reason == "stop"
    assert merged_child.usage is not None and alone.usage is not None
    assert merged_child.usage.model_dump(exclude={"latency_sec"}) == (
        alone.usage.model_dump(exclude={"latency_sec"})
    )


@pytest.mark.parametrize(
    ("generated", "pad", "own"),
    [
        ([5, _EOS, 6, _PAD], _PAD, [5, _EOS]),
        ([5, 6, _PAD, _PAD], _PAD, [5, 6]),
        ([_PAD, _PAD], _PAD, []),
        ([5, 6, _PAD], None, [5, 6, _PAD]),
    ],
    ids=["through-eos", "trailing-pads", "all-pads", "no-pad-token"],
)
def test_a_rows_own_generation(
    generated: list[int], pad: int | None, own: list[int]
) -> None:
    executor = HFTransformersExecutor(DEFAULT_WORKER_CONFIG)
    tokenizer = _Tokenizer()
    tokenizer.pad_token_id = pad  # type: ignore[assignment]
    executor._tok = tokenizer  # type: ignore[assignment]

    assert executor._own_generation(torch.tensor(generated)).tolist() == own


def test_each_task_writes_its_own_export_and_lineage(tmp_path: Path) -> None:
    _run(
        _spec("alpha", postprocess=_EXPORT),
        [_child("tsk-b", _spec("bravo", postprocess=_EXPORT), owner_id="usr-b")],
        tmp_path,
    )

    for task, prompt, owner in (
        ("tsk-a", "alpha", "usr-test"),
        ("tsk-b", "bravo", "usr-b"),
    ):
        rows = (tmp_path / task / "artifacts" / "rows.jsonl").read_text().splitlines()
        assert [json.loads(row) for row in rows] == [{"answer": f"out-{prompt}"}]
        assets = (tmp_path / task / "logs" / "assets.jsonl").read_text()
        assert [
            (json.loads(row)["data_id"], json.loads(row)["user_id"])
            for row in assets.splitlines()
        ] == [(task, owner)]


def test_a_child_whose_own_export_fails_is_left_out(tmp_path: Path) -> None:
    export = {
        "jsonl_export": {
            "path": "rows.jsonl",
            "fields": {"answer": "output", "tag": "metadata.tag"},
            "required_fields": ["tag"],
        }
    }
    parent = _spec("alpha", postprocess=export)
    parent["data"]["metadata"] = [{"tag": "x"}]

    result, _ = _run(
        parent, [_child("tsk-b", _spec("bravo", postprocess=export))], tmp_path
    )

    assert result.children == {}
    assert (tmp_path / "tsk-a" / "artifacts" / "rows.jsonl").is_file()
    assert not (tmp_path / "tsk-b" / "logs" / "assets.jsonl").exists()


def test_a_child_whose_own_input_fails_is_left_out(tmp_path: Path) -> None:
    missing = _spec("x") | {"data": {"type": "list", "items": []}}
    result, model = _run(
        _spec("alpha"),
        [_child("tsk-empty", missing), _child("tsk-c", _spec("charlie"))],
        tmp_path,
    )

    assert _outputs(result) == {"tsk-c": ["out-charlie"]}
    assert len(model.generate.call_args.kwargs["input_ids"]) == 2


def test_any_error_in_a_childs_own_preparation_leaves_it_out(tmp_path: Path) -> None:
    original = HFTransformersExecutor._collect_prompts_for_spec

    def _collect(self: Any, spec: Any, task_id: str, **kwargs: Any) -> Any:
        if task_id == "tsk-b":
            raise KeyError("row")
        return original(self, spec, task_id=task_id, **kwargs)

    with patch.object(HFTransformersExecutor, "_collect_prompts_for_spec", _collect):
        result, _ = _run(_spec("alpha"), [_child("tsk-b", _spec("bravo"))], tmp_path)

    assert result.children == {}


def test_a_child_templated_differently_is_left_out(tmp_path: Path) -> None:
    class _ChatTokenizer(_Tokenizer):
        chat_template = "template"

        def apply_chat_template(self, messages: Any, **_kwargs: Any) -> str:
            return " ".join(message["content"] for message in messages)

    parent = _spec("alpha", inference={"apply_chat_template": False})
    child = _spec("x", inference={"apply_chat_template": False}) | {
        "data": {"type": "list", "items": [[{"role": "user", "content": "bravo"}]]}
    }
    tokenizer = _ChatTokenizer()

    result, _ = _run(parent, [_child("tsk-b", child)], tmp_path, tokenizer)

    assert result.children == {}
    assert tokenizer.calls[-1]["prompts"] == ["alpha"]


def test_a_failed_generation_fails_the_dispatch(tmp_path: Path) -> None:
    with patch(f"{__name__}._generate", side_effect=RuntimeError("engine")):
        with pytest.raises(RuntimeError, match="engine"):
            _run(_spec("alpha"), [_child("tsk-b", _spec("bravo"))], tmp_path)


def test_the_parents_own_failure_fails_the_dispatch(tmp_path: Path) -> None:
    empty = _spec("x") | {"data": {"type": "list", "items": []}}

    with pytest.raises(ExecutionError):
        _run(empty, [_child("tsk-c", _spec("charlie"))], tmp_path)
