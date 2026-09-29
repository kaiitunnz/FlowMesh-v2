"""A merged vLLM dispatch returns a result for each child it can run, and only those."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

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
    out_dir: Path,
    rejected: str | None = None,
    chat_template: str | None = None,
    empty: str | None = None,
    window: int = 4096,
) -> tuple[InferenceResult, MagicMock]:
    """Run a merged dispatch against a stand-in engine, which aborts the whole batch
    when it rejects the ``rejected`` prompt, as vLLM does, and reports no completion
    for the ``empty`` prompt. A ``chat_template`` renders each conversation as its last
    message's content. The tokenizer counts one token per character against the
    engine's ``window``."""
    executor = VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    llm = MagicMock()
    llm.llm_engine.model_config.max_model_len = window
    tokenizer = llm.get_tokenizer.return_value
    tokenizer.encode.side_effect = lambda text: [0] * len(text)
    tokenizer.chat_template = chat_template
    tokenizer.apply_chat_template.side_effect = lambda messages, **_kwargs: messages[
        -1
    ]["content"]

    def _generate(prompts: list[Any], **_kwargs: Any) -> list[SimpleNamespace]:
        if rejected in prompts:
            raise ValueError("The decoder prompt is longer than max_model_len")
        return [
            SimpleNamespace(
                outputs=(
                    []
                    if prompt == empty
                    else [
                        SimpleNamespace(
                            text=f"out-{prompt}", finish_reason="stop", token_ids=[1]
                        )
                    ]
                ),
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


def test_an_output_with_no_completion_fails_the_dispatch(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match="task=tsk-b, prompt_index=0"):
        _run(
            _spec("parent"),
            [_child("tsk-ok", _spec("ok")), _child("tsk-b", _spec("blank"))],
            tmp_path,
            empty="blank",
        )


def test_each_task_writes_its_own_export_and_lineage(tmp_path: Path) -> None:
    export = {"jsonl_export": {"path": "rows.jsonl", "fields": {"answer": "output"}}}
    _run(
        _spec("parent", postprocess=export),
        [_child("tsk-b", _spec("bravo", postprocess=export), owner_id="usr-b")],
        tmp_path / "tsk-test",
    )

    for task, prompt, owner in (
        ("tsk-test", "parent", "usr-test"),
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
    parent = _spec("parent", postprocess=export)
    parent["data"]["metadata"] = [{"tag": "x"}]

    result, _ = _run(
        parent, [_child("tsk-b", _spec("bravo", postprocess=export))], tmp_path
    )

    assert result.children == {}
    assert (tmp_path / "artifacts" / "rows.jsonl").is_file()


def test_the_parent_reports_only_its_own_usage(tmp_path: Path) -> None:
    result, _ = _run(
        _spec("parent"),
        [_child("tsk-b", _spec("bravo")), _child("tsk-c", _spec("charlie"))],
        tmp_path,
    )

    for task_result in (result, *result.children.values()):
        assert isinstance(task_result, InferenceResult)
        assert task_result.usage is not None
        assert task_result.usage.model_dump(
            include={"prompt_tokens", "completion_tokens", "num_requests"}
        ) == {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "num_requests": 1,
        }


def _table_spec(*groups: list[str]) -> dict[str, Any]:
    return {
        "taskType": "inference",
        "model": _MODEL,
        "data": {
            "type": "dataframe",
            "columns": [
                {"label": "q", "data": {"type": "list", "items": list(groups)}}
            ],
            "messages": [{"role": "user", "content": "Q: {q}"}],
        },
    }


def test_a_merged_table_child_gets_the_items_it_gets_alone(tmp_path: Path) -> None:
    parent = _table_spec(["a", "b"], ["c"])
    child = _table_spec(["d"], ["e", "f"])

    merged, _ = _run(
        parent, [_child("tsk-b", child)], tmp_path / "merged", chat_template="chat"
    )
    parent_alone, _ = _run(parent, [], tmp_path / "parent", chat_template="chat")
    child_alone, _ = _run(child, [], tmp_path / "child", chat_template="chat")

    merged_child = merged.children["tsk-b"]
    assert isinstance(merged_child, InferenceResult)
    assert merged.items == parent_alone.items
    assert merged_child.items == child_alone.items
    assert [
        (item.index, item.prompt, item.output, item.finish_reason)
        for item in merged_child.items
    ] == [
        (0, "Q: d", ["out-Q: d"], ["stop"]),
        (1, "Q: e", ["out-Q: e", "out-Q: f"], ["stop", "stop"]),
    ]


def test_a_merged_child_reports_its_own_max_tokens_cap(tmp_path: Path) -> None:
    long_prompt = "x" * 600
    merged, llm = _run(
        _spec("parent"),
        [_child("tsk-b", _spec(long_prompt))],
        tmp_path / "merged",
        window=1024,
    )
    child_alone, _ = _run(_spec(long_prompt), [], tmp_path / "child", window=1024)

    params = llm.generate.call_args.kwargs["sampling_params"]
    assert [p.max_tokens for p in params] == [512, 1024 - 600 - 16]
    merged_child = merged.children["tsk-b"]
    assert isinstance(merged_child, InferenceResult)
    assert merged_child.items == child_alone.items
    assert merged_child.items[0].diagnostics == {
        "auto_cap": {"max_tokens": 408, "requested": 512}
    }
    assert merged.items[0].diagnostics is None


def test_a_table_item_reports_one_cap_per_row(tmp_path: Path) -> None:
    long_row = "y" * 600
    result, _ = _run(
        _table_spec(["a", long_row]), [], tmp_path, chat_template="chat", window=1024
    )

    assert result.items[0].diagnostics == {
        "auto_cap": {
            "max_tokens": [None, 1024 - len(f"Q: {long_row}") - 16],
            "requested": 512,
        }
    }


def test_an_item_missing_a_required_field_fails_the_task(tmp_path: Path) -> None:
    # A grouping that reports only the outputs fails the producer check.
    def outputs_only(
        items: list[dict[str, Any]], tables: list[Any]
    ) -> list[dict[str, Any]]:
        return [{"output": [item["output"] for item in items]}]

    with (
        patch.object(VLLMExecutor, "_populate_table", staticmethod(outputs_only)),
        pytest.raises(ExecutionError, match="cannot report"),
    ):
        _run(_table_spec(["a"]), [], tmp_path, chat_template="chat")
