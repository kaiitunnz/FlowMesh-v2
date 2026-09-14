"""A self-contained run of a resolved contract generates from its conversations.

Both embodiments of a leaf have to issue one request. A replica applies the model's chat
template to the conversation it is sent, so a local generation that fed the raw prompt
straight to the engine was running a different request from the same declaration.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference-gpu)")
pytest.importorskip("torch", reason="torch not installed (needs --extra inference)")

from shared.inference import CanonicalInferenceRequest
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    DEFAULT_WORKER_CONFIG,
    make_worker_task_message,
)
from worker.executors.vllm_executor import VLLMExecutor


def _completion(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(text=text, finish_reason="stop", token_ids=[1])],
        prompt_token_ids=[1, 2],
    )


def _run(prompts: list[str], contract: str | None, out_dir: Path) -> MagicMock:
    """Run one task through the executor against a stand-in engine.

    The stand-in tokenizer carries a chat template, which is what makes the executor
    render prompts itself unless a contract says the engine will.
    """
    executor = VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    llm = MagicMock()
    llm.get_tokenizer.return_value.chat_template = "a-chat-template"
    llm.get_tokenizer.return_value.apply_chat_template.return_value = "<rendered>"
    llm.chat.return_value = [_completion(f"out-{i}") for i in range(len(prompts))]
    llm.generate.return_value = llm.chat.return_value

    def _ensure(*_args: Any, **_kwargs: Any) -> None:
        executor._llm = llm

    executor._ensure_llm = _ensure  # type: ignore[method-assign]
    msg = make_worker_task_message(
        {
            "taskType": "inference",
            "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
            "data": {"type": "list", "items": prompts},
        },
        task_type=TaskType.INFERENCE,
    )
    msg.declared_contract = contract
    executor.run(msg, out_dir)
    return llm


def _contract(*prompts: str) -> str:
    return CanonicalInferenceRequest(
        model="Qwen/Qwen3-4B", prompts=tuple(prompts), params={"max_tokens": 8}
    ).model_dump_json()


def test_a_resolved_contract_generates_from_its_conversations(tmp_path: Path) -> None:
    llm = _run(["a", "b", "c"], _contract("a", "b", "c"), tmp_path)

    llm.generate.assert_not_called()
    conversations = llm.chat.call_args.args[0]
    assert conversations == [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
        [{"role": "user", "content": "c"}],
    ]


def test_the_engine_is_the_only_place_a_contract_is_rendered(tmp_path: Path) -> None:
    # Rendering here as well would apply the model's chat template twice and prepend a
    # second sequence of special tokens, which is not the request the replica runs.
    llm = _run(["a"], _contract("a"), tmp_path)

    llm.get_tokenizer.return_value.apply_chat_template.assert_not_called()


def test_a_leaf_without_a_contract_still_renders_and_generates_its_prompts(
    tmp_path: Path,
) -> None:
    # The path every task without a resolved contract takes is unchanged.
    llm = _run(["a", "b"], None, tmp_path)

    llm.chat.assert_not_called()
    llm.get_tokenizer.return_value.apply_chat_template.assert_called()
    assert llm.generate.call_args.args[0] == ["<rendered>", "<rendered>"]
