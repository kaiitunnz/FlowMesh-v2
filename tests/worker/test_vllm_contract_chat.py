"""A self-contained run of a resolved contract generates from its conversations.

Both embodiments of a leaf issue one request. A replica has the engine apply the model's
chat template to the conversation it is sent, so a local generation renders the same
conversation the same way, exactly once.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference-gpu)")
pytest.importorskip("torch", reason="torch not installed (needs --extra inference)")

from shared.inference import CanonicalInferenceRequest
from shared.schemas.result import InferenceResult
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


def _run(
    prompts: list[str],
    contract: CanonicalInferenceRequest | None,
    out_dir: Path,
    chat_template: str | None = "a-chat-template",
    window: int = 4096,
) -> tuple[InferenceResult, MagicMock]:
    """Run one task through the executor against a stand-in engine.

    The stand-in tokenizer carries a chat template unless a test takes it away, which is
    what decides whether conversations can be rendered at all. Each prompt it encodes
    counts 100 tokens against the engine's ``window``.
    """
    executor = VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    llm = MagicMock()
    llm.llm_engine.model_config.max_model_len = window
    llm.get_tokenizer.return_value.encode.return_value = [0] * 100
    llm.get_tokenizer.return_value.chat_template = chat_template
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
    msg.resolved_contract = contract
    result = executor.run(msg, out_dir)
    assert isinstance(result, InferenceResult)
    return result, llm


def _contract(*prompts: str) -> CanonicalInferenceRequest:
    return CanonicalInferenceRequest(
        model="Qwen/Qwen3-4B", prompts=tuple(prompts), params={"max_tokens": 8}
    )


def test_a_resolved_contract_generates_from_its_conversations(tmp_path: Path) -> None:
    _, llm = _run(["a", "b", "c"], _contract("a", "b", "c"), tmp_path)

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
    _, llm = _run(["a"], _contract("a"), tmp_path)

    llm.get_tokenizer.return_value.apply_chat_template.assert_not_called()


def test_a_leaf_without_a_contract_still_renders_and_generates_its_prompts(
    tmp_path: Path,
) -> None:
    # The path every task without a resolved contract takes is unchanged.
    _, llm = _run(["a", "b"], None, tmp_path)

    llm.chat.assert_not_called()
    llm.get_tokenizer.return_value.apply_chat_template.assert_called()
    assert llm.generate.call_args.args[0] == ["<rendered>", "<rendered>"]


def test_a_model_without_a_chat_template_generates_from_its_prompts(
    tmp_path: Path,
) -> None:
    # A contract names conversations, and only a model whose tokenizer carries a chat
    # template can render one. A base model generates from the prompts its spec
    # prepared, as a leaf carrying no contract does, rather than failing on a template
    # it does not have.
    _, llm = _run(["a", "b"], _contract("a", "b"), tmp_path, chat_template=None)

    llm.chat.assert_not_called()
    assert llm.generate.call_args.args[0] == ["a", "b"]


def test_a_model_without_a_chat_template_still_stores_the_declared_shape(
    tmp_path: Path,
) -> None:
    # The result projection is embodiment-blind and reads the contract either way, so
    # the generation path a model forces does not change what the leaf reports.
    _, llm = _run(["a"], _contract("a"), tmp_path, chat_template=None)

    assert llm.generate.called
    llm.get_tokenizer.return_value.apply_chat_template.assert_not_called()


def _sampling_params(llm: MagicMock) -> Any:
    call = llm.chat.call_args or llm.generate.call_args
    return call.kwargs["sampling_params"]


def test_a_leaf_without_a_contract_caps_max_tokens_to_the_window(
    tmp_path: Path,
) -> None:
    result, llm = _run(["a"], None, tmp_path, window=356)

    params = _sampling_params(llm)
    assert isinstance(params, list)
    assert [p.max_tokens for p in params] == [356 - 100 - 16]
    assert result.items[0].diagnostics == {
        "auto_cap": {"max_tokens": 240, "requested": 512}
    }


@pytest.mark.parametrize("chat_template", ["a-chat-template", None])
def test_a_contract_runs_its_declared_max_tokens_on_either_path(
    tmp_path: Path, chat_template: str | None
) -> None:
    # A replica rejects an over-window request rather than clamping it, so a local run
    # of a contract issues the same request, whether the engine renders conversations
    # or generates from prompts.
    result, llm = _run(
        ["a"], _contract("a"), tmp_path, chat_template=chat_template, window=356
    )

    params = _sampling_params(llm)
    assert not isinstance(params, list)
    assert params.max_tokens == 512
    assert result.items[0].diagnostics is None
