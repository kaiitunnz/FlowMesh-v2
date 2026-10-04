"""A self-contained run of a resolved contract generates from its conversations.

Both embodiments of a leaf issue one request. A replica has the engine apply the model's
chat template to the conversation it is sent, so a local generation renders the same
conversation the same way, exactly once.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference-gpu)")
pytest.importorskip("torch", reason="torch not installed (needs --extra inference)")

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.renderers.params import TokenizeParams

from shared.inference import CanonicalInferenceRequest
from shared.schemas.result import InferenceResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    DEFAULT_WORKER_CONFIG,
    make_worker_task_message,
)
from worker.executors.base_executor import ExecutionError
from worker.executors.vllm_executor import VLLMExecutor

# The tokens a stand-in engine counts for each prompt or rendered conversation.
_PROMPT_TOKENS = 100


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
    what decides whether conversations can be rendered at all. The engine counts
    ``_PROMPT_TOKENS`` for each input and checks it against its ``window`` with vLLM's
    own tokenize parameters, as the offline renderer does.
    """
    executor = VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    llm = MagicMock()
    llm.llm_engine.model_config.max_model_len = window
    llm.get_tokenizer.return_value.chat_template = chat_template
    llm.get_tokenizer.return_value.apply_chat_template.return_value = "<rendered>"

    def _engine(
        inputs: list[Any],
        tokenization_kwargs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> list[SimpleNamespace]:
        params = TokenizeParams(max_total_tokens=window).with_kwargs(
            **(tokenization_kwargs or {})
        )
        for _ in inputs:
            params.apply_post_tokenization(
                None, {"prompt_token_ids": [0] * _PROMPT_TOKENS}
            )
        return [_completion(f"out-{i}") for i in range(len(inputs))]

    llm.chat.side_effect = _engine
    llm.generate.side_effect = _engine

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


def test_a_model_without_a_chat_template_generates_without_rendering(
    tmp_path: Path,
) -> None:
    # The result projection is embodiment-blind and reads the contract either way, so
    # the generation path a model forces does not change what the leaf reports.
    _, llm = _run(["a"], _contract("a"), tmp_path, chat_template=None)

    assert llm.generate.called
    llm.get_tokenizer.return_value.apply_chat_template.assert_not_called()


@pytest.mark.parametrize("prompt_tokens", [99, 100])
def test_a_local_bound_admits_what_the_server_admits(prompt_tokens: int) -> None:
    window, max_tokens = 107, 8
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": max_tokens,
        }
    )
    model_config = cast(ModelConfig, SimpleNamespace(max_model_len=window))
    server = request.build_tok_params(model_config)
    local = TokenizeParams(max_total_tokens=window).with_kwargs(
        max_length=window - max_tokens
    )

    def admits(params: TokenizeParams) -> bool:
        try:
            params.apply_post_tokenization(
                None, {"prompt_token_ids": [0] * prompt_tokens}
            )
        except VLLMValidationError:
            return False
        return True

    assert admits(local) == admits(server) == (prompt_tokens + max_tokens <= window)


@pytest.mark.parametrize("chat_template", ["a-chat-template", None])
def test_a_contract_over_the_window_fails_on_either_path(
    tmp_path: Path, chat_template: str | None
) -> None:
    # A replica's server rejects a prompt plus max_tokens past the window, and a local
    # run of the same contract refuses it with the same count.
    window = _PROMPT_TOKENS + 8 - 1
    with pytest.raises(ExecutionError, match="maximum context length") as failure:
        _run(
            ["a"], _contract("a"), tmp_path, chat_template=chat_template, window=window
        )
    assert failure.value.retryable is False
    assert f"{_PROMPT_TOKENS} input tokens" in str(failure.value)


@pytest.mark.parametrize("chat_template", ["a-chat-template", None])
def test_a_contract_exactly_filling_the_window_runs(
    tmp_path: Path, chat_template: str | None
) -> None:
    window = _PROMPT_TOKENS + 8
    result, _ = _run(
        ["a"], _contract("a"), tmp_path, chat_template=chat_template, window=window
    )
    assert [item.output for item in result.items] == ["out-0"]


def test_a_leaf_without_a_contract_runs_over_the_window(tmp_path: Path) -> None:
    result, _ = _run(["a"], None, tmp_path, window=_PROMPT_TOKENS + 8 - 1)
    assert [item.output for item in result.items] == ["out-0"]


def test_a_contract_without_max_tokens_is_not_bounded(tmp_path: Path) -> None:
    contract = CanonicalInferenceRequest(model="Qwen/Qwen3-4B", prompts=("a",))
    result, _ = _run(["a"], contract, tmp_path, window=_PROMPT_TOKENS)
    assert [item.output for item in result.items] == ["out-0"]
