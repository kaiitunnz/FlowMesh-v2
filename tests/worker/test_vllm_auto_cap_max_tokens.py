"""Tests for vLLM max_tokens auto-capping — the pure clamp, plus the wiring
around it, which runs on a bare instance with fakes and needs no GPU deps.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from worker.executors.vllm_executor import (
    _AUTO_CAP_MIN_OUTPUT,
    VLLMExecutor,
    _auto_capped_max_tokens,
)


# ── pure clamp ───────────────────────────────────────────────────────────
def test_clamp_fits_output_to_window() -> None:
    # the field bug: 65536 requested, 40960 window, 8193-tok prompt
    assert _auto_capped_max_tokens(65536, 8193, 40960) == 40960 - 8193 - 16


def test_clamp_untouched_when_window_unknown() -> None:
    assert _auto_capped_max_tokens(65536, 8193, None) == 65536
    assert _auto_capped_max_tokens(65536, 8193, 0) == 65536


def test_clamp_untouched_when_request_already_fits() -> None:
    assert _auto_capped_max_tokens(512, 8193, 40960) == 512


def test_clamp_floors_when_prompt_at_or_over_window() -> None:
    assert _auto_capped_max_tokens(65536, 40960, 40960) == _AUTO_CAP_MIN_OUTPUT
    assert _auto_capped_max_tokens(65536, 50000, 40960) == _AUTO_CAP_MIN_OUTPUT


def test_clamp_never_exceeds_requested_even_below_floor() -> None:
    assert _auto_capped_max_tokens(8, 40950, 40960) == 8


# ── wiring ───────────────────────────────────────────────────────────────
def _executor_with(
    window: int | None,
    prompts: list[Any],
    tok_len: int | Callable[[str], int],
) -> VLLMExecutor:
    ex = object.__new__(VLLMExecutor)  # skip __init__ (no GPU deps)
    ex._batched_inputs = prompts
    ex._llm = SimpleNamespace(  # type: ignore[assignment]
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=window))
    )

    def _encode(s: str) -> list[int]:
        return [0] * (tok_len(s) if callable(tok_len) else tok_len)

    ex._get_tokenizer = lambda: SimpleNamespace(  # type: ignore[method-assign]
        encode=_encode
    )
    return ex


def _sp(max_tokens: int) -> MagicMock:
    sp = MagicMock()
    sp.max_tokens = max_tokens
    # fresh clone per call; a shared mock would let one clamp overwrite another
    sp.clone.side_effect = lambda: SimpleNamespace(max_tokens=None)
    return sp


def test_wiring_returns_shared_object_when_window_unknown() -> None:
    ex = _executor_with(None, ["hello"], 5)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}


def test_wiring_returns_shared_object_when_nothing_clamped() -> None:
    # short prompts and a small request sit well within the window
    ex = _executor_with(40960, ["a", "b"], 10)
    sp = _sp(512)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}


def test_wiring_builds_per_prompt_list_and_clamps() -> None:
    ex = _executor_with(40960, ["big-prompt"], 8193)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert isinstance(out, list) and len(out) == 1
    assert out[0].max_tokens == 40960 - 8193 - 16
    assert capped == {0: {"max_tokens": 40960 - 8193 - 16, "requested": 65536}}


def test_wiring_leaves_multimodal_prompts_at_requested() -> None:
    # non-str entries are multimodal and keep the requested budget
    ex = _executor_with(40960, [{"prompt": "x", "multi_modal_data": {}}], 8193)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}


def test_wiring_clamps_only_the_oversized_prompt_in_a_mixed_batch() -> None:
    # requested=35000 fits after the 10-tok prompt but not the 8193-tok one
    ex = _executor_with(40960, ["big", "small"], lambda s: 8193 if s == "big" else 10)
    sp = _sp(35000)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert isinstance(out, list) and len(out) == 2
    assert out[0].max_tokens == 40960 - 8193 - 16
    assert out[1] is sp
    assert capped == {0: {"max_tokens": 40960 - 8193 - 16, "requested": 35000}}


def test_wiring_tokenizer_failure_skips_clamp() -> None:
    def _raise(_s: str) -> int:
        raise RuntimeError("tokenizer unavailable")

    ex = _executor_with(40960, ["big-prompt"], _raise)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}
