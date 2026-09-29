"""The retryable classification of vLLM engine-init failures.

A load that had to shrink below its requested memory may fit on another worker, so it
is retryable; a deterministic failure fails fast. The engine is mocked, so no GPU or
model download is needed.
"""

from collections.abc import Callable
from unittest.mock import patch

import pytest

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import EmbeddingSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors.base_executor import ExecutionError
from worker.executors.vllm_embedding_executor import VLLMEmbeddingExecutor
from worker.executors.vllm_executor import VLLMExecutor

MEMORY_CONSTRAINED_FREE_RATIO = 0.04
AMPLE_FREE_RATIO = 0.99
REQUESTED_UTIL = 0.9


def _init_inference_engine() -> None:
    VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)._init_vllm_engine(
        ident="org/model",
        vllm_cfg={"gpu_memory_utilization": REQUESTED_UTIL},
        checkpoint_cfg={},
        new_inference_spec={},
        requested_gpu_count=1,
        revision=None,
        extra_llm_kwargs={},
        adjust_tp=lambda size: size,
        task_ids=None,
    )


def _init_embedding_engine() -> None:
    spec = EmbeddingSpecStrict(
        taskType=TaskType.EMBEDDING,
        model=ModelConfig(
            source=ModelSource(identifier="org/embed"),
            vllm={"gpu_memory_utilization": REQUESTED_UTIL},
        ),
        data={"type": "list", "items": ["a"]},
    )
    VLLMEmbeddingExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)._ensure_embedding_llm(
        spec
    )


def _init_failure(init: Callable[[], None], free_ratio: float | None) -> ExecutionError:
    """Drive an engine init with a mocked memory signal and an engine that fails."""

    def _safe_util(requested_util: float) -> tuple[float, float | None]:
        if free_ratio is None:
            return requested_util, None
        if free_ratio - 0.05 >= requested_util:
            return requested_util, free_ratio
        return max(0.02, free_ratio * 0.8), free_ratio

    with (
        patch.object(VLLMExecutor, "_compute_safe_utilization", side_effect=_safe_util),
        patch("worker.executors.vllm_executor.LLM", side_effect=RuntimeError("boom")),
        pytest.raises(ExecutionError) as excinfo,
    ):
        init()
    return excinfo.value


_INITS = pytest.mark.parametrize(
    "init",
    [_init_inference_engine, _init_embedding_engine],
    ids=["inference", "embedding"],
)


@_INITS
def test_memory_constrained_failure_is_retryable(init: Callable[[], None]) -> None:
    assert _init_failure(init, MEMORY_CONSTRAINED_FREE_RATIO).retryable is True


@_INITS
@pytest.mark.parametrize(
    "free_ratio", [None, AMPLE_FREE_RATIO], ids=["unmeasured", "ample"]
)
def test_unconstrained_failure_is_not_retryable(
    init: Callable[[], None], free_ratio: float | None
) -> None:
    assert _init_failure(init, free_ratio).retryable is False
