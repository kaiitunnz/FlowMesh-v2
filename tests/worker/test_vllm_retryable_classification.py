"""The retryable classification of vLLM engine-init failures.

A load that found less than its requested memory free may fit on another worker, so it
is retryable; a deterministic failure fails fast. The engine and the GPU memory reading
are mocked, so no GPU or model download is needed.
"""

from collections.abc import Callable
from unittest.mock import patch

import pytest

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import EmbeddingSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors import vllm_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.vllm_embedding_executor import VLLMEmbeddingExecutor
from worker.executors.vllm_executor import VLLMExecutor

torch = pytest.importorskip("torch", reason="torch not installed")

_TOTAL_BYTES = 1000


def _init_inference_engine(requested_util: float, tensor_parallel_size: int) -> None:
    VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)._init_vllm_engine(
        ident="org/model",
        vllm_cfg={"gpu_memory_utilization": requested_util},
        checkpoint_cfg={},
        new_inference_spec={},
        requested_gpu_count=1,
        revision=None,
        extra_llm_kwargs={},
        adjust_tp=lambda _size: tensor_parallel_size,
        task_ids=None,
    )


def _init_embedding_engine(requested_util: float, _tensor_parallel_size: int) -> None:
    spec = EmbeddingSpecStrict(
        taskType=TaskType.EMBEDDING,
        model=ModelConfig(
            source=ModelSource(identifier="org/embed"),
            vllm={"gpu_memory_utilization": requested_util},
        ),
        data={"type": "list", "items": ["a"]},
    )
    VLLMEmbeddingExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)._ensure_embedding_llm(
        spec
    )


def _init_failure(
    init: Callable[[float, int], None],
    requested_util: float,
    free_ratios: list[float] | None,
    tensor_parallel_size: int = 1,
) -> ExecutionError:
    """Drive an engine init whose engine fails, reading ``free_ratios`` in turn as the
    GPU's free memory, or no GPU when ``None``."""
    readings = [(int(r * _TOTAL_BYTES), _TOTAL_BYTES) for r in free_ratios or []]
    with (
        patch.object(torch.cuda, "is_available", return_value=free_ratios is not None),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(torch.cuda, "mem_get_info", side_effect=readings),
        patch.object(vllm_executor, "LLM", side_effect=RuntimeError("boom")),
        pytest.raises(ExecutionError) as excinfo,
    ):
        init(requested_util, tensor_parallel_size)
    return excinfo.value


_INITS = pytest.mark.parametrize(
    "init",
    [_init_inference_engine, _init_embedding_engine],
    ids=["inference", "embedding"],
)


@_INITS
@pytest.mark.parametrize("requested,free", [(0.9, 0.04), (0.9, 0.5)])
def test_memory_constrained_failure_is_retryable(
    init: Callable[[float, int], None], requested: float, free: float
) -> None:
    assert _init_failure(init, requested, [free]).retryable is True


@_INITS
@pytest.mark.parametrize(
    "requested,free_ratios",
    [(0.9, None), (0.9, [0.99]), (0.9, [0.94]), (0.95, [0.99])],
    ids=["unmeasured", "ample", "within-margin", "high-request"],
)
def test_unconstrained_failure_is_not_retryable(
    init: Callable[[float, int], None],
    requested: float,
    free_ratios: list[float] | None,
) -> None:
    assert _init_failure(init, requested, free_ratios).retryable is False


def test_a_reading_after_a_failed_attempt_does_not_count() -> None:
    error = _init_failure(
        _init_inference_engine, 0.9, [0.99, 0.1], tensor_parallel_size=2
    )

    assert error.retryable is False
