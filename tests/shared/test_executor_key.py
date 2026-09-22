"""A task spec resolves to the executor a worker runs it on."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from shared.tasks.executor_key import ExecutorKey, resolve_executor_key
from shared.tasks.specs import (
    EmbeddingSpecStrict,
    InferenceSpecStrict,
    InferenceSpecTemplate,
)
from shared.tasks.task_type import TaskType
from worker.executors import EXECUTOR_MODULES

_MODEL: dict[str, Any] = {"source": {"identifier": "Qwen/Qwen3-0.6B"}}


def _inference(**fields: Any) -> InferenceSpecStrict:
    return InferenceSpecStrict.model_validate({"taskType": "inference", **fields})


@pytest.mark.parametrize(
    ("spec", "key"),
    [
        (_inference(model=_MODEL), ExecutorKey.VLLM),
        (_inference(), ExecutorKey.VLLM),
        (_inference(model={**_MODEL, "vllm": {}}), ExecutorKey.VLLM),
        (
            _inference(model={**_MODEL, "transformers": {"dtype": "auto"}}),
            ExecutorKey.DEFAULT,
        ),
        (_inference(model=_MODEL, enforce_cpu=True), ExecutorKey.DEFAULT),
        (
            _inference(model={**_MODEL, "adapters": [{"type": "lora"}]}),
            ExecutorKey.VLLM_LORA,
        ),
    ],
)
def test_an_inference_spec_resolves_to_its_backend_executor(
    spec: InferenceSpecStrict, key: ExecutorKey
) -> None:
    assert resolve_executor_key(spec) is key


def test_an_unresolved_template_field_leaves_the_executor_undecided() -> None:
    spec = InferenceSpecTemplate.model_validate(
        {"taskType": "inference", "model": _MODEL, "enforce_cpu": "${cpu_only}"}
    )

    assert resolve_executor_key(spec) is None


def test_an_embedding_spec_resolves_by_its_vllm_config() -> None:
    vllm = EmbeddingSpecStrict.model_validate(
        {"taskType": "embedding", "model": {**_MODEL, "vllm": {}}}
    )
    plain = EmbeddingSpecStrict.model_validate(
        {"taskType": "embedding", "model": _MODEL}
    )

    assert resolve_executor_key(vllm) is ExecutorKey.VLLM_EMBEDDING
    assert resolve_executor_key(plain) is ExecutorKey.DEFAULT


def test_every_executor_key_names_a_registered_executor() -> None:
    assert set(EXECUTOR_MODULES) == set(ExecutorKey)


def test_every_task_type_resolves_to_a_registered_executor() -> None:
    for task_type in TaskType:
        if task_type in {TaskType.INFERENCE, TaskType.EMBEDDING}:
            continue
        key = resolve_executor_key(cast(Any, SimpleNamespace(taskType=task_type)))
        assert key in EXECUTOR_MODULES
