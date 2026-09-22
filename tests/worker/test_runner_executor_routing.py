"""The serve executor is registered under its key and services serve tasks alone."""

from shared.tasks.executor_key import ExecutorKey
from shared.tasks.task_type import TaskType
from worker.executors import EXECUTOR_REGISTRY
from worker.executors.vllm_serve_executor import VLLMServeExecutor


class TestServeRoutingViaRegistry:
    def test_vllm_serve_key_is_registered(self) -> None:
        assert "vllm_serve" in EXECUTOR_REGISTRY

    def test_vllm_serve_executor_handles_serve_task_type(self) -> None:
        cls = EXECUTOR_REGISTRY.get(ExecutorKey.VLLM_SERVE)
        assert cls is not None
        assert TaskType.SERVE in cls.supported_task_types

    def test_vllm_serve_executor_only_handles_serve(self) -> None:
        assert VLLMServeExecutor.supported_task_types == frozenset({TaskType.SERVE})
