"""Tests for the inference spec: backend classification and dispatch validation."""

from typing import Any

import pytest

from shared.tasks.specs import (
    EmbeddingSpecStrict,
    InferenceBackend,
    InferenceSpecStrict,
)


def _spec(**fields: Any) -> InferenceSpecStrict:
    return InferenceSpecStrict.model_validate({"taskType": "inference", **fields})


class TestInferenceBackend:
    def test_enforce_cpu_is_transformers(self) -> None:
        assert _spec(enforce_cpu=True).backend() is InferenceBackend.TRANSFORMERS

    def test_transformers_only_is_transformers(self) -> None:
        spec = _spec(model={"transformers": {"torch_dtype": "float32"}})
        assert spec.backend() is InferenceBackend.TRANSFORMERS

    def test_explicit_vllm_is_vllm(self) -> None:
        spec = _spec(model={"vllm": {"gpu_memory_utilization": 0.9}})
        assert spec.backend() is InferenceBackend.VLLM

    def test_transformers_and_vllm_is_vllm(self) -> None:
        spec = _spec(
            model={
                "transformers": {"torch_dtype": "float32"},
                "vllm": {"gpu_memory_utilization": 0.9},
            }
        )
        assert spec.backend() is InferenceBackend.VLLM

    def test_empty_vllm_config_is_not_vllm(self) -> None:
        # An empty vLLM dict is treated as unconfigured (matching the runner): with
        # a transformers model it stays on transformers, otherwise it is AUTO.
        with_transformers = _spec(model={"transformers": {"x": 1}, "vllm": {}})
        assert with_transformers.backend() is InferenceBackend.TRANSFORMERS
        assert _spec(model={"vllm": {}}).backend() is InferenceBackend.AUTO

    def test_lora_adapter_is_vllm(self) -> None:
        spec = _spec(model={"adapters": [{"type": "lora"}]})
        assert spec.backend() is InferenceBackend.VLLM

    def test_non_lora_adapter_is_vllm(self) -> None:
        spec = _spec(model={"adapters": [{"type": "ia3"}]})
        assert spec.backend() is InferenceBackend.VLLM

    def test_no_model_is_auto(self) -> None:
        assert _spec().backend() is InferenceBackend.AUTO

    def test_unhinted_model_is_auto(self) -> None:
        spec = _spec(model={"source": {"identifier": "gpt2"}})
        assert spec.backend() is InferenceBackend.AUTO


class TestValidateDispatchable:
    """Exercises the resolved-spec path used at dispatch (no placeholder deferral)."""

    def test_vllm_without_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="requests no GPU"):
            _spec(
                model={"vllm": {"gpu_memory_utilization": 0.9}}
            ).validate_dispatchable()

    def test_vllm_with_gpu_ok(self) -> None:
        _spec(
            model={"vllm": {"gpu_memory_utilization": 0.9}},
            resources={"hardware": {"gpu": {"count": 1}}},
        ).validate_dispatchable()

    def test_enforce_cpu_with_vllm_raises(self) -> None:
        with pytest.raises(ValueError, match="enforce_cpu is set but the model"):
            _spec(enforce_cpu=True, model={"vllm": {"x": 1}}).validate_dispatchable()

    def test_adapter_without_source_raises(self) -> None:
        with pytest.raises(ValueError, match="no path, url, or task_id"):
            _spec(model={"adapters": [{"type": "lora"}]}).validate_dispatchable()

    def test_adapter_with_task_id_ok(self) -> None:
        _spec(
            model={"adapters": [{"type": "lora", "task_id": "tsk-abc"}]},
            resources={"hardware": {"gpu": {"count": 1}}},
        ).validate_dispatchable()

    def test_unhinted_auto_ok(self) -> None:
        _spec(model={"source": {"identifier": "gpt2"}}).validate_dispatchable()

    def test_resident_adapter_without_source_raises(self) -> None:
        with pytest.raises(ValueError, match="no path, url, or task_id"):
            _spec(
                model={"adapters": [{"type": "lora"}]},
                service={"mode": "resident"},
            ).validate_dispatchable()

    def test_resident_multiple_adapters_raise(self) -> None:
        with pytest.raises(ValueError, match="single adapter"):
            _spec(
                model={
                    "adapters": [
                        {"type": "lora", "name": "a", "path": "hf/a"},
                        {"type": "lora", "name": "b", "path": "hf/b"},
                    ]
                },
                service={"mode": "resident"},
            ).validate_dispatchable()

    def test_resident_single_adapter_with_source_ok(self) -> None:
        _spec(
            model={"adapters": [{"type": "lora", "name": "a", "path": "hf/a"}]},
            service={"mode": "resident"},
        ).validate_dispatchable()


class TestLocalEligibleBinding:
    def test_primary_is_optional(self) -> None:
        binding = _spec(service={"mode": "local_eligible"}).service
        assert binding is not None and binding.primary is None

    def test_undeclared_mode_leaves_the_leaf_to_decide(self) -> None:
        binding = _spec(service={"isolation": "tenant-a"}).service
        assert binding is not None and binding.mode is None

    def test_primary_rejected_on_a_resident_binding(self) -> None:
        with pytest.raises(ValueError, match="a resident binding admits one"):
            _spec(service={"mode": "resident", "primary": "self_contained"})

    def test_local_eligible_keeps_the_local_gpu_requirement(self) -> None:
        # A resident binding admits to a replica and carries no worker-local GPU
        # requirement; a local-eligible one must still place its self-contained
        # embodiment, so the vLLM backend's GPU requirement survives.
        _spec(
            model={"vllm": {"gpu_memory_utilization": 0.9}},
            service={"mode": "resident"},
        ).validate_dispatchable()
        with pytest.raises(ValueError, match="requests no GPU"):
            _spec(
                model={"vllm": {"gpu_memory_utilization": 0.9}},
                service={"mode": "local_eligible", "primary": "resident_served"},
            ).validate_dispatchable()

    def test_local_eligible_with_a_declared_gpu_is_dispatchable(self) -> None:
        _spec(
            model={"vllm": {"gpu_memory_utilization": 0.9}},
            resources={"hardware": {"gpu": {"count": 1}}},
            service={"mode": "local_eligible", "primary": "resident_served"},
        ).validate_dispatchable()

    def test_local_eligible_keeps_the_resident_adapter_limit(self) -> None:
        with pytest.raises(ValueError, match="single adapter"):
            _spec(
                model={
                    "adapters": [
                        {"type": "lora", "name": "a", "path": "hf/a"},
                        {"type": "lora", "name": "b", "path": "hf/b"},
                    ]
                },
                resources={"hardware": {"gpu": {"count": 1}}},
                service={"mode": "local_eligible", "primary": "resident_served"},
            ).validate_dispatchable()

    def test_embedding_rejects_local_eligible(self) -> None:
        spec = EmbeddingSpecStrict.model_validate(
            {
                "taskType": "embedding",
                "service": {"mode": "local_eligible", "primary": "self_contained"},
            }
        )
        with pytest.raises(ValueError, match="only a resident service binding"):
            spec.validate_dispatchable()


class TestMergeKey:
    _MODEL = {"source": {"identifier": "m"}}

    def test_inputs_do_not_change_the_key(self) -> None:
        one = _spec(
            model=self._MODEL,
            data={"type": "list", "items": ["a"]},
            inference={"system_prompt": "x", "temperature": 0.1},
        )
        two = _spec(
            model=self._MODEL,
            data={"type": "list", "items": ["b"]},
            inference={"system_prompt": "y", "temperature": 0.1},
            _upstreamResults={"up": {"value": 1}},
        )
        assert one.merge_key() is not None
        assert one.merge_key() == two.merge_key()

    @pytest.mark.parametrize(
        "fields",
        [
            {"model": {"source": {"identifier": "other"}}},
            {"model": _MODEL, "inference": {"temperature": 0.9}},
            {
                "model": {
                    **_MODEL,
                    "adapters": [{"type": "lora", "path": "/other"}],
                }
            },
        ],
    )
    def test_the_model_sampling_or_adapters_change_the_key(
        self, fields: dict[str, Any]
    ) -> None:
        base = _spec(model=self._MODEL, inference={"temperature": 0.1})
        assert _spec(**fields).merge_key() != base.merge_key()

    def test_a_visual_embedding_task_never_merges(self) -> None:
        spec = _spec(
            model={**self._MODEL, "transformers": {"mode": "visual-embedding"}}
        )
        assert spec.merge_key() is None

    def test_another_task_type_never_merges(self) -> None:
        spec = EmbeddingSpecStrict.model_validate(
            {"taskType": "embedding", "model": self._MODEL}
        )
        assert spec.merge_key() is None
