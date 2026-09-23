"""Tests for the org-scoped task merge key."""

import pytest

from server.task.runtime import _compute_merge_key
from shared.tasks import TaskEnvelopeTemplate


class TestComputeMergeKey:
    def _make_task(
        self, task_type: str = "inference", **spec_kw
    ) -> TaskEnvelopeTemplate:
        spec_data = {"taskType": task_type, **spec_kw}
        return TaskEnvelopeTemplate.model_validate(
            {"apiVersion": "mloc/v1", "kind": "Task", "spec": spec_data}
        )

    def test_deterministic(self) -> None:
        t = self._make_task(
            "inference",
            model={"source": {"identifier": "llama"}},
            inference={"temperature": 0.7},
        )
        k1 = _compute_merge_key(t, "org")
        k2 = _compute_merge_key(t, "org")
        assert k1 is not None
        assert k1 == k2

    def test_different_specs_different_keys(self) -> None:
        t1 = self._make_task("inference", model={"source": {"identifier": "llama"}})
        t2 = self._make_task("inference", model={"source": {"identifier": "gpt-4"}})
        k1 = _compute_merge_key(t1, "org")
        k2 = _compute_merge_key(t2, "org")
        assert k1 != k2

    def test_different_scopes_different_keys(self) -> None:
        t = self._make_task("inference", model={"source": {"identifier": "llama"}})
        assert _compute_merge_key(t, "org-x") != _compute_merge_key(t, "org-y")

    def test_visual_embedding_inference_returns_none(self) -> None:
        t = self._make_task(
            "inference",
            model={
                "source": {"identifier": "llava"},
                "transformers": {"mode": "visual-embedding"},
            },
        )
        assert _compute_merge_key(t, "org") is None

    @pytest.mark.parametrize("task_type", ["echo", "rag", "diffusion"])
    def test_non_inference_type_returns_none(self, task_type: str) -> None:
        t = self._make_task(task_type)
        assert _compute_merge_key(t, "org") is None

    def test_ignores_data_field(self) -> None:
        """Two tasks with same model but different prompts should merge."""
        t1 = self._make_task(
            "inference",
            model={"source": {"identifier": "llama"}},
            data={"messages": [{"role": "user", "content": "hello"}]},
        )
        t2 = self._make_task(
            "inference",
            model={"source": {"identifier": "llama"}},
            data={"messages": [{"role": "user", "content": "goodbye"}]},
        )
        k1 = _compute_merge_key(t1, "org")
        k2 = _compute_merge_key(t2, "org")
        assert k1 is not None
        assert k1 == k2
