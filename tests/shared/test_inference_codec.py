"""Tests for the canonical inference request and result projection."""

from typing import Any

import pytest

from shared.inference import (
    CanonicalProjectionError,
    canonical_request,
    canonical_result,
)
from shared.tasks.specs import InferenceSpecStrict


def _spec(**fields: Any) -> InferenceSpecStrict:
    return InferenceSpecStrict.model_validate(
        {
            "taskType": "inference",
            "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
            "data": {"type": "list", "items": ["hello"]},
            **fields,
        }
    )


class TestCanonicalRequest:
    def test_a_single_literal_prompt_projects(self) -> None:
        request = canonical_request(_spec())
        assert request.model == "Qwen/Qwen3-4B"
        assert request.prompt == "hello"

    @pytest.mark.parametrize(
        "data, reason",
        [
            ({"type": "list", "items": ["a", "b"]}, "exactly one prompt"),
            ({"type": "list", "items": []}, "non-empty literal list"),
            ({"type": "list", "items": [{"role": "user"}]}, "one literal string"),
            ({"type": "dataset", "url": "squad"}, "not projectable"),
            ({"type": "list", "expr": "up.items"}, "non-empty literal list"),
            (
                {"type": "list", "items": ["a"], "s3_cfg": "s3://bucket"},
                "literal items",
            ),
        ],
    )
    def test_an_unshared_input_shape_is_rejected(
        self, data: dict[str, Any], reason: str
    ) -> None:
        with pytest.raises(CanonicalProjectionError, match=reason):
            canonical_request(_spec(data=data))

    def test_a_leaf_without_a_model_is_rejected(self) -> None:
        spec = InferenceSpecStrict.model_validate(
            {"taskType": "inference", "data": {"type": "list", "items": ["hi"]}}
        )
        with pytest.raises(CanonicalProjectionError, match="no model source"):
            canonical_request(spec)


class TestCanonicalResult:
    def test_a_completion_reports_the_declared_singleton_output(self) -> None:
        result = canonical_result(canonical_request(_spec()), "world")
        assert result.model == "Qwen/Qwen3-4B"
        assert len(result.items) == 1
        assert result.items[0].index == 0
        assert result.items[0].prompt == "hello"
        assert result.items[0].output == "world"

    def test_token_accounting_is_not_part_of_the_declared_output(self) -> None:
        # An embodiment that does not report token counts leaves usage unset rather
        # than changing what the leaf declares.
        assert canonical_result(canonical_request(_spec()), "world").usage is None
