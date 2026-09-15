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


def _batch_spec() -> InferenceSpecStrict:
    return _spec(data={"type": "list", "items": ["hello", "goodbye", "again"]})


class TestCanonicalRequest:
    def test_a_single_literal_prompt_projects(self) -> None:
        request = canonical_request(_spec())
        assert request.model == "Qwen/Qwen3-4B"
        assert request.prompts == ("hello",)

    def test_several_literal_prompts_project_in_declared_order(self) -> None:
        assert canonical_request(_batch_spec()).prompts == (
            "hello",
            "goodbye",
            "again",
        )

    def test_each_prompt_becomes_its_own_conversation(self) -> None:
        bodies = canonical_request(_batch_spec()).chat_bodies()
        assert [body["messages"] for body in bodies] == [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "goodbye"}],
            [{"role": "user", "content": "again"}],
        ]

    def test_every_prompt_issues_the_same_sampling_request(self) -> None:
        bodies = canonical_request(_batch_spec()).chat_bodies()
        sampling = [
            {key: value for key, value in body.items() if key != "messages"}
            for body in bodies
        ]
        assert sampling[0] == sampling[1] == sampling[2]
        assert sampling[0]["max_tokens"] == 512

    @pytest.mark.parametrize(
        "data, reason",
        [
            ({"type": "list", "items": []}, "non-empty literal list"),
            ({"type": "list", "items": [{"role": "user"}]}, "literal non-empty"),
            ({"type": "list", "items": ["a", ""]}, "literal non-empty"),
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
        result = canonical_result(canonical_request(_spec()), ["world"])
        assert result.model == "Qwen/Qwen3-4B"
        assert len(result.items) == 1
        assert result.items[0].index == 0
        assert result.items[0].prompt == "hello"
        assert result.items[0].output == "world"

    def test_a_batch_reports_one_item_per_declared_prompt(self) -> None:
        result = canonical_result(canonical_request(_batch_spec()), ["a", "b", "c"])
        assert [(item.index, item.prompt, item.output) for item in result.items] == [
            (0, "hello", "a"),
            (1, "goodbye", "b"),
            (2, "again", "c"),
        ]

    def test_a_run_reporting_the_wrong_count_is_rejected(self) -> None:
        # Truncating to the shorter side would silently drop a declared prompt's output.
        with pytest.raises(CanonicalProjectionError, match="3 prompts"):
            canonical_result(canonical_request(_batch_spec()), ["a", "b"])

    def test_token_accounting_is_not_part_of_the_declared_output(self) -> None:
        # An embodiment that does not report token counts leaves usage unset rather
        # than changing what the leaf declares.
        assert canonical_result(canonical_request(_spec()), ["world"]).usage is None
