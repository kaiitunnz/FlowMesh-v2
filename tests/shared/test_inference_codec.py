"""Tests for the canonical inference request and result projection."""

import json
from typing import Any

import pytest

from shared.inference import (
    CanonicalInferenceRequest,
    CanonicalProjectionError,
    canonical_request,
    canonical_result,
    generated_outputs,
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


class TestGeneratedOutputs:
    def test_a_local_generation_reports_its_items(self) -> None:
        request = canonical_request(_batch_spec())
        payload = {
            "items": [
                {"index": 0, "prompt": "hello", "output": "a"},
                {"index": 1, "prompt": "goodbye", "output": "b"},
                {"index": 2, "prompt": "again", "output": "c"},
            ]
        }
        assert generated_outputs(payload, request) == ["a", "b", "c"]

    def test_a_relayed_batch_reports_one_completion_per_prompt(self) -> None:
        request = canonical_request(_batch_spec())
        payload = {"value": json.dumps(["a", "b", "c"])}
        assert generated_outputs(payload, request) == ["a", "b", "c"]

    def test_a_single_completion_is_read_verbatim(self) -> None:
        # A one-prompt contract never parses its completion, so a model that happens to
        # generate a JSON array is reported as the text it generated.
        request = canonical_request(_spec())
        assert generated_outputs({"value": '["a", "b"]'}, request) == ['["a", "b"]']

    def test_projecting_a_projected_result_reproduces_it(self) -> None:
        request = canonical_request(_batch_spec())
        once = canonical_result(request, ["a", "b", "c"])
        reread = generated_outputs(once.model_dump(), request)
        assert reread is not None
        assert canonical_result(request, reread) == once

    @pytest.mark.parametrize(
        "value",
        ["not json", json.dumps(["a", "b"]), json.dumps(["a", "b", 3])],
    )
    def test_a_batch_value_that_does_not_match_the_contract_is_not_read(
        self, value: str
    ) -> None:
        assert (
            generated_outputs({"value": value}, canonical_request(_batch_spec()))
            is None
        )

    def test_a_step_that_generated_nothing_reports_nothing(self) -> None:
        assert generated_outputs({}, canonical_request(_spec())) is None

    def test_an_item_count_that_does_not_match_the_contract_is_not_read(self) -> None:
        request = CanonicalInferenceRequest(
            model="Qwen/Qwen3-4B", prompts=("hello", "goodbye")
        )
        payload = {"items": [{"index": 0, "prompt": "hello", "output": "a"}]}
        assert generated_outputs(payload, request) is None
