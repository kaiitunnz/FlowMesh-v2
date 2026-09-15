"""Tests for reading an inference leaf's generated texts out of its own result."""

import json

import pytest

from shared.harness import HarnessResult, HarnessResultKind
from shared.inference import (
    CanonicalInferenceRequest,
    canonical_request,
    canonical_result,
)
from shared.schemas.result.catalog import EmbeddingResult, InferenceResult
from shared.schemas.result.payloads import InferenceItem
from shared.tasks.specs import InferenceSpecStrict
from worker.executors.episode_support import EpisodeStepResult
from worker.executors.inference_projection import generated_outputs


def _spec(**fields: object) -> InferenceSpecStrict:
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


def _step(value: str | None) -> EpisodeStepResult:
    return EpisodeStepResult(
        harness_result=HarnessResult(kind=HarnessResultKind.COMPLETION), value=value
    )


class TestGeneratedOutputs:
    def test_a_local_generation_reports_its_items(self) -> None:
        request = canonical_request(_batch_spec())
        result = InferenceResult(
            items=[
                InferenceItem(index=0, prompt="hello", output="a"),
                InferenceItem(index=1, prompt="goodbye", output="b"),
                InferenceItem(index=2, prompt="again", output="c"),
            ]
        )
        assert generated_outputs(result, request) == ["a", "b", "c"]

    def test_a_relayed_batch_reports_one_completion_per_prompt(self) -> None:
        request = canonical_request(_batch_spec())
        assert generated_outputs(_step(json.dumps(["a", "b", "c"])), request) == [
            "a",
            "b",
            "c",
        ]

    def test_a_single_completion_is_read_verbatim(self) -> None:
        # A one-prompt contract never parses its completion, so a model that happens to
        # generate a JSON array is reported as the text it generated.
        request = canonical_request(_spec())
        assert generated_outputs(_step('["a", "b"]'), request) == ['["a", "b"]']

    def test_projecting_a_projected_result_reproduces_it(self) -> None:
        request = canonical_request(_batch_spec())
        once = canonical_result(request, ["a", "b", "c"])
        reread = generated_outputs(once, request)
        assert reread is not None
        assert canonical_result(request, reread) == once

    @pytest.mark.parametrize(
        "value",
        ["not json", json.dumps(["a", "b"]), json.dumps(["a", "b", 3])],
    )
    def test_a_batch_value_that_does_not_match_the_contract_is_not_read(
        self, value: str
    ) -> None:
        assert generated_outputs(_step(value), canonical_request(_batch_spec())) is None

    def test_a_step_that_generated_nothing_reports_nothing(self) -> None:
        assert generated_outputs(_step(None), canonical_request(_spec())) is None

    def test_an_item_count_that_does_not_match_the_contract_is_not_read(self) -> None:
        request = CanonicalInferenceRequest(
            model="Qwen/Qwen3-4B", prompts=("hello", "goodbye")
        )
        result = InferenceResult(
            items=[InferenceItem(index=0, prompt="hello", output="a")]
        )
        assert generated_outputs(result, request) is None

    def test_a_structured_item_output_is_not_read_as_generated_text(self) -> None:
        # An item's output is polymorphic: a template schema makes it a JSON value, and
        # only generated text projects into the declared contract.
        request = CanonicalInferenceRequest(model="m", prompts=("hello",))
        result = InferenceResult(
            items=[InferenceItem(index=0, prompt="hello", output={"verdict": "yes"})]
        )
        assert generated_outputs(result, request) is None

    def test_another_result_shape_declares_nothing(self) -> None:
        # Only the two shapes the embodiments produce are read; anything else stores
        # what it already reported rather than being reshaped or failing.
        request = CanonicalInferenceRequest(model="m", prompts=("hello",))
        assert generated_outputs(EmbeddingResult(), request) is None
