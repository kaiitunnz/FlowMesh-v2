"""Tests for the shared upstream-expression walker."""

from typing import Any

import pandas as pd
import pytest

from shared.schemas.result import BaseExecutorResult
from shared.schemas.result.catalog import InferenceResult
from shared.schemas.result.payloads import InferenceItem
from worker.executors.base_executor import ExecutionError
from worker.executors.utils.expressions import project_expression, split_indexes


def _context(**results: BaseExecutorResult) -> dict[str, BaseExecutorResult]:
    return dict(results)


def _result(**fields: Any) -> BaseExecutorResult:
    return BaseExecutorResult.model_validate(fields)


def _inference(*outputs: Any) -> InferenceResult:
    return InferenceResult(
        model="Qwen/Qwen3-4B",
        items=[
            InferenceItem(index=index, prompt=f"p{index}", output=output)
            for index, output in enumerate(outputs)
        ],
    )


class TestStructuredProjection:
    def test_a_model_attribute_resolves(self) -> None:
        assert project_expression("up.model", _context(up=_inference("a"))) == (
            "Qwen/Qwen3-4B"
        )

    def test_an_indexed_model_element_resolves(self) -> None:
        context = _context(up=_inference("a", "b"))
        assert project_expression("up.items[1].output", context) == "b"

    def test_a_list_of_models_plucks_one_attribute_per_element(self) -> None:
        # The list generalization of reading an attribute off one model, so a leaf can
        # project an upstream result's items into one ordered vector.
        context = _context(up=_inference("a", "b", "c"))
        assert project_expression("up.items.output", context) == ["a", "b", "c"]

    def test_a_list_of_models_missing_the_attribute_is_rejected(self) -> None:
        context = _context(up=_inference("a", "b"))
        with pytest.raises(ExecutionError, match="not a valid attribute"):
            project_expression("up.items.nonexistent", context)

    def test_a_list_of_dicts_plucks_one_key_per_element(self) -> None:
        context = _context(up=_result(rows=[{"t": "a"}, {"t": "b"}]))
        assert project_expression("up.rows.t", context) == ["a", "b"]

    def test_an_empty_list_resolves_to_an_empty_list(self) -> None:
        context = _context(up=_result(rows=[]))
        assert project_expression("up.rows.t", context) == []

    def test_an_unknown_root_resolves_to_nothing(self) -> None:
        assert project_expression("missing.items", _context()) is None

    def test_an_unindexable_value_is_rejected(self) -> None:
        context = _context(up=_result(count=3))
        with pytest.raises(ExecutionError, match="not a valid key"):
            project_expression("up.count.t", context)


class TestFrameProjection:
    def test_a_frame_column_resolves_when_frames_are_admitted(self) -> None:
        frame = pd.DataFrame({"t": ["a", "b"]})
        context = _context(up=_result(frame=frame))
        assert project_expression("up.frame.t", context) == ["a", "b"]

    def test_a_frame_is_unprojectable_to_a_consumer_that_excludes_frames(self) -> None:
        frame = pd.DataFrame({"t": ["a", "b"]})
        context = _context(up=_result(frame=frame))
        with pytest.raises(ExecutionError, match="not a valid key"):
            project_expression("up.frame.t", context, frames=False)

    def test_a_missing_frame_column_is_rejected(self) -> None:
        frame = pd.DataFrame({"t": ["a"]})
        context = _context(up=_result(frame=frame))
        with pytest.raises(ExecutionError, match="not a valid column"):
            project_expression("up.frame.other", context)


class TestSplitIndexes:
    @pytest.mark.parametrize(
        "token, expected",
        [
            ("items", ("items", [])),
            ("items[0]", ("items", [0])),
            ("items[0][2]", ("items", [0, 2])),
            ("items[-1]", ("items", [-1])),
            ("items[bad]", ("items", [-1])),
        ],
    )
    def test_a_token_splits_into_its_attribute_and_indexes(
        self, token: str, expected: tuple[str, list[int]]
    ) -> None:
        assert split_indexes(token) == expected
