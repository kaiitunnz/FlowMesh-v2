"""Tests for the canonical inference request and result projection."""

from typing import Any

import pytest

from shared.inference import (
    CanonicalProjectionError,
    InferenceSourceKind,
    InputResolutionError,
    canonical_contract,
    canonical_result,
    canonical_source,
    resolve_contract,
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


def _request(spec: InferenceSpecStrict):
    """The request a spec's contract resolves to, through the path production runs."""
    return resolve_contract(canonical_contract(spec), None).request


def _batch_spec() -> InferenceSpecStrict:
    return _spec(data={"type": "list", "items": ["hello", "goodbye", "again"]})


class TestCanonicalRequest:
    def test_a_single_literal_prompt_projects(self) -> None:
        request = _request(_spec())
        assert request.model == "Qwen/Qwen3-4B"
        assert request.prompts == ("hello",)

    def test_several_literal_prompts_project_in_declared_order(self) -> None:
        assert _request(_batch_spec()).prompts == (
            "hello",
            "goodbye",
            "again",
        )

    def test_each_prompt_becomes_its_own_conversation(self) -> None:
        bodies = _request(_batch_spec()).chat_bodies()
        assert [body["messages"] for body in bodies] == [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "goodbye"}],
            [{"role": "user", "content": "again"}],
        ]

    def test_every_prompt_issues_the_same_sampling_request(self) -> None:
        bodies = _request(_batch_spec()).chat_bodies()
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
            ({"type": "list", "expr": "up.items"}, "must declare spec.data.max_items"),
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
            canonical_contract(_spec(data=data))

    def test_a_leaf_without_a_model_is_rejected(self) -> None:
        spec = InferenceSpecStrict.model_validate(
            {"taskType": "inference", "data": {"type": "list", "items": ["hi"]}}
        )
        with pytest.raises(CanonicalProjectionError, match="no model source"):
            canonical_contract(spec)


class TestCanonicalResult:
    def test_a_completion_reports_the_declared_singleton_output(self) -> None:
        result = canonical_result(_request(_spec()), ["world"])
        assert result.model == "Qwen/Qwen3-4B"
        assert len(result.items) == 1
        assert result.items[0].index == 0
        assert result.items[0].prompt == "hello"
        assert result.items[0].output == "world"

    def test_a_batch_reports_one_item_per_declared_prompt(self) -> None:
        result = canonical_result(_request(_batch_spec()), ["a", "b", "c"])
        assert [(item.index, item.prompt, item.output) for item in result.items] == [
            (0, "hello", "a"),
            (1, "goodbye", "b"),
            (2, "again", "c"),
        ]

    def test_a_run_reporting_the_wrong_count_is_rejected(self) -> None:
        # Truncating to the shorter side would silently drop a declared prompt's output.
        with pytest.raises(CanonicalProjectionError, match="3 prompts"):
            canonical_result(_request(_batch_spec()), ["a", "b"])

    def test_token_accounting_is_not_part_of_the_declared_output(self) -> None:
        # An embodiment that does not report token counts leaves usage unset rather
        # than changing what the leaf declares.
        assert canonical_result(_request(_spec()), ["world"]).usage is None


class TestCanonicalSource:
    def test_literal_items_are_a_source_bounded_by_their_own_count(self) -> None:
        source = canonical_source(_batch_spec())
        assert source.kind is InferenceSourceKind.LITERAL
        assert source.items == ("hello", "goodbye", "again")
        assert source.max_items == 3

    @pytest.mark.parametrize(
        "data",
        [
            {"type": "list", "expr": "up.items.output", "max_items": 8},
            {"type": "list", "node": "up", "path": "items.output", "max_items": 8},
        ],
    )
    def test_both_upstream_forms_normalize_to_one_descriptor(
        self, data: dict[str, Any]
    ) -> None:
        source = canonical_source(_spec(data=data))
        assert source.kind is InferenceSourceKind.UPSTREAM
        assert (source.node, source.path) == ("up", "items.output")
        assert source.expression == "up.items.output"

    def test_the_two_upstream_forms_digest_identically(self) -> None:
        by_expr = canonical_source(
            _spec(data={"type": "list", "expr": "up.items.output", "max_items": 8})
        )
        by_node = canonical_source(
            _spec(
                data={
                    "type": "list",
                    "node": "up",
                    "path": "items.output",
                    "max_items": 8,
                }
            )
        )
        assert by_expr.digest() == by_node.digest()

    @pytest.mark.parametrize(
        "data, reason",
        [
            ({"type": "list", "expr": "up", "max_items": 2}, "as '<node>.<path>'"),
            ({"type": "list", "node": "up", "max_items": 2}, "spec.data.path"),
            (
                {"type": "list", "items": ["a"], "expr": "up.x", "max_items": 1},
                "exactly one source",
            ),
            (
                {"type": "list", "expr": "up.x", "node": "up", "max_items": 1},
                "declare one projection",
            ),
            ({"type": "list", "expr": "up.x", "max_items": 0}, "positive integer"),
        ],
    )
    def test_an_unprojectable_source_is_rejected(
        self, data: dict[str, Any], reason: str
    ) -> None:
        with pytest.raises(CanonicalProjectionError, match=reason):
            canonical_source(_spec(data=data))


def _upstream_contract(max_items: int = 4, **data: Any):
    return canonical_contract(
        _spec(
            data={"type": "list", "expr": "up.items.output", **data}
            | {"max_items": max_items}
        )
    )


class TestResolveContract:
    def test_a_literal_contract_resolves_to_its_own_items(self) -> None:
        resolved = resolve_contract(canonical_contract(_batch_spec()), None)
        assert resolved.request.prompts == ("hello", "goodbye", "again")
        assert resolved.binding.cardinality == 3
        assert resolved.binding.upstream == ()

    def test_an_upstream_contract_resolves_to_the_projected_vector(self) -> None:
        resolved = resolve_contract(_upstream_contract(), ["a", "b"])
        assert resolved.request.prompts == ("a", "b")
        assert resolved.binding.cardinality == 2
        assert resolved.binding.upstream == ()

    def test_both_kinds_reach_the_same_request_from_the_same_values(self) -> None:
        # One resolution seam, so a literal leaf and an upstream one that yields the
        # same vector are the same request to everything downstream.
        literal = resolve_contract(canonical_contract(_batch_spec()), None)
        upstream = resolve_contract(_upstream_contract(), ["hello", "goodbye", "again"])
        assert literal.request == upstream.request
        assert literal.binding.request_digest == upstream.binding.request_digest

    def test_the_binding_projects_the_whole_vectors_token_demand(self) -> None:
        resolved = resolve_contract(_upstream_contract(), ["a", "b"])
        assert resolved.binding.projected_output_tokens == 1024

    @pytest.mark.parametrize(
        "projected, reason",
        [
            (None, "resolved nothing"),
            ("a", "resolved str"),
            ([], "resolved list"),
            (["a", 2], "non-string or empty"),
            (["a", " "], "non-string or empty"),
            (["a"] * 5, "declares at most 4"),
        ],
    )
    def test_a_source_outside_its_declared_shape_fails_typed(
        self, projected: Any, reason: str
    ) -> None:
        with pytest.raises(InputResolutionError, match=reason):
            resolve_contract(_upstream_contract(), projected)

    def test_a_long_prompt_resolves(self) -> None:
        # A prompt's size is the engine's context length to answer, not the source
        # envelope's: the envelope bounds how many conversations a resolution admits.
        long_prompt = "x" * 200_000
        resolved = resolve_contract(_upstream_contract(), [long_prompt])
        assert resolved.request.prompts == (long_prompt,)
