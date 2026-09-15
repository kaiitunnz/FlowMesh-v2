"""Tests for materializing an inference contract on the origin worker."""

from typing import Any

import pandas as pd
import pytest

from shared.inference import (
    CanonicalInferenceContract,
    InputResolutionError,
    canonical_contract,
)
from shared.schemas.result import BaseExecutorResult
from shared.schemas.result.catalog import InferenceResult
from shared.schemas.result.payloads import InferenceItem
from shared.tasks.specs import InferenceSpecStrict
from tests.worker.factories import make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.inference.resolution import resolve_task_contract
from worker.runner import Runner


def _spec(data: dict[str, Any], upstream: dict[str, Any] | None = None):
    body: dict[str, Any] = {
        "taskType": "inference",
        "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
        "data": data,
    }
    if upstream is not None:
        body["_upstreamResults"] = upstream
    return InferenceSpecStrict.model_validate(body)


def _upstream(*outputs: Any) -> InferenceResult:
    return InferenceResult(
        model="up/model",
        items=[
            InferenceItem(index=index, prompt=f"p{index}", output=output)
            for index, output in enumerate(outputs)
        ],
    )


def _task(data: dict[str, Any], upstream: dict[str, Any] | None = None):
    spec = _spec(data, upstream)
    return make_worker_task_message(
        spec.model_dump(by_alias=True),
        declared_contract=canonical_contract(spec).model_dump_json(),
    )


_EXPR = {"type": "list", "expr": "up.items.output", "max_items": 8}
_NODE_PATH = {"type": "list", "node": "up", "path": "items.output", "max_items": 8}


class TestLiteralResolution:
    def test_a_literal_contract_resolves_without_an_upstream_input(self) -> None:
        resolved = resolve_task_contract(
            _task({"type": "list", "items": ["hello", "bye"]})
        )
        assert resolved is not None
        assert resolved.request.prompts == ("hello", "bye")
        assert resolved.binding.upstream == ()

    def test_a_task_carrying_no_contract_resolves_to_nothing(self) -> None:
        task = _task({"type": "list", "items": ["hello"]})
        task.declared_contract = None
        assert resolve_task_contract(task) is None


class TestUpstreamResolution:
    @pytest.mark.parametrize("data", [_EXPR, _NODE_PATH])
    def test_both_declared_forms_resolve_the_same_request(
        self, data: dict[str, Any]
    ) -> None:
        resolved = resolve_task_contract(_task(data, {"up": _upstream("a", "b")}))
        assert resolved is not None
        assert resolved.request.prompts == ("a", "b")

    def test_the_two_forms_reach_an_identical_request_and_digest(self) -> None:
        # One source contract, one resolution seam: how the leaf wrote its projection
        # cannot change the request either embodiment runs.
        upstream = {"up": _upstream("a", "b")}
        by_expr = resolve_task_contract(_task(_EXPR, upstream))
        by_node = resolve_task_contract(_task(_NODE_PATH, upstream))
        assert by_expr is not None and by_node is not None
        assert by_expr.request == by_node.request
        assert by_expr.binding.request_digest == by_node.binding.request_digest

    def test_declared_order_and_cardinality_are_preserved(self) -> None:
        resolved = resolve_task_contract(
            _task(_EXPR, {"up": _upstream("first", "second", "third")})
        )
        assert resolved is not None
        assert resolved.request.prompts == ("first", "second", "third")
        assert resolved.binding.cardinality == 3

    def test_the_binding_records_which_upstream_content_it_read(self) -> None:
        resolved = resolve_task_contract(_task(_EXPR, {"up": _upstream("a")}))
        changed = resolve_task_contract(_task(_EXPR, {"up": _upstream("b")}))
        assert resolved is not None and changed is not None
        assert resolved.binding.upstream[0].node == "up"
        # A re-drive reading different upstream content carries different provenance,
        # which is what lets a fresh retry refuse substituted input.
        assert not resolved.binding.matches(changed.binding)

    def test_an_index_narrowing_to_one_value_is_not_a_prompt_vector(self) -> None:
        # Indexing is admitted, but a leaf's prompts are a list; a projection that
        # narrows to a single value fails rather than becoming a one-prompt request.
        data = {"type": "list", "expr": "up.items[0].output", "max_items": 8}
        with pytest.raises(InputResolutionError, match="resolved str"):
            resolve_task_contract(_task(data, {"up": _upstream("a", "b")}))


class TestTypedInputFailures:
    def test_an_undeclared_upstream_input_fails(self) -> None:
        with pytest.raises(InputResolutionError, match="does not declare a dependency"):
            resolve_task_contract(_task(_EXPR, {"other": _upstream("a")}))

    def test_a_path_that_does_not_project_fails(self) -> None:
        data = {"type": "list", "expr": "up.items.missing", "max_items": 8}
        with pytest.raises(InputResolutionError, match="does not project"):
            resolve_task_contract(_task(data, {"up": _upstream("a")}))

    def test_a_non_string_element_fails(self) -> None:
        with pytest.raises(InputResolutionError, match="non-string or empty"):
            resolve_task_contract(_task(_EXPR, {"up": _upstream("a", 7)}))

    def test_a_non_list_projection_fails(self) -> None:
        data = {"type": "list", "expr": "up.model", "max_items": 8}
        with pytest.raises(InputResolutionError, match="resolved str"):
            resolve_task_contract(_task(data, {"up": _upstream("a")}))

    def test_an_over_bound_projection_fails(self) -> None:
        data = {"type": "list", "expr": "up.items.output", "max_items": 2}
        with pytest.raises(InputResolutionError, match="declares at most 2"):
            resolve_task_contract(_task(data, {"up": _upstream("a", "b", "c")}))

    def test_a_table_upstream_is_an_unprojectable_input(self) -> None:
        # A table projects under its own canonical contract, so it fails here rather
        # than resolving into a request only one embodiment would read the same way.
        upstream = BaseExecutorResult.model_validate(
            {"frame": pd.DataFrame({"t": ["a", "b"]})}
        )
        data = {"type": "list", "expr": "up.frame.t", "max_items": 8}
        with pytest.raises(InputResolutionError, match="does not project"):
            resolve_task_contract(_task(data, {"up": upstream}))


def test_a_resolved_request_carries_no_source_syntax() -> None:
    # What crosses to a replica is the request, not the projection that produced it.
    resolved = resolve_task_contract(_task(_EXPR, {"up": _upstream("a", "b")}))
    assert resolved is not None
    serialized = resolved.request.model_dump_json()
    assert "up.items.output" not in serialized
    assert "expr" not in serialized


def test_a_contract_declaring_an_unimplemented_resolver_fails() -> None:
    task = _task(_EXPR, {"up": _upstream("a")})
    contract = CanonicalInferenceContract.model_validate_json(task.declared_contract)
    bumped = contract.model_copy(
        update={"source": contract.source.model_copy(update={"resolver_version": "99"})}
    )
    task.declared_contract = bumped.model_dump_json()
    with pytest.raises(InputResolutionError, match="resolver version"):
        resolve_task_contract(task)


class TestRecoveryFence:
    def _runner(self):
        updates: list[tuple[str, dict[str, Any]]] = []

        class _Lifecycle:
            def notify_task_update(self, task_id: str, payload: dict[str, Any]) -> None:
                updates.append((task_id, payload))

        runner = Runner.__new__(Runner)
        runner.lifecycle = _Lifecycle()
        return runner, updates

    def test_a_resolution_is_recorded_before_the_embodiment_runs(self) -> None:
        runner, updates = self._runner()
        task = _task(_EXPR, {"up": _upstream("a", "b")})
        runner._materialize_contract(task)
        assert task.resolved_contract is not None
        assert updates and "input_resolution" in updates[0][1]
        assert updates[0][1]["input_resolution"]["cardinality"] == 2

    def test_a_re_drive_reaching_the_same_request_runs(self) -> None:
        runner, _updates = self._runner()
        first = _task(_EXPR, {"up": _upstream("a", "b")})
        runner._materialize_contract(first)
        again = _task(_EXPR, {"up": _upstream("a", "b")})
        resolved = resolve_task_contract(first)
        assert resolved is not None
        again.recorded_resolution = resolved.binding.model_dump_json()
        runner._materialize_contract(again)
        assert again.resolved_contract == first.resolved_contract

    def test_a_re_drive_on_substituted_upstream_content_fails_closed(self) -> None:
        # The committed invocation runs the inputs it resolved; a retry that would
        # carry different ones fails rather than silently running them.
        runner, _updates = self._runner()
        committed = resolve_task_contract(_task(_EXPR, {"up": _upstream("a", "b")}))
        assert committed is not None
        retry = _task(_EXPR, {"up": _upstream("a", "changed")})
        retry.recorded_resolution = committed.binding.model_dump_json()
        with pytest.raises(ExecutionError, match="committed to the inputs"):
            runner._materialize_contract(retry)

    def test_a_source_that_does_not_resolve_fails_the_task(self) -> None:
        runner, updates = self._runner()
        with pytest.raises(ExecutionError, match="does not declare a dependency"):
            runner._materialize_contract(_task(_EXPR, {"other": _upstream("a")}))
        # Nothing was recorded, so no claim can be sized from a resolution that failed.
        assert updates == []
