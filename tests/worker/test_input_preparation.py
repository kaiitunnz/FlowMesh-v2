"""Tests for the worker leg of preparing an inference leaf's inputs."""

from typing import Any, cast
from unittest import mock

import pytest

from shared.content import ContentReference, ContentUnavailable
from shared.inference import (
    RESOLVED_INPUT_MEDIA_TYPE,
    ResolvedInputMaterialization,
    canonical_contract,
    hydrate_resolved_input,
)
from shared.schemas.event import TaskFailureKind
from shared.schemas.result.catalog import InferenceResult
from shared.schemas.result.payloads import InferenceItem
from shared.tasks.specs import InferenceSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.shared.outcome_helpers import InMemoryContentStore
from tests.worker.factories import make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.runner import Runner

_UNBOUNDED = {"type": "list", "expr": "up.items.output"}


def _upstream(*outputs: str) -> dict[str, Any]:
    result = InferenceResult(
        model="up/model",
        items=[
            InferenceItem(index=index, prompt=f"p{index}", output=output)
            for index, output in enumerate(outputs)
        ],
    )
    return {"up": result.model_dump(mode="json")}


def _task(
    data: dict[str, Any] | None = None,
    *,
    upstream: dict[str, Any] | None = None,
    **kwargs: Any,
) -> WorkerTaskMessage:
    body: dict[str, Any] = {
        "taskType": "inference",
        "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
        "data": data if data is not None else dict(_UNBOUNDED),
    }
    body["_upstreamResults"] = upstream if upstream is not None else _upstream("a", "b")
    spec = InferenceSpecStrict.model_validate(body)
    return make_worker_task_message(
        spec.model_dump(by_alias=True),
        declared_contract=canonical_contract(spec),
        **kwargs,
    )


class _Plane:
    """A content plane whose every task reads and writes one in-memory store."""

    def __init__(self, store: InMemoryContentStore) -> None:
        self._store = store

    def for_task(self, task_id: str) -> InMemoryContentStore:
        return self._store


def _runner(store: InMemoryContentStore | None) -> Runner:
    """A runner with only what resolving and storing a request reads."""
    runner = object.__new__(Runner)
    runner.lifecycle = mock.Mock()
    runner.lifecycle.content_plane = (
        cast(Any, _Plane(store)) if store is not None else None
    )
    return runner


def _prepare(
    store: InMemoryContentStore, msg: WorkerTaskMessage
) -> ResolvedInputMaterialization:
    return _runner(store)._prepare_inputs(msg)


def _materialize(store: InMemoryContentStore, msg: WorkerTaskMessage) -> None:
    _runner(store)._materialize_contract(msg)


class TestPreparation:
    def test_an_unbounded_source_prepares_the_request_it_resolved(self) -> None:
        store = InMemoryContentStore()
        prepared = _prepare(store, _task())

        assert prepared.binding.cardinality == 2
        assert prepared.reference.media_type == RESOLVED_INPUT_MEDIA_TYPE
        assert prepared.reference.size_bytes > 0

    def test_the_stored_object_holds_the_exact_request_the_binding_names(self) -> None:
        store = InMemoryContentStore()
        prepared = _prepare(store, _task(upstream=_upstream("x", "y", "z")))

        hydrated = hydrate_resolved_input(store, prepared.reference)
        assert hydrated.request.prompts == ("x", "y", "z")
        assert hydrated.binding.matches(prepared.binding)

    def test_the_prompts_stay_out_of_what_the_worker_reports(self) -> None:
        # Only the request's identity travels; the prompts stay in the object.
        store = InMemoryContentStore()
        prepared = _prepare(store, _task(upstream=_upstream("secret prompt")))

        assert "secret prompt" not in prepared.model_dump_json()

    def test_a_source_that_does_not_resolve_stores_nothing(self) -> None:
        store = InMemoryContentStore()
        with pytest.raises(ExecutionError) as excinfo:
            _prepare(store, _task(upstream={"up": {"model": "m", "items": []}}))

        assert excinfo.value.retryable is False
        assert store.write_count == 0

    def test_a_worker_without_a_content_store_prepares_nothing(self) -> None:
        with pytest.raises(ExecutionError) as excinfo:
            Runner._prepare_inputs(_runner(None), _task())

        # The task has not failed on its own terms, so another worker can take it.
        assert excinfo.value.retryable is True


class TestHydration:
    def test_a_prepared_task_runs_the_recorded_request(self) -> None:
        store = InMemoryContentStore()
        prepared = _prepare(store, _task(upstream=_upstream("one", "two")))
        msg = _task(
            upstream={},
            recorded_input=prepared.reference,
            recorded_resolution=prepared.binding,
        )
        _materialize(store, msg)

        # It resolved nothing: the upstream value is gone and the request still runs.
        assert msg.resolved_contract is not None
        assert msg.resolved_contract.prompts == ("one", "two")

    def test_a_relocated_run_reads_the_object_rather_than_the_source(self) -> None:
        store = InMemoryContentStore()
        prepared = _prepare(store, _task(upstream=_upstream("one", "two")))
        # A substituted upstream value would resolve to a different request; the
        # recorded one is what runs.
        msg = _task(
            upstream=_upstream("swapped", "values"),
            recorded_input=prepared.reference,
            recorded_resolution=prepared.binding,
        )
        _materialize(store, msg)

        assert msg.resolved_contract is not None
        assert msg.resolved_contract.prompts == ("one", "two")

    def test_a_missing_object_fails_before_any_model_reaches_it(self) -> None:
        store = InMemoryContentStore()
        msg = _task(
            recorded_input=ContentReference(
                authorization_scope="local",
                content_digest="0" * 64,
                size_bytes=16,
                media_type=RESOLVED_INPUT_MEDIA_TYPE,
            )
        )
        with pytest.raises(ExecutionError) as excinfo:
            _materialize(store, msg)

        assert excinfo.value.retryable is False
        assert msg.resolved_contract is None

    def test_an_unreachable_store_reports_the_request_unavailable(self) -> None:
        store = InMemoryContentStore()
        prepared = _prepare(store, _task())

        class _Away(InMemoryContentStore):
            def fetch(self, reference: ContentReference) -> bytes:
                raise ContentUnavailable("store down")

        with pytest.raises(ExecutionError) as excinfo:
            _materialize(_Away(), _task(recorded_input=prepared.reference))
        assert excinfo.value.retryable is True
        assert excinfo.value.failure_kind is TaskFailureKind.INPUT_UNAVAILABLE

    def test_content_that_is_not_the_digest_names_fails_closed(self) -> None:
        store = InMemoryContentStore()
        prepared = _prepare(store, _task())
        reference = prepared.reference
        store._objects[(reference.authorization_scope, reference.content_digest)] = (
            b"{}"
        )

        with pytest.raises(ExecutionError) as excinfo:
            _materialize(store, _task(recorded_input=reference))
        assert excinfo.value.retryable is False

    def test_an_object_recording_another_resolution_fails_closed(self) -> None:
        store = InMemoryContentStore()
        prepared = _prepare(store, _task(upstream=_upstream("one", "two")))
        other = _prepare(store, _task(upstream=_upstream("three")))

        with pytest.raises(ExecutionError) as excinfo:
            _materialize(
                store,
                _task(
                    recorded_input=other.reference,
                    recorded_resolution=prepared.binding,
                ),
            )
        assert excinfo.value.retryable is False

    def test_a_preparation_whose_report_was_lost_rewrites_the_same_object(self) -> None:
        # The object is named by its own content, so the preparation that re-runs after
        # a lost report leaves one object rather than a second, competing one.
        store = InMemoryContentStore()
        first = _prepare(store, _task())
        second = _prepare(store, _task())

        assert first.reference == second.reference
        assert store.write_count == 1
