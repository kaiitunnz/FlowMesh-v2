"""Tests for how the dispatch loop handles a leaf whose inputs are prepared first."""

import logging
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.dispatcher.embodiment import EmbodimentSnapshot, relay_placement_task
from server.task.runtime import TaskRuntime
from shared.content import ContentReference
from shared.inference import (
    RESOLVED_INPUT_MEDIA_TYPE,
    InputResolutionBinding,
    ResolvedInputMaterialization,
)
from shared.tasks.specs import InferenceEmbodimentKind
from shared.utils.time import now_iso
from tests.server.dispatch import record_dispatch
from tests.server.dispatcher.helpers import (
    CapturingDispatcher,
    make_capturing_dispatcher,
)
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_embodiment_fence import (
    _NoopSecretVault,
    _upstream_task,
    _WorkerRegistryStub,
)
from tests.server.task.test_v2_orchestration import FakeRegistry


def _runtime() -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("preparation-dispatch-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


async def _setup(
    max_items: int | None = None, **kwargs: Any
) -> tuple[CapturingDispatcher, TaskRuntime, str]:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=max_items)
    dispatcher = make_capturing_dispatcher(
        runtime=runtime, satisfying_ids=["wkr-1"], **kwargs
    )
    return dispatcher, runtime, task_id


def _commit(runtime: TaskRuntime, task_id: str, cardinality: int) -> None:
    materialization = ResolvedInputMaterialization(
        binding=InputResolutionBinding(
            source_digest="src",
            resolver_version="1",
            request_digest="req",
            cardinality=cardinality,
        ),
        reference=ContentReference(
            authorization_scope="local",
            content_digest="d" * 64,
            size_bytes=64,
            media_type=RESOLVED_INPUT_MEDIA_TYPE,
        ),
    )
    record_dispatch(runtime, task_id, input_preparation=True)
    runtime.mark_succeeded(
        task_id,
        "wkr-1",
        {"input_materialization": materialization.model_dump(mode="json")},
        now_iso(),
    )


@pytest.mark.anyio
async def test_an_unprepared_leaf_resolves_no_embodiment_yet() -> None:
    _dispatcher, runtime, task_id = await _setup()
    assert runtime.prepares_inputs(task_id) is True
    assert runtime.resolved_embodiment(task_id) is None


@pytest.mark.anyio
async def test_a_prepared_leaf_selects_on_the_count_it_materialized() -> None:
    dispatcher, runtime, task_id = await _setup(resident_capacity_enabled=True)
    _commit(runtime, task_id, cardinality=2)
    record = runtime.get_record(task_id)
    assert record is not None

    assert dispatcher._resolve_embodiment(task_id, record) is True
    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED


@pytest.mark.anyio
async def test_a_batch_past_the_admission_bound_runs_the_other_embodiment() -> None:
    # The count is known only now, and it needs one admission slot per conversation:
    # more than a replica admits is a bound no waiting frees, so the leaf runs locally.
    dispatcher, runtime, task_id = await _setup(
        resident_capacity_enabled=True, resident_admission_slots=4
    )
    _commit(runtime, task_id, cardinality=16)
    record = runtime.get_record(task_id)
    assert record is not None

    assert dispatcher._resolve_embodiment(task_id, record) is True
    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.kind is InferenceEmbodimentKind.SELF_CONTAINED


@pytest.mark.anyio
async def test_an_unknown_count_binds_no_embodiment() -> None:
    # Reaching selection with neither a declared bound nor a prepared count would screen
    # every candidate against nothing, so it fails instead of admitting blind.
    dispatcher, runtime, task_id = await _setup(
        resident_capacity_enabled=True, grace_sec=0
    )
    record = runtime.get_record(task_id)
    assert record is not None

    assert dispatcher._resolve_embodiment(task_id, record) is False
    assert runtime.resolved_embodiment(task_id) is None
    [(failed, _message, kwargs)] = dispatcher.failed
    assert failed == task_id
    assert "unknown_batch_size" in kwargs["payload"]["reason"]


@pytest.mark.anyio
async def test_a_declared_bound_still_screens_before_any_value_exists() -> None:
    dispatcher, runtime, task_id = await _setup(
        max_items=8, resident_capacity_enabled=True
    )
    record = runtime.get_record(task_id)
    assert record is not None

    assert dispatcher._resolve_embodiment(task_id, record) is True
    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED


@pytest.mark.anyio
async def test_a_preparation_needs_no_accelerator_of_its_own() -> None:
    # It reads an upstream value and runs no model, so it places like a relay does.
    _dispatcher, runtime, task_id = await _setup()
    record = runtime.get_record(task_id)
    assert record is not None
    resources = record.task.spec.resources
    assert resources is not None and resources.hardware is not None
    assert resources.hardware.gpu is not None

    placed = relay_placement_task(record.task).spec.resources
    assert placed is not None and placed.hardware is not None
    assert placed.hardware.gpu is None


def test_an_unknown_batch_is_screened_against_nothing() -> None:
    snapshot = EmbodimentSnapshot(
        local_capable_workers=1,
        relay_capable_workers=1,
        resident_capacity_enabled=True,
        resident_admission_slots=4,
    )
    assert snapshot.admits_batch(None) is True
    assert snapshot.admits_batch(4) is True
    assert snapshot.admits_batch(5) is False
