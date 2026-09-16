"""Tests for preparing an undeclared-envelope leaf's inputs before it is selected."""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.inference import (
    InputResolutionBinding,
    ResolvedInputMaterialization,
    ResolvedInputReference,
)
from shared.utils.time import now_iso
from tests.server.task.test_v2_embodiment_fence import (
    _NoopSecretVault,
    _upstream_task,
    _WorkerRegistryStub,
    _workflow_of,
)
from tests.server.task.test_v2_orchestration import FakeRegistry


def _runtime(
    max_prepared_input_bytes: int | None = None,
    registry: FakeRegistry | None = None,
) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry or FakeRegistry()),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(max_prepared_input_bytes=max_prepared_input_bytes),
        Path(tempfile.gettempdir()),
        logging.getLogger("input-preparation-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def _materialization(
    cardinality: int = 3, size_bytes: int = 512
) -> ResolvedInputMaterialization:
    return ResolvedInputMaterialization(
        binding=InputResolutionBinding(
            source_digest="src",
            resolver_version="1",
            request_digest="req",
            cardinality=cardinality,
            projected_output_tokens=512 * cardinality,
        ),
        reference=ResolvedInputReference(
            content_digest="d" * 64, size_bytes=size_bytes
        ),
    )


def _report(runtime: TaskRuntime, task_id: str, **kwargs: Any) -> None:
    runtime.mark_succeeded(
        task_id,
        "wkr-1",
        {"input_materialization": _materialization(**kwargs).model_dump(mode="json")},
        now_iso(),
    )


@pytest.mark.anyio
async def test_a_leaf_without_a_declared_bound_carries_an_unknown_one() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    menu = runtime.embodiment_menu(task_id)
    assert menu is not None and menu.max_batch_size is None


@pytest.mark.anyio
async def test_a_declared_bound_is_screenable_without_preparing_anything() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=8)
    menu = runtime.embodiment_menu(task_id)
    assert menu is not None and menu.max_batch_size == 8
    assert runtime.prepares_inputs(task_id) is False


@pytest.mark.anyio
async def test_a_literal_leaf_needs_no_preparation() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, "{mode: resident}", max_items=4)
    assert runtime.prepares_inputs(task_id) is False


@pytest.mark.anyio
async def test_an_undeclared_bound_prepares_before_it_runs() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    assert runtime.prepares_inputs(task_id) is True


@pytest.mark.anyio
async def test_a_committed_preparation_is_readable_and_ends_the_preparation() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    _report(runtime, task_id, cardinality=3)

    binding = runtime.input_resolution_binding(task_id)
    reference = runtime.recorded_input_reference(task_id)
    assert binding is not None and binding.cardinality == 3
    assert reference is not None and reference.content_digest == "d" * 64
    # Both facts are durable, so the leaf dispatches like any other from here.
    assert runtime.prepares_inputs(task_id) is False


@pytest.mark.anyio
async def test_a_prepared_leaf_returns_to_the_ready_queue_unsettled() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    _report(runtime, task_id)

    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.PENDING


@pytest.mark.anyio
async def test_a_preparation_commits_no_embodiment_and_no_attempt() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    engine = runtime.orchestration_engine(_workflow_of(runtime, task_id))
    assert engine is not None
    engine.on_input_preparation_dispatched(task_id, "wkr-1")
    _report(runtime, task_id)

    # Nothing about the preparation names an embodiment, so neither candidate is
    # committed and the choice is still open.
    work_item = engine.work_item(task_id)
    assert work_item is not None
    assert work_item.attempt_ids == []
    assert work_item.invocation_id is None
    assert engine.embodiment_selection(task_id) is None
    assert engine.embodiment_pinned(task_id) is False
    assert runtime.resolved_embodiment(task_id) is None


@pytest.mark.anyio
async def test_a_preparation_in_flight_leaves_the_leaf_unrunnable() -> None:
    # The bytes may already be written, but nothing binds them until the report lands,
    # so the leaf is still waiting to be prepared rather than ready to be chosen.
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    engine = runtime.orchestration_engine(_workflow_of(runtime, task_id))
    assert engine is not None
    engine.on_input_preparation_dispatched(task_id, "wkr-1")

    assert runtime.recorded_input_reference(task_id) is None
    assert runtime.prepares_inputs(task_id) is True


@pytest.mark.anyio
async def test_a_preparation_dispatch_is_recorded_in_its_own_right() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    engine = runtime.orchestration_engine(_workflow_of(runtime, task_id))
    assert engine is not None
    engine.on_input_preparation_dispatched(task_id, "wkr-1")

    preparation = engine.input_preparation(task_id)
    assert preparation is not None and preparation.worker_id == "wkr-1"


@pytest.mark.anyio
async def test_the_first_committed_preparation_stands() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    _report(runtime, task_id, cardinality=3)
    _report(runtime, task_id, cardinality=9)

    binding = runtime.input_resolution_binding(task_id)
    assert binding is not None and binding.cardinality == 3


@pytest.mark.anyio
async def test_a_preparation_success_does_not_revive_a_cancelled_task() -> None:
    # A preparation runs no model and finishes fast, so its success can land after a
    # cancel has already settled the task. Re-readying it here would re-admit cancelled
    # work and leave a PENDING record contradicting a cancelled work item.
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    record = runtime.get_record(task_id)
    assert record is not None
    record.status = TaskStatus.CANCELLED

    _report(runtime, task_id)

    assert record.status == TaskStatus.CANCELLED
    assert runtime.recorded_input_reference(task_id) is None


@pytest.mark.anyio
async def test_a_preparation_success_settles_a_cancelling_task() -> None:
    # The interrupt cannot reach a preparation the worker has already finished, and
    # withholding its next dispatch withholds the terminal that dispatch would carry,
    # so the racing success settles the cancellation rather than re-admitting the task.
    registry = FakeRegistry()
    runtime = _runtime(registry=registry)
    task_id = await _upstream_task(runtime, max_items=None)
    record = runtime.get_record(task_id)
    assert record is not None
    record.status = TaskStatus.CANCELLING

    _report(runtime, task_id)

    assert record.status == TaskStatus.CANCELLED
    assert runtime.recorded_input_reference(task_id) is None
    # The task leaves the workflow's remaining set, so the workflow can reach a
    # terminal status instead of hanging on a task no dispatch will ever settle.
    assert task_id not in registry.remaining_of(record.workflow_id)


@pytest.mark.anyio
async def test_a_replayed_preparation_success_does_not_redispatch_the_leaf() -> None:
    # The task event stream is at-least-once. A replay after the embodiment already
    # dispatched would otherwise yank the running task back to the queue and run a
    # second embodiment, admitting a second claim against one work item.
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    _report(runtime, task_id)
    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.PENDING
    record.status = TaskStatus.DISPATCHED

    _report(runtime, task_id)

    assert record.status == TaskStatus.DISPATCHED


@pytest.mark.anyio
async def test_an_unset_aggregate_limit_admits_a_large_prepared_request() -> None:
    runtime = _runtime(max_prepared_input_bytes=None)
    task_id = await _upstream_task(runtime, max_items=None)
    _report(runtime, task_id, size_bytes=50_000_000)

    assert runtime.recorded_input_reference(task_id) is not None
    assert runtime.prepares_inputs(task_id) is False


@pytest.mark.anyio
async def test_a_configured_aggregate_limit_fails_an_oversized_request() -> None:
    runtime = _runtime(max_prepared_input_bytes=1024)
    task_id = await _upstream_task(runtime, max_items=None)
    _report(runtime, task_id, size_bytes=4096)

    # It fails where no embodiment has been chosen and no claim exists.
    assert runtime.recorded_input_reference(task_id) is None
    engine = runtime.orchestration_engine(_workflow_of(runtime, task_id))
    assert engine is not None
    work_item = engine.work_item(task_id)
    assert work_item is not None and work_item.invocation_id is None


@pytest.mark.anyio
async def test_a_request_inside_the_aggregate_limit_commits() -> None:
    runtime = _runtime(max_prepared_input_bytes=4096)
    task_id = await _upstream_task(runtime, max_items=None)
    _report(runtime, task_id, size_bytes=1024)

    assert runtime.recorded_input_reference(task_id) is not None


@pytest.mark.anyio
async def test_an_unreadable_materialization_commits_nothing() -> None:
    runtime = _runtime()
    task_id = await _upstream_task(runtime, max_items=None)
    runtime.mark_succeeded(
        task_id, "wkr-1", {"input_materialization": {"binding": "?"}}, now_iso()
    )

    assert runtime.input_resolution_binding(task_id) is None
    assert runtime.recorded_input_reference(task_id) is None
