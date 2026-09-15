"""Tests for the durable embodiment selection and its recovery fence."""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.task.runtime import TaskRuntime
from shared.inference import CanonicalInferenceRequest
from shared.tasks.specs import InferenceEmbodimentKind

from .test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _register,
    _worker,
    _WorkerRegistryStub,
)

LOCAL_ELIGIBLE = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: menu}
spec:
  graph:
    nodes:
      - name: gen
        spec:
          taskType: inference
          model:
            source: {identifier: Qwen/Qwen3-4B}
            vllm: {gpu_memory_utilization: 0.9}
          data: {type: list, items: ["hello"]}
          resources: {hardware: {gpu: {count: 1}}}
          service: {mode: local_eligible, primary: PRIMARY}
"""


def _runtime(registry: FakeRegistry) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("embodiment-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


async def _menu_task(
    runtime: TaskRuntime, primary: str = "resident_served"
) -> tuple[str, str]:
    _wfl, ids = await _register(runtime, LOCAL_ELIGIBLE.replace("PRIMARY", primary))
    task_id = ids["gen"]
    menu = runtime.embodiment_menu(task_id)
    assert menu is not None
    return task_id, menu.primary


@pytest.mark.anyio
async def test_an_unresolved_menu_task_carries_no_embodiment() -> None:
    runtime = _runtime(FakeRegistry())
    task_id, _primary = await _menu_task(runtime)
    assert runtime.resolved_embodiment(task_id) is None
    # Nothing routes resident until an embodiment is bound.
    assert runtime.service_episode_dispatch(task_id) is None


@pytest.mark.anyio
async def test_a_recorded_selection_is_readable_and_durable() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    task_id, primary = await _menu_task(runtime)

    assert runtime.record_embodiment_selection(task_id, primary, "primary", "e") == (
        primary
    )
    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.alternative_id == primary
    assert resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED

    engine = runtime.orchestration_engine(_workflow_of(runtime, task_id))
    assert engine is not None
    selection = engine.embodiment_selection(task_id)
    assert selection is not None
    assert selection.selector == "primary" and selection.evidence == "e"
    assert selection.plan_version


def _workflow_of(runtime: TaskRuntime, task_id: str) -> str:
    record = runtime.get_record(task_id)
    assert record is not None
    return record.workflow_id


@pytest.mark.anyio
async def test_a_resident_selection_routes_resident_and_a_local_one_does_not() -> None:
    runtime = _runtime(FakeRegistry())
    task_id, primary = await _menu_task(runtime)
    runtime.record_embodiment_selection(task_id, primary, "primary", "e")
    assert runtime.service_episode_dispatch(task_id) is not None

    local_runtime = _runtime(FakeRegistry())
    local_task, local_primary = await _menu_task(local_runtime, "self_contained")
    local_runtime.record_embodiment_selection(local_task, local_primary, "t", "e")
    # The leaf still declares a service dependency for its resident candidate; the
    # resolved embodiment is what decides the path.
    engine = local_runtime.orchestration_engine(_workflow_of(local_runtime, local_task))
    assert engine is not None and engine.service_dependency(local_task) is not None
    assert local_runtime.service_episode_dispatch(local_task) is None


@pytest.mark.anyio
async def test_the_attempt_records_the_embodiment_it_ran() -> None:
    runtime = _runtime(FakeRegistry())
    task_id, primary = await _menu_task(runtime)
    runtime.record_embodiment_selection(task_id, primary, "primary", "e")
    runtime.mark_dispatched(task_id, _worker())

    engine = runtime.orchestration_engine(_workflow_of(runtime, task_id))
    assert engine is not None
    snapshot = engine.to_snapshot()
    assert [a.alternative_id for a in snapshot.attempts] == [primary]


@pytest.mark.anyio
async def test_an_issued_embodiment_is_pinned_against_re_resolution() -> None:
    runtime = _runtime(FakeRegistry())
    task_id, primary = await _menu_task(runtime)
    menu = runtime.embodiment_menu(task_id)
    assert menu is not None
    other = next(
        c.alternative_id for c in menu.candidates if c.alternative_id != primary
    )

    runtime.record_embodiment_selection(task_id, primary, "primary", "e")
    # Before issue a fresh resolution may still move.
    assert runtime.record_embodiment_selection(task_id, other, "test", "e") == other

    runtime.record_embodiment_selection(task_id, primary, "primary", "e")
    runtime.mark_dispatched(task_id, _worker())
    # The dispatch issued the work item's invocation, so the embodiment is committed.
    assert runtime.record_embodiment_selection(task_id, other, "test", "e") == primary
    pinned = runtime.resolved_embodiment(task_id)
    assert pinned is not None and pinned.alternative_id == primary


@pytest.mark.anyio
async def test_a_restart_resumes_the_recorded_embodiment() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    task_id, primary = await _menu_task(runtime)
    runtime.record_embodiment_selection(task_id, primary, "primary", "e")
    runtime.mark_dispatched(task_id, _worker())

    restored = _runtime(registry)
    await restored.rehydrate()
    resolved = restored.resolved_embodiment(task_id)
    assert resolved is not None and resolved.alternative_id == primary


RESIDENT_PINNED = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: pinned}
spec:
  graph:
    nodes:
      - name: gen
        spec:
          taskType: inference
          model:
            source: {identifier: Qwen/Qwen3-4B}
            vllm: {gpu_memory_utilization: 0.9}
          data: {type: list, items: [ITEMS]}
          resources: {hardware: {gpu: {count: 1}}}
          service: {mode: resident}
"""


async def _pinned_task(runtime: TaskRuntime, items: str) -> str:
    _wfl, ids = await _register(runtime, RESIDENT_PINNED.replace("ITEMS", items))
    return ids["gen"]


@pytest.mark.anyio
async def test_a_pinned_resident_batch_carries_its_contract() -> None:
    # A pinned batch has no menu to resolve, so the contract is what carries every
    # conversation onto its one boundary and names the result they report.
    runtime = _runtime(FakeRegistry())
    task_id = await _pinned_task(runtime, '"a", "b"')
    assert runtime.embodiment_menu(task_id) is None

    contract = runtime.declared_contract(task_id)
    assert contract is not None
    assert CanonicalInferenceRequest.model_validate_json(contract).prompts == ("a", "b")
    # The pin forbids the other embodiment, so it dispatches resident with no selection.
    assert runtime.service_episode_dispatch(task_id) is not None


@pytest.mark.anyio
async def test_a_pinned_single_prompt_leaf_declares_no_contract() -> None:
    # A single-prompt pin keeps reporting the native result its embodiment always has.
    runtime = _runtime(FakeRegistry())
    assert runtime.declared_contract(await _pinned_task(runtime, '"a"')) is None


@pytest.mark.anyio
async def test_a_menu_leaf_still_carries_its_contract() -> None:
    runtime = _runtime(FakeRegistry())
    task_id, _primary = await _menu_task(runtime)
    assert runtime.declared_contract(task_id) is not None
