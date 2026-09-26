"""A store that cannot be reached pauses a workflow; it never fails one.

A settled producer's result is in the store before its success is reported, so a read
that cannot reach the store defers what depends on it and a re-drive completes it once
the store answers. Only a result that is missing or corrupt fails.
"""

import logging
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration import Advance, PublicationOutcome
from server.task import runtime as runtime_module
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from server.task.v2.representations.template import TemplateEdge
from shared.content import (
    OCTET_STREAM,
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
)
from shared.tasks.result_binding import ResultBinding
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader, store_result
from tests.server.task.test_agent_dataflow import (
    _bundle,
    _decl,
    _engine,
    _input_agent,
    _leaf,
)
from tests.server.task.test_v2_orchestration import (
    _TS,
    AUTORESEARCH,
    FakeRegistry,
    _child_count,
    _NoopSecretVault,
    _planned,
    _pop_ready,
    _register,
    _worker,
    _WorkerRegistryStub,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FlakyStore(FabricObjectStore):
    """A store whose reads fail with a chosen error while one is set."""

    def __init__(self, store: FabricObjectStore) -> None:
        self._store = store
        self.error: Exception | None = None

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._store.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        if self.error is not None:
            raise self.error
        return self._store.fetch(reference)


@pytest.fixture(autouse=True)
def _no_read_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "_FANOUT_READ_BACKOFF_SEC", 0.0)


def _runtime(
    registry: FakeRegistry,
) -> tuple[TaskRuntime, _FlakyStore, list[StoreRedriveScheduler], _Clock]:
    reader = make_result_reader()
    flaky = _FlakyStore(reader.store)
    reader._store = flaky
    clock = _Clock()
    schedulers: list[StoreRedriveScheduler] = []

    def scheduler(fire: Any, logger: logging.Logger) -> StoreRedriveScheduler:
        schedulers.append(
            StoreRedriveScheduler(fire, logger, clock=clock, run_thread=False)
        )
        return schedulers[-1]

    runtime = TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        reader,
        logging.getLogger("store-redrive"),
        secret_vault=cast(Any, _NoopSecretVault()),
        redrive=scheduler,
    )
    return runtime, flaky, schedulers, clock


@pytest.mark.anyio
async def test_a_fan_out_waits_out_an_unreachable_store_and_then_completes() -> None:
    registry = FakeRegistry()
    runtime, flaky, (scheduler,), clock = _runtime(registry)
    workflow_id, ids = await _register(runtime, AUTORESEARCH)
    planner, trial = ids["planner"], ids["trial"]
    payload = _planned(runtime, planner, ["h1", "h2", "h3"])

    flaky.error = ContentUnavailable("store down")
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", payload, _TS)
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    # Paused, not failed: no children yet, the spawn still holds the workflow open, and
    # one re-drive is pending.
    assert _child_count(engine) == 0 and not engine.region_closed("collect")
    assert runtime._tasks[trial].status != "FAILED"
    assert trial in registry.remaining_of(workflow_id)
    assert scheduler.pending(workflow_id)

    # Still away when the re-drive fires: it backs off and waits again.
    clock.now += 1.0
    assert scheduler.run_due() == [workflow_id]
    assert _child_count(engine) == 0 and scheduler.pending(workflow_id)

    flaky.error = None
    clock.now += 2.0
    assert scheduler.run_due() == [workflow_id]
    assert len(_pop_ready(runtime)) == 3 and _child_count(engine) == 3
    assert not scheduler.pending(workflow_id)


@pytest.mark.anyio
async def test_a_missing_producer_result_still_fails_the_workflow() -> None:
    registry = FakeRegistry()
    runtime, flaky, (scheduler,), _clock = _runtime(registry)
    workflow_id, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    payload = _planned(runtime, planner, ["h1"])

    flaky.error = ContentHydrationError("no such object")
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", payload, _TS)

    assert runtime._tasks[ids["trial"]].status == "FAILED"
    assert not scheduler.pending(workflow_id)


def _agent_consuming(runtime: TaskRuntime) -> Any:
    engine = _engine(
        _bundle(
            [_leaf("P"), _input_agent("M", ("reviews",))],
            [TemplateEdge(from_op="P", to_op="M", to_port="reviews")],
            (_decl("out:M", "M"),),
        ),
        granted=frozenset({"model"}),
    )
    engine.on_succeeded("P")
    reference = store_result(runtime._results, "P", {"value": "grounded"})
    bound = {"P": ResultBinding(task_id="P", reference=reference)}
    runtime._result_binding_locked = lambda task_id: bound.get(  # type: ignore[method-assign]
        task_id
    )
    return engine


def test_an_agent_input_waits_out_an_unreachable_store() -> None:
    runtime, flaky, (scheduler,), clock = _runtime(FakeRegistry())
    engine = _agent_consuming(runtime)
    runtime._engines["wfl-agent"] = engine

    flaky.error = ContentUnavailable("store down")
    runtime._stage_agent_inputs_locked("wfl-agent", engine, Advance())
    assert scheduler.run_due() == ["wfl-agent"]
    assert engine.work_item("M").outcome is None
    assert not engine.accepted_inputs_for_task("M")
    assert scheduler.pending("wfl-agent")
    # The waiting agent does not undo the backoff: nothing fires until it elapses.
    assert scheduler.run_due() == []

    flaky.error = None
    clock.now += 1.0
    assert scheduler.run_due() == ["wfl-agent"]
    assert engine.accepted_inputs_for_task("M")
    assert not scheduler.pending("wfl-agent")


def test_a_missing_agent_input_fails_the_agent() -> None:
    runtime, flaky, (scheduler,), _clock = _runtime(FakeRegistry())
    engine = _agent_consuming(runtime)
    runtime._engines["wfl-agent"] = engine

    flaky.error = ContentHydrationError("no such object")
    runtime._stage_agent_inputs_locked("wfl-agent", engine, Advance())
    scheduler.run_due()
    assert engine.work_item("M").outcome is PublicationOutcome.DECLARED_FAILURE
    assert not scheduler.pending("wfl-agent")


_AGENT_INPUT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: agent-input}
spec:
  graph:
    nodes:
      - name: p
        spec: {taskType: echo, data: {type: list, items: [grounded]}}
      - name: m
        spec:
          taskType: agent
          task: merge the reviews
          v2:
            inputs: [{name: reviews, from: p}]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


@pytest.mark.anyio
async def test_a_restart_records_the_inputs_a_crash_left_unread() -> None:
    registry = FakeRegistry()
    runtime, _flaky, (scheduler,), _clock = _runtime(registry)
    workflow_id, ids = await _register(runtime, _AGENT_INPUT_WF)
    producer, agent = ids["p"], ids["m"]
    record_dispatch(runtime, producer, cast(Any, _worker()))
    runtime.mark_succeeded(
        producer,
        "wkr-1",
        {
            "result_reference": store_result(
                runtime._results,
                producer,
                {"value": "grounded"},
                runtime._tasks[producer].org_id,
            ).model_dump(mode="json")
        },
        _TS,
    )
    # The server stops before the drive that reads the agent's input runs.
    assert scheduler.pending(workflow_id)
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None and not engine.accepted_inputs_for_task(agent)

    restored, _flaky2, (rescheduler,), _clock2 = _runtime(registry)
    restored._results = runtime._results
    assert await restored.rehydrate() == 1
    # The agent waits, blocked on its inputs.
    assert agent not in _pop_ready(restored)
    assert rescheduler.pending(workflow_id)
    rescheduler.run_due()
    restored_engine = restored.orchestration_engine(workflow_id)
    assert restored_engine is not None
    assert restored_engine.accepted_inputs_for_task(agent)
    assert agent in _pop_ready(restored)
