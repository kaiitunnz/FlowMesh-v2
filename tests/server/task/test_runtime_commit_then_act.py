"""A runtime transition's side effects run only after its durable commit."""

import asyncio
import logging
from collections.abc import Iterator
from typing import Any, cast
from unittest import mock

import pytest

from server.config import OrchestrationConfig
from server.orchestration import OrchestrationEngine
from server.orchestration.state import InvocationState, LedgerSnapshot
from server.registries.workflow import PersistedTask
from server.task.models import PublishGate, TaskStatus
from server.task.runtime import TaskRuntime, TransitionNotDurable
from server.task.workflow_retry import WorkflowRetryScheduler
from shared.inference import InputResolutionBinding, UpstreamProvenance
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader
from tests.server.task.test_resident_origin_loss import (
    _RESIDENT_WF,
    _capture_resident_boundary,
)
from tests.server.task.test_v2_embodiment_fence import _menu_task
from tests.server.task.test_v2_embodiment_fence import _runtime as _menu_runtime
from tests.server.task.test_v2_orchestration import (
    _TS,
    AUTORESEARCH,
    FakeRegistry,
    _planned,
    _register,
    _worker,
)
from tests.server.task.test_worker_originated_boundary import _WorkerStub
from tests.support.waiting import pop_ready


def _runtime(registry: FakeRegistry) -> TaskRuntime:
    """A runtime whose durability retry runs only when a test drives it."""
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("commit-then-act"),
        credential_vault=InMemoryCredentialVault(),
        durability_retry=lambda fire, logger: WorkflowRetryScheduler(
            fire, logger, base_delay_sec=0.0, run_thread=False
        ),
    )


def _durable_invocation(
    registry: FakeRegistry, workflow_id: str, invocation_id: str
) -> InvocationState | None:
    if (blob := registry.ledger_blobs.get(workflow_id)) is None:
        return None
    snapshot = LedgerSnapshot.model_validate_json(blob)
    return next(
        (i.state for i in snapshot.invocations if i.invocation_id == invocation_id),
        None,
    )


class _Resident:
    """A resident agent boundary issued to resident admission, with every credit
    release recorded beside the durable state of its invocation at that moment."""

    def __init__(self) -> None:
        self.registry = FakeRegistry()
        self.runtime = _runtime(self.registry)
        self.issued: list[tuple[Any, InvocationState | None]] = []
        self.runtime._resident_originate = self._originate
        self.releases: list[tuple[str, bool, InvocationState | None]] = []
        self.runtime.set_resident_terminal_hook(self._release)
        self.workflow_id, ids = asyncio.run(_register(self.runtime, _RESIDENT_WF))
        self.writer = ids["writer"]
        self._save = self.registry.save_ledger_snapshot

    def _originate(self, env: Any) -> bool:
        self.issued.append((env, self._durable(env.invocation_id)))
        return True

    def _release(self, invocation_id: str, failed: bool) -> None:
        self.releases.append((invocation_id, failed, self._durable(invocation_id)))

    def _durable(self, invocation_id: str) -> InvocationState | None:
        return _durable_invocation(self.registry, self.workflow_id, invocation_id)

    def capture(self) -> Any:
        _capture_resident_boundary(self.runtime, self.writer)
        ((env, _),) = self.issued
        return env

    def ledger_down(self) -> None:
        def down(*_: Any, **__: Any) -> None:
            raise ConnectionError("control redis unavailable")

        self.registry.save_ledger_snapshot = down  # type: ignore[method-assign]

    def ledger_up(self) -> None:
        self.registry.save_ledger_snapshot = self._save  # type: ignore[method-assign]


@pytest.mark.parametrize("error", [None, "upstream failed"])
def test_a_failed_save_outside_a_report_holds_the_credit_for_its_retry(
    error: str | None,
) -> None:
    resident = _Resident()
    env = resident.capture()
    value = None if error is not None else "a completion"

    resident.ledger_down()
    assert resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, value, error=error
    )
    assert resident.releases == []
    resident.ledger_up()

    # The resident controller settles again; the boundary already settled in memory,
    # so the retry is absorbed, and it makes the first settle durable.
    assert not resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, value, error=error
    )
    assert not resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, value, error=error
    )
    assert resident.releases == [
        (env.invocation_id, error is not None, InvocationState.TERMINAL)
    ]


def test_a_settle_no_caller_retries_is_made_durable_by_the_retry_alone() -> None:
    resident = _Resident()
    env = resident.capture()

    resident.ledger_down()
    resident.runtime.settle_episode_invocation(
        resident.writer, env.call_correlation, "a completion"
    )
    assert resident.runtime._durability.run_due() == [resident.workflow_id]
    assert resident.releases == []
    resident.ledger_up()

    assert resident.runtime._durability.run_due() == [resident.workflow_id]

    assert resident.releases == [(env.invocation_id, False, InvocationState.TERMINAL)]
    assert not resident.runtime._durability.pending(resident.workflow_id)


def test_a_settle_inside_a_held_report_releases_nothing_before_its_save() -> None:
    resident = _Resident()
    env = resident.capture()

    resident.ledger_down()
    with resident.runtime._transition(raises=False):
        resident.runtime._settle_episode_invocation(
            resident.writer, env.call_correlation, "a completion"
        )

    assert resident.releases == []
    assert resident._durable(env.invocation_id) is InvocationState.ISSUED


def test_a_boundary_issues_to_its_handler_only_once_its_routing_is_durable() -> None:
    resident = _Resident()

    resident.ledger_down()
    with pytest.raises(TransitionNotDurable):
        _capture_resident_boundary(resident.runtime, resident.writer)

    # Nothing reached resident admission for an invocation the ledger may lose.
    assert resident.issued == []
    resident.ledger_up()

    assert resident.runtime._durability.run_due() == [resident.workflow_id]

    ((env, durable),) = resident.issued
    assert durable is InvocationState.ISSUED
    assert resident._durable(env.invocation_id) is InvocationState.ISSUED


def test_a_crash_after_a_held_routing_leaves_no_claim_behind() -> None:
    resident = _Resident()
    resident.ledger_down()
    with pytest.raises(TransitionNotDurable):
        _capture_resident_boundary(resident.runtime, resident.writer)
    resident.runtime.shutdown()
    resident.ledger_up()

    restored = _runtime(resident.registry)
    restored._resident_originate = resident._originate
    assert asyncio.run(restored.rehydrate()) == 1

    # The routing never became durable, so its invocation never reached resident
    # admission: no claim exists for an invocation the restored ledger lacks, and the
    # agent runs its step again.
    assert resident.issued == []
    record = restored.get_record(resident.writer)
    assert record is not None and record.status == TaskStatus.PENDING


def _binding(request_digest: str, cardinality: int) -> Any:
    return InputResolutionBinding(
        source_digest="src",
        resolver_version="1",
        request_digest=request_digest,
        cardinality=cardinality,
        upstream=(UpstreamProvenance(node="up", content_digest="c1"),),
    ).model_dump(mode="json")


async def _resolving(registry: FakeRegistry) -> tuple[TaskRuntime, str]:
    runtime = _menu_runtime(registry)
    task_id, primary = await _menu_task(runtime, "self_contained")
    assert runtime.record_embodiment_selection(task_id, primary, "primary", "e")
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    return runtime, task_id


@pytest.mark.anyio
async def test_a_recorded_input_resolution_survives_a_restart() -> None:
    registry = FakeRegistry()
    runtime, task_id = await _resolving(registry)

    runtime.record_input_resolution(task_id, "wkr-1", _binding("req", 2), "dsp-1")

    restored = _menu_runtime(registry)
    assert await restored.rehydrate() == 1
    standing = restored.input_resolution_binding(task_id)
    assert standing is not None and standing.request_digest == "req"
    # A re-drive that resolves to a different request leaves the recorded one.
    restored.record_input_resolution(
        task_id, "wkr-1", _binding("other", 9), restored._tasks[task_id].dispatch_id
    )
    kept = restored.input_resolution_binding(task_id)
    assert kept is not None and kept.request_digest == "req"


@pytest.mark.anyio
async def test_an_input_resolution_is_acknowledged_only_once_durable() -> None:
    registry = FakeRegistry()
    runtime, task_id = await _resolving(registry)
    save = registry.save_ledger_snapshot

    def down(*_: Any, **__: Any) -> None:
        raise ConnectionError("control redis unavailable")

    registry.save_ledger_snapshot = down  # type: ignore[method-assign]
    with pytest.raises(TransitionNotDurable):
        runtime.record_input_resolution(task_id, "wkr-1", _binding("req", 2), "dsp-1")
    # The equal report handed over again is not acknowledged while the store is down.
    with pytest.raises(TransitionNotDurable):
        runtime.record_input_resolution(task_id, "wkr-1", _binding("req", 2), "dsp-1")
    registry.save_ledger_snapshot = save  # type: ignore[method-assign]

    runtime.record_input_resolution(task_id, "wkr-1", _binding("req", 2), "dsp-1")

    restored = _menu_runtime(registry)
    assert await restored.rehydrate() == 1
    standing = restored.input_resolution_binding(task_id)
    assert standing is not None and standing.request_digest == "req"


class _ChildrenDown(FakeRegistry):
    """Refuses every spawned-children commit while ``down``."""

    down = False

    def commit_dynamic_tasks(self, workflow_id: str, *args: Any, **kwargs: Any) -> None:
        if self.down:
            raise ConnectionError("control redis unavailable")
        super().commit_dynamic_tasks(workflow_id, *args, **kwargs)


@pytest.mark.anyio
async def test_a_child_whose_materialization_is_held_is_not_published() -> None:
    registry = _ChildrenDown()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    assert pop_ready(runtime) == planner
    record_dispatch(runtime, planner, "wkr-1", "dsp-1")
    registry.down = True
    with pytest.raises(TransitionNotDurable):
        runtime.mark_succeeded(
            planner, "wkr-1", _planned(runtime, planner, ["h1", "h2"]), _TS, "dsp-1"
        )
    child, sibling = pop_ready(runtime), pop_ready(runtime)
    assert child is not None and sibling is not None
    assert child not in registry.task_blobs

    worker = cast(Any, _worker("wkr-2"))
    assert runtime.begin_publish(child, worker, "dsp-2") is PublishGate.NOT_DURABLE
    assert child not in runtime._fence.publishing
    # Another workflow's work publishes meanwhile.
    _, other = await _register(runtime, AUTORESEARCH)
    assert pop_ready(runtime) == other["planner"]
    assert (
        runtime.begin_publish(other["planner"], worker, "dsp-3") is PublishGate.PUBLISH
    )
    registry.down = False

    assert runtime.begin_publish(child, worker, "dsp-2") is PublishGate.PUBLISH
    assert child in registry.dynamic_task_ids[workflow_id]
    assert child in registry.task_blobs


_DENIED_ROOT = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: denied-root}
spec:
  graph:
    nodes:
      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
      - name: after
        dependsOn: [caller]
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""


@pytest.fixture
def denied_root() -> Iterator[None]:
    """Build every engine under an empty root grant, so the initial advance fails the
    root and everything downstream of it."""
    build = OrchestrationEngine.build

    def denying(*args: Any, **kwargs: Any) -> OrchestrationEngine:
        return build(*args, **{**kwargs, "granted_interfaces": frozenset()})

    with mock.patch.object(OrchestrationEngine, "build", side_effect=denying):
        yield


def _durable_statuses(registry: FakeRegistry, ids: dict[str, str]) -> dict[str, Any]:
    return {
        name: PersistedTask.model_validate_json(registry.task_blobs[task_id]).record
        for name, task_id in ids.items()
    }


@pytest.mark.usefixtures("denied_root")
def test_a_crash_before_a_registration_goes_live_restores_its_initial_advance() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    with mock.patch.object(TaskRuntime, "_install_registration"):
        workflow_id, ids = asyncio.run(_register(runtime, _DENIED_ROOT))
    runtime.shutdown()

    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 1

    assert restored.orchestration_engine(workflow_id) is not None
    durable = _durable_statuses(registry, ids)
    assert durable["caller"].status == TaskStatus.FAILED
    assert durable["caller"].error is not None
    assert "authority denied" in durable["caller"].error
    assert durable["after"].status == TaskStatus.FAILED
    assert pop_ready(restored) is None


@pytest.mark.usefixtures("denied_root")
def test_a_held_initial_advance_leaves_its_registration_standing() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    commit = registry.commit_transition

    def down(*_: Any, **__: Any) -> None:
        raise ConnectionError("control redis unavailable")

    registry.commit_transition = down  # type: ignore[method-assign]
    workflow_id, ids = asyncio.run(_register(runtime, _DENIED_ROOT))

    assert workflow_id in registry.workflow_task_ids
    assert workflow_id in registry.ledger_blobs
    assert runtime._durability.pending(workflow_id)
    registry.commit_transition = commit  # type: ignore[method-assign]

    assert runtime._durability.run_due() == [workflow_id]

    durable = _durable_statuses(registry, ids)
    assert {record.status for record in durable.values()} == {TaskStatus.FAILED}
    assert not runtime._durability.pending(workflow_id)
