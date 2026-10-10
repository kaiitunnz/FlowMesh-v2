"""A repayment of held writes never lets the durable ledger lead the task records.

A held ledger save owed ahead of a held record write is made after it: whichever write
of the repayment a crash lands after, every work item the durable ledger settled has
its durable task record settled, and the restored workflow runs to its end with its
durable remaining set empty.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from server.orchestration.state import TERMINAL_WORK_ITEM_STATUSES
from server.task.models import SETTLING_TASK_STATUSES, TaskStatus
from server.task.runtime import TaskRuntime
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_runtime_durability_faults import (
    _FaultyRegistry,
    _restarted,
)
from tests.server.task.test_v2_orchestration import (
    _TS,
    AUTORESEARCH,
    LINEAR,
    _planned,
    _register,
)
from tests.support.waiting import pop_ready

PAIR = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: pair}
spec:
  graph:
    nodes:
      - name: x
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: y
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""


class _Store(_FaultyRegistry):
    """Refuses each kind of write in ``refused``, besides the fault cut."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: set[str] = set()

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        self._refuse("records")
        super().commit_transition(workflow_id, **kwargs)

    def commit_dynamic_tasks(self, workflow_id: str, *args: Any, **kwargs: Any) -> None:
        self._refuse("ledger")
        super().commit_dynamic_tasks(workflow_id, *args, **kwargs)

    def save_ledger_snapshot(
        self, workflow_id: str, snapshot: Any, control: Any = None
    ) -> None:
        self._refuse("ledger")
        super().save_ledger_snapshot(workflow_id, snapshot)

    def _refuse(self, kind: str) -> None:
        if kind in self.refused:
            raise ConnectionError(f"{kind} refused")


@dataclass
class _Scenario:
    name: str
    template: str
    # Runs the transition whose record write is held behind the held ledger save.
    act: Callable[[TaskRuntime, dict[str, str], str], None]


def _settle(runtime: TaskRuntime, ids: dict[str, str], workflow_id: str) -> None:
    runtime.mark_succeeded(ids["y"], "wkr-1", {}, _TS, "dsp-y")


def _skip(runtime: TaskRuntime, ids: dict[str, str], workflow_id: str) -> None:
    runtime.mark_succeeded(
        ids["y"],
        None,
        {"finished_at": _TS, "started_at": _TS},
        _TS,
        skip={"skipped": True, "reason": "condition_not_met"},
    )


def _cancel(runtime: TaskRuntime, ids: dict[str, str], workflow_id: str) -> None:
    runtime.cancel_workflow(workflow_id)


def _fan_out(runtime: TaskRuntime, ids: dict[str, str], workflow_id: str) -> None:
    planner = ids["planner"]
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["h1", "h2"]), _TS, "dsp-planner"
    )


def _fan_out_cancelled(
    runtime: TaskRuntime, ids: dict[str, str], workflow_id: str
) -> None:
    _fan_out(runtime, ids, workflow_id)
    runtime.cancel_workflow(workflow_id)


_SCENARIOS = (
    _Scenario("settle", PAIR, _settle),
    _Scenario("skip", PAIR, _skip),
    _Scenario("cancel", LINEAR, _cancel),
    _Scenario("fan-out", AUTORESEARCH, _fan_out),
    # Its children settle while their records and the ledger are held.
    _Scenario("fan-out-cancelled", AUTORESEARCH, _fan_out_cancelled),
)


def _dispatch_ready(runtime: TaskRuntime) -> None:
    while (task_id := pop_ready(runtime, 0.02)) is not None:
        record_dispatch(runtime, task_id, "wkr-1", f"dsp-{task_id}")


def _indebted(scenario: _Scenario) -> tuple[_Store, TaskRuntime, str]:
    """A workflow owing a ledger save made before a record write it also owes."""
    store = _Store()
    runtime = _runtime(store)
    workflow_id, ids = asyncio.run(_register(runtime, scenario.template))
    names = {task_id: name for name, task_id in ids.items()}
    dispatched = []
    while (task_id := pop_ready(runtime, 0.02)) is not None:
        record_dispatch(runtime, task_id, "wkr-1", f"dsp-{names[task_id]}")
        dispatched.append(task_id)
    store.refused = {"ledger"}
    runtime.mark_started(dispatched[0], "wkr-1", {}, _TS, f"dsp-{names[dispatched[0]]}")
    store.refused = {"ledger", "records"}
    scenario.act(runtime, ids, workflow_id)
    assert runtime._committer.debt[workflow_id]
    store.refused = set()
    return store, runtime, workflow_id


def _repay_writes(scenario: _Scenario) -> int:
    store, runtime, workflow_id = _indebted(scenario)
    before = store.writes
    with runtime._lock:
        assert runtime._committer.close_locked(workflow_id)
    return store.writes - before


def _assert_ledger_never_leads(store: _Store, workflow_id: str) -> None:
    ledger = store.ledger(workflow_id)
    assert ledger is not None
    for item in ledger.work_items:
        if item.legacy_task_id is None or item.status not in (
            TERMINAL_WORK_ITEM_STATUSES
        ):
            continue
        record = store.record(item.legacy_task_id)
        assert record is not None, item.legacy_task_id
        assert record.status in SETTLING_TASK_STATUSES, (item, record.status)


def _run_to_end(runtime: TaskRuntime, workflow_id: str) -> None:
    for _ in range(10):
        if runtime.workflow_settlement(workflow_id).settled:
            return
        for record in runtime.task_records():
            if record.status == TaskStatus.DISPATCHED:
                assert record.assigned_worker is not None
                runtime.mark_succeeded(
                    record.task_id, record.assigned_worker, {}, _TS, record.dispatch_id
                )
            elif record.status == TaskStatus.CANCELLING:
                assert record.assigned_worker is not None
                runtime.mark_cancelled(
                    record.task_id, record.assigned_worker, {}, _TS, record.dispatch_id
                )
        _dispatch_ready(runtime)


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s.name)
def test_a_crash_between_repaid_writes_restores_a_workflow_that_settles(
    scenario: _Scenario,
) -> None:
    writes = _repay_writes(scenario)
    assert writes >= 2
    for cut in range(writes + 1):
        store, runtime, workflow_id = _indebted(scenario)
        store.fail_from = store.writes + cut + 1
        runtime._retry_durability(workflow_id)
        _assert_ledger_never_leads(store, workflow_id)

        runtime.shutdown()
        store.heal()
        restored = _restarted(runtime, store)
        asyncio.run(restored.rehydrate())
        _run_to_end(restored, workflow_id)
        settlement = restored.workflow_settlement(workflow_id)
        assert settlement.settled, (cut, restored.task_records())
        assert store.remaining_of(workflow_id) == set(), cut
        restored.shutdown()


class _Ambiguous(_FaultyRegistry):
    """Applies one child seam and loses its reply, then refuses every write while
    ``down``."""

    def __init__(self) -> None:
        super().__init__()
        self.ambiguous_seams = 0
        self.down = False

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        if self.down:
            raise ConnectionError("down")
        super().commit_transition(workflow_id, **kwargs)

    def save_ledger_snapshot(
        self, workflow_id: str, snapshot: Any, control: Any = None
    ) -> None:
        if self.down:
            raise ConnectionError("down")
        super().save_ledger_snapshot(workflow_id, snapshot)

    def commit_dynamic_tasks(self, workflow_id: str, *args: Any, **kwargs: Any) -> None:
        if self.down:
            raise ConnectionError("down")
        super().commit_dynamic_tasks(workflow_id, *args, **kwargs)
        if self.ambiguous_seams:
            self.ambiguous_seams -= 1
            self.down = True
            raise ConnectionError("reply lost after EXEC")


def test_children_settled_after_an_ambiguous_seam_leave_the_remaining_set() -> None:
    store = _Ambiguous()
    runtime = _runtime(store)
    workflow_id, ids = asyncio.run(_register(runtime, AUTORESEARCH))
    planner = ids["planner"]
    assert pop_ready(runtime, 0.05) == planner
    record_dispatch(runtime, planner, "wkr-1", "dsp-planner")
    store.ambiguous_seams = 1
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["h1", "h2"]), _TS, "dsp-planner"
    )
    children = runtime._committer._unwritten_children[workflow_id].copy()
    assert children <= store.remaining_of(workflow_id)

    runtime.cancel_workflow(workflow_id)
    store.down = False
    runtime._retry_durability(workflow_id)

    assert workflow_id not in runtime._committer.debt
    assert runtime.workflow_settlement(workflow_id).settled
    assert not children & store.remaining_of(workflow_id)
