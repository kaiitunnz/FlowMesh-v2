"""The runtime's task table answers for one workflow from that workflow's own tasks."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.task.models import TaskRecord, TaskStatus
from server.task.runtime.task_table import TaskTable
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_runtime_rehydrate import (
    WIDE,
    FakeWorkflowRegistry,
    _register,
    _runtime,
)


def _record(task_id: str, workflow_id: str) -> TaskRecord:
    return cast(TaskRecord, SimpleNamespace(task_id=task_id, workflow_id=workflow_id))


def test_a_workflow_reads_its_own_tasks_in_the_order_they_were_added() -> None:
    table = TaskTable()
    table["t2"] = _record("t2", "w1")
    table["x"] = _record("x", "w2")
    table["t1"] = _record("t1", "w1")
    assert table.ids_of("w1") == ["t2", "t1"]
    assert [r.task_id for r in table.of_workflow("w2")] == ["x"]
    assert table.ids_of("w3") == []


def test_a_replaced_or_removed_task_leaves_its_workflow() -> None:
    table = TaskTable()
    table["t"] = _record("t", "w1")
    table["t"] = _record("t", "w2")
    assert (table.ids_of("w1"), table.ids_of("w2")) == ([], ["t"])
    table.setdefault("u", _record("u", "w2"))
    table.update({"v": _record("v", "w2")})
    assert table.ids_of("w2") == ["t", "u", "v"]
    del table["t"]
    assert table.pop("u").task_id == "u"
    assert table.pop("missing", None) is None
    assert table.ids_of("w2") == ["v"]
    table.clear()
    assert table.ids_of("w2") == [] and not table


@pytest.mark.anyio
async def test_settlement_reads_only_its_workflows_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(FakeWorkflowRegistry())
    workflow_id, ids = await _register(runtime, WIDE)
    await _register(runtime, WIDE)
    for task_id in ids.values():
        record_dispatch(runtime, task_id)
        runtime.mark_succeeded(task_id, "wkr-1", {}, "2026-06-01T00:00:00Z")

    def every_task(*args: Any) -> Any:
        raise AssertionError("settlement walked every task of the runtime")

    for scan in ("values", "items", "keys", "__iter__"):
        monkeypatch.setattr(TaskTable, scan, every_task)
    settlement = runtime.workflow_settlement(workflow_id)
    assert settlement.settled
    assert all(
        record.status == TaskStatus.DONE
        for record in runtime._tasks.of_workflow(workflow_id)
    )
