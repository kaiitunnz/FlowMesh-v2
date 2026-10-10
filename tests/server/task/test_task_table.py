"""The runtime's task table answers for one workflow from that workflow's own tasks."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.task.models import TaskRecord, TaskStatus, TerminalStatusReverted
from server.task.runtime.reports import reset_to_pending
from server.task.runtime.task_table import TaskTable
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_runtime_rehydrate import (
    WIDE,
    FakeWorkflowRegistry,
    _register,
    _runtime,
)


def _record(
    task_id: str,
    workflow_id: str,
    status: str = TaskStatus.PENDING,
    finished_ts: float | None = None,
) -> TaskRecord:
    return cast(
        TaskRecord,
        SimpleNamespace(
            task_id=task_id,
            workflow_id=workflow_id,
            status=status,
            finished_ts=finished_ts,
        ),
    )


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


def test_a_workflow_is_settled_once_its_last_open_task_is_terminal() -> None:
    table = TaskTable()
    table["a"] = _record("a", "w", TaskStatus.DONE, finished_ts=5.0)
    table["b"] = _record("b", "w")
    table["c"] = _record("c", "w", TaskStatus.FAILED, finished_ts=9.0)
    assert table.first_unsettled("w") == "b"
    assert table.last_finish("w") == 5.0
    table["b"].status = TaskStatus.CANCELLED
    table["b"].finished_ts = 7.0
    assert table.first_unsettled("w") is None
    assert table.last_finish("w") == 9.0
    # A task added after the workflow settled opens it again.
    table["d"] = _record("d", "w")
    assert table.first_unsettled("w") == "d"
    assert table.holds("w") and not table.holds("other")


def test_tasks_settling_behind_a_long_open_task_are_dropped_as_found() -> None:
    table = TaskTable()
    table["head"] = _record("head", "w")
    for i in range(3):
        table[f"t{i}"] = _record(f"t{i}", "w")
    assert table.first_unsettled("w") == "head"
    for i in range(3):
        table[f"t{i}"].status = TaskStatus.DONE
        table[f"t{i}"].finished_ts = float(i)
    assert table.first_unsettled("w") == "head"
    # Each call walks only up to an open task, so the settled ones leave the walk.
    assert list(table._open["w"]) == ["head"]
    assert table.last_finish("w") == 2.0
    assert table.ids_of("w") == ["head", "t0", "t1", "t2"]


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


@pytest.mark.anyio
async def test_a_terminal_task_never_returns_to_an_active_status() -> None:
    runtime = _runtime(FakeWorkflowRegistry())
    _, ids = await _register(runtime, WIDE)
    task_id = ids["a"]
    record_dispatch(runtime, task_id)
    runtime.mark_succeeded(task_id, "wkr-1", {}, "2026-06-01T00:00:00Z")
    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.DONE
    with pytest.raises(TerminalStatusReverted):
        reset_to_pending(record)
    with pytest.raises(TerminalStatusReverted):
        record.status = TaskStatus.CANCELLING
    record.status = TaskStatus.DONE
    assert record.status == TaskStatus.DONE
