"""A freed workflow's tasks and outputs read as they did while it was resident."""

from collections.abc import Callable
from typing import Any

import pytest
from starlette.datastructures import QueryParams

from server.task.models import TaskOrder, TaskStatus, task_order
from server.task.runtime import TaskRuntime
from server.utils.query import QueryFilter
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_region_outputs import (
    _CONCAT,
    _FAN_OVER_CHILD,
    _LOOP,
    _LOOP_BODY,
    _PENDING_OUTPUTS,
    _TWO_RETURNS,
)
from tests.server.task.test_runtime_control_flow import _Run, _workflow
from tests.server.task.test_settled_history import (
    _TS,
    CHAIN,
    _register,
    _registry,
    _resident,
    _runtime,
    _succeed_ready,
)
from tests.server.task.test_v2_orchestration import LINEAR
from tests.support.waiting import pop_ready

_FIELDS = frozenset({"workflow_id", "status", "task_type", "completed", "depends_on"})


def _described(runtime: TaskRuntime, task_ids: list[str]) -> dict[str, Any]:
    return {
        task_id: info.model_dump(mode="json") if info is not None else None
        for task_id in task_ids
        for info in [runtime.describe_task(task_id)]
    }


def _fail_first(runtime: TaskRuntime) -> None:
    task_id = pop_ready(runtime)
    assert task_id is not None
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    runtime.mark_failed(task_id, "wkr-1", {}, _TS)


def _cancel(runtime: TaskRuntime) -> None:
    record = next(iter(runtime._tasks.values()))
    runtime.cancel_workflow(record.workflow_id)


def _succeed(runtime: TaskRuntime) -> None:
    _succeed_ready(runtime)


@pytest.mark.parametrize("text", [CHAIN, LINEAR])
@pytest.mark.parametrize("settle", [_succeed, _fail_first, _cancel])
def test_a_freed_task_describes_records_and_reads_as_it_did(
    text: str, settle: Callable[[TaskRuntime], None]
) -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, text)
    settle(runtime)
    task_ids = runtime._tasks.ids_of(workflow_id)
    described = _described(runtime, task_ids)
    records = {t: runtime.task_record(t) for t in task_ids}
    bindings = {t: runtime.result_binding(t) for t in task_ids}
    results = {t: runtime.read_result_bytes(t) for t in task_ids}

    runtime.settled_history.drain()

    assert not _resident(runtime, workflow_id)
    assert _described(runtime, task_ids) == described
    assert {
        t: (
            record.model_dump(mode="json")
            if (record := runtime.task_record(t))
            else None
        )
        for t in task_ids
    } == {
        t: record.model_dump(mode="json") if record else None
        for t, record in records.items()
    }
    assert {t: runtime.result_binding(t) for t in task_ids} == bindings
    assert {t: runtime.read_result_bytes(t) for t in task_ids} == results
    assert runtime.describe_task("tsk-unknown") is None
    assert runtime.task_record("tsk-unknown") is None


def _outputs(run: _Run) -> dict[str, Any]:
    listed = run.runtime.published_outputs(run.workflow_id)
    assert listed is not None
    served: dict[str, Any] = {"open": listed.open}
    for member in sorted(listed.members, key=lambda each: each.order):
        found = run.runtime.published_output(
            run.workflow_id, member.name, member.scope_id, member.key, member.sequence
        )
        assert found is not None and found.member is not None
        assert found.declaration == member.declaration
        served[member.cursor] = (
            member.publication,
            run.runtime.read_output(member).model_dump(mode="json"),
            found.open,
        )
    for name in sorted({member.name for member in listed.members} | {"nothing"}):
        named = run.runtime.published_outputs(run.workflow_id, name)
        assert named is not None
        served[f"named:{name}"] = sorted(
            (member.cursor, member.publication) for member in named.members
        )
        served[f"absent:{name}"] = run.runtime.published_output(
            run.workflow_id, name, "scp-absent", "absent", None
        )
    return served


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("nodes", "templates", "steps"),
    [
        (_CONCAT, "", [("a", {"inner": {"v": 1}, "outer": 2}), ("b", {"v": 3})]),
        (
            _LOOP,
            _LOOP_BODY,
            [
                ("seed", {"draft": "a"}),
                ("step", {"route": "again", "next": {"draft": "b"}}),
                ("step", {"route": "done", "next": {"draft": "z"}}),
            ],
        ),
        (
            _FAN_OVER_CHILD,
            _TWO_RETURNS,
            [
                ("plan", {"items": ["x", "y"]}),
                ("work", {"value": "one"}),
                ("work", {"value": "two"}),
            ],
        ),
        (_PENDING_OUTPUTS, _LOOP_BODY, [("a", {"v": 1}), ("cancel", {})]),
    ],
    ids=["merge", "loop", "spawn", "cancelled"],
)
async def test_a_freed_workflow_serves_its_published_outputs_as_it_did(
    nodes: str, templates: str, steps: list[tuple[str, dict[str, Any]]]
) -> None:
    registry = _registry()
    run = await _Run(registry=registry).start(_workflow(nodes, templates))  # type: ignore[arg-type]
    for name, result in steps:
        if name == "cancel":
            run.runtime.cancel_workflow(run.workflow_id)
        else:
            run.run(name, result)
    assert run.settled()
    served = _outputs(run)
    await run.runtime.start_settled_history()

    run.runtime.settled_history.drain()

    assert not _resident(run.runtime, run.workflow_id)
    assert _outputs(run) == served


def _query(**terms: str | list[str]) -> QueryFilter:
    return QueryFilter.parse(QueryParams(terms), _FIELDS)  # type: ignore[arg-type]


def _page(
    runtime: TaskRuntime,
    query: QueryFilter,
    limit: int,
    after: TaskOrder | None = None,
    before: TaskOrder | None = None,
    accessible: Any = None,
) -> list[str]:
    return [
        info.task_id
        for info in runtime.task_page(query, limit, after, before, accessible)
    ]


def test_a_listing_pages_freed_and_resident_tasks_as_it_did() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    freed = [_register(runtime, CHAIN), _register(runtime, LINEAR)]
    _succeed_ready(runtime)
    cancelled = _register(runtime, CHAIN)
    runtime.cancel_workflow(cancelled)
    running = _register(runtime, CHAIN)
    every = sorted(
        (task_order(record), task_id) for task_id, record in runtime._tasks.items()
    )
    orders = [order for order, _ in every]
    some = {task_id for _, task_id in every[::2]}
    queries = [
        (_query(), None),
        (_query(), some),
        (_query(workflow_id=[freed[0], running]), None),
        (_query(workflow_id=freed[1]), some),
        (_query(status=TaskStatus.DONE), None),
        (_query(status=[TaskStatus.CANCELLED, TaskStatus.DONE]), None),
        (_query(status=TaskStatus.PENDING), None),
        (_query(task_type="echo", completed="true"), None),
        (_query(workflow_id="wfl-unknown"), None),
    ]
    cursors: list[dict[str, Any]] = [
        {},
        {"after": orders[3]},
        {"before": orders[-2]},
        {"after": orders[0]},
        {"before": orders[1]},
    ]

    def pages() -> list[list[str]]:
        return [
            _page(runtime, query, limit, accessible=accessible, **cursor)
            for query, accessible in queries
            for limit in (1, 3, 100)
            for cursor in cursors
        ]

    before = pages()
    counts = runtime.task_status_counts()
    runtime.settled_history.drain()
    assert all(not _resident(runtime, w) for w in [*freed, cancelled])
    assert _resident(runtime, running)

    assert pages() == before
    assert runtime.task_status_counts() == counts


def test_a_listing_leaves_out_a_workflow_freed_after_its_moment() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    _succeed_ready(runtime)
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)
    history = runtime.settled_history
    freed_through = history.published_through
    history._published_through = freed_through - 1

    assert _page(runtime, _query(), 100) == []

    history._published_through = freed_through
    assert len(_page(runtime, _query(), 100)) == 3


def test_a_page_holds_a_sealed_workflow_once() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    _succeed_ready(runtime)
    certify = registry.certify_history
    seen: list[list[str]] = []

    def listing_while_certified(publication: Any, revision: int) -> str:
        outcome = certify(publication, revision)
        seen.append(_page(runtime, _query(), 100))
        return outcome

    setattr(registry, "certify_history", listing_while_certified)
    runtime.settled_history.drain()

    assert not _resident(runtime, workflow_id)
    assert len(seen) == 1 and len(seen[0]) == 3 == len(set(seen[0]))
    assert _page(runtime, _query(), 100) == seen[0]


def test_an_uncertified_preparation_is_never_listed() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    _succeed_ready(runtime)
    before = _page(runtime, _query(), 100)
    setattr(registry, "certify_history", lambda *args: "refused")

    runtime.settled_history.drain()

    assert _resident(runtime, workflow_id)
    assert _page(runtime, _query(), 100) == before
    assert runtime.task_status_counts()[-1] == 3


def test_a_store_outage_fails_a_freed_task_read() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    _succeed_ready(runtime)
    task_id = runtime._tasks.ids_of(workflow_id)[0]
    runtime.settled_history.drain()

    def down(*args: Any) -> Any:
        raise ConnectionError("store unavailable")

    setattr(registry, "load_history", down)
    with pytest.raises(ConnectionError):
        runtime.describe_task(task_id)
