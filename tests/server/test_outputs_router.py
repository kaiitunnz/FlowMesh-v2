"""A workflow's published outputs are listed and fetched by the names authored."""

import logging
from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi import HTTPException, status
from lumid_hooks import PrincipalContext, ResourceRef

from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import outputs as outputs_router
from server.schemas.outputs import OutputOutcome, WorkflowOutputPage
from server.task.runtime import TaskRuntime
from shared.content import ContentReference, ContentUnavailable
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import result_payload
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _live_runtime,
    _planned,
    _pop_ready,
    _register,
    _worker,
)

_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: published}
spec:
  graph:
    nodes:
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: reviewer
        spec: {taskType: echo, data: {type: list, items: [tmpl]}}
      - name: fanout
        dependsOn: [planner]
        region: {kind: spawn, child: reviewer, result: {visibility: published}}
      - name: collect
        dependsOn: [fanout]
        region: {kind: join, completion: all_settled}
      - name: summarize
        dependsOn: [collect]
        spec:
          taskType: echo
          data: {type: list, items: [s]}
          v2: {result: {visibility: published}}
      - name: internal
        spec: {taskType: echo, data: {type: list, items: [i]}}
"""

_LOGGER = logging.getLogger("test.outputs_router")


def _principal(principal_id: str = "p-1", org_id: str = "org") -> PrincipalContext:
    return PrincipalContext(
        principal_id=principal_id,
        org_id=org_id,
        external_id="ext",
        principal_type="user",
        scopes=[],
    )


class _Workflow:
    """A submitted workflow whose spawn settles 12 children: 10 with values, one
    skipped, and one failed."""

    def __init__(self, runtime: TaskRuntime, workflow_id: str, ids: dict[str, str]):
        self.runtime = runtime
        self.workflow_id = workflow_id
        self.ids = ids


async def _workflow() -> _Workflow:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _WF)
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, [f"h{i}" for i in range(12)]), _TS
    )
    children = [task for task in _pop_ready(runtime) if task.startswith("act-")]
    by_index = {}
    for child in children:
        element = runtime.input_element(child)
        assert element is not None and element.element is not None
        by_index[element.element] = child
    for index, child in sorted(by_index.items()):
        if index == 10:
            runtime.mark_succeeded(child, None, {}, _TS, skip={"skipped": True})
        elif index == 11:
            record_dispatch(runtime, child, cast(Any, _worker()))
            runtime.fail_dispatch(child, "wkr-1", {}, _TS, error="x", retryable=False)
        else:
            record_dispatch(runtime, child, cast(Any, _worker()))
            runtime.mark_succeeded(
                child,
                "wkr-1",
                result_payload(
                    runtime._results,
                    child,
                    {"items": [f"reviewed-{index}"]},
                    runtime._tasks[child].org_id,
                ),
                _TS,
            )
    return _Workflow(runtime, workflow_id, ids)


async def _list(wf: _Workflow, **params: Any) -> WorkflowOutputPage:
    defaults: dict[str, Any] = dict(
        limit=100, before=None, after=None, output=None, scope=None
    )
    return await outputs_router.list_outputs(
        wf.workflow_id,
        principal=params.pop("principal", _principal()),
        runtime=wf.runtime,
        logger=_LOGGER,
        **{**defaults, **params},
    )


async def _get(wf: _Workflow, name: str, **params: Any) -> Any:
    defaults: dict[str, Any] = dict(scope=None, key=None, sequence=None)
    return await outputs_router.get_output(
        wf.workflow_id,
        name,
        principal=params.pop("principal", _principal()),
        runtime=wf.runtime,
        logger=_LOGGER,
        **{**defaults, **params},
    )


async def _status(call: Any) -> tuple[int, str]:
    with pytest.raises(HTTPException) as caught:
        await call
    detail = caught.value.detail
    return caught.value.status_code, (
        detail["code"] if isinstance(detail, dict) else str(detail)
    )


@pytest.mark.anyio
async def test_published_outputs_are_listed_by_their_authored_names() -> None:
    wf = await _workflow()

    listed = await _list(wf)

    names = [entry.name for entry in listed.entries]
    assert set(names) == {"fanout", "summarize"}
    assert names.count("fanout") == 12
    fanout = [e for e in listed.entries if e.name == "fanout"]
    assert [e.key for e in fanout] == [str(i) for i in range(12)]
    assert {e.outcome for e in fanout[:10]} == {OutputOutcome.SUCCESS}
    assert fanout[10].outcome is OutputOutcome.EXPLICIT_EMPTY
    assert fanout[11].outcome is OutputOutcome.DECLARED_FAILURE
    assert all(e.cardinality == "keyed_collection" for e in fanout)
    assert all(e.value_type == "echo" and e.scope for e in fanout)
    (summarize,) = [e for e in listed.entries if e.name == "summarize"]
    assert summarize.outcome is OutputOutcome.PENDING
    assert summarize.cardinality == "singleton" and summarize.scope is None
    assert listed.open


@pytest.mark.anyio
async def test_a_collection_pages_by_cursor_in_key_order() -> None:
    wf = await _workflow()
    everything = [e.cursor for e in (await _list(wf, output="fanout")).entries]

    paged: list[str] = []
    cursor: str | None = None
    while True:
        batch = await _list(wf, output="fanout", limit=5, after=cursor)
        if not batch.entries:
            break
        paged.extend(entry.cursor for entry in batch.entries)
        cursor = batch.next_cursor
    assert paged == everything

    back = await _list(wf, output="fanout", limit=3, before=everything[6])
    assert [e.cursor for e in back.entries] == everything[3:6]


@pytest.mark.anyio
async def test_a_cursor_holds_its_place_as_members_settle() -> None:
    wf = await _workflow()
    first = await _list(wf, limit=4)
    rest_before = await _list(wf, after=first.next_cursor)

    summarize = wf.ids["summarize"]
    record_dispatch(wf.runtime, summarize, cast(Any, _worker()))
    wf.runtime.mark_succeeded(
        summarize,
        "wkr-1",
        result_payload(
            wf.runtime._results,
            summarize,
            {"items": ["summary"]},
            wf.runtime._tasks[summarize].org_id,
        ),
        _TS,
    )

    rest_after = await _list(wf, after=first.next_cursor)
    assert [e.cursor for e in rest_after.entries] == [
        e.cursor for e in rest_before.entries
    ]
    assert rest_after.entries[-1].outcome is OutputOutcome.SUCCESS
    assert rest_after.open
    internal = wf.ids["internal"]
    record_dispatch(wf.runtime, internal, cast(Any, _worker()))
    wf.runtime.mark_succeeded(internal, None, {}, _TS, skip={"skipped": True})
    assert not (await _list(wf)).open
    fetched = await _get(wf, "summarize")
    assert fetched.value.model_dump(mode="json")["items"] == ["summary"]


@pytest.mark.anyio
async def test_a_scope_filter_selects_its_members() -> None:
    wf = await _workflow()
    scope = (await _list(wf, output="fanout")).entries[0].scope
    assert scope is not None
    assert len((await _list(wf, scope=scope)).entries) == 12
    assert (await _list(wf, scope="scp-other")).entries == []


@pytest.mark.anyio
async def test_a_member_value_is_the_declared_projection() -> None:
    wf = await _workflow()
    scope = (await _list(wf, output="fanout")).entries[0].scope

    fetched = await _get(wf, "fanout", scope=scope, key="3")

    assert fetched.outcome is OutputOutcome.SUCCESS
    assert fetched.name == "fanout" and fetched.key == "3"
    assert fetched.value is not None
    assert fetched.value.model_dump(mode="json")["items"] == ["reviewed-3"]
    assert "task_id" not in fetched.model_dump(mode="json")


@pytest.mark.anyio
async def test_empty_and_failed_members_are_typed_outcomes() -> None:
    wf = await _workflow()
    scope = (await _list(wf, output="fanout")).entries[0].scope

    empty = await _get(wf, "fanout", scope=scope, key="10")
    failed = await _get(wf, "fanout", scope=scope, key="11")

    assert empty.outcome is OutputOutcome.EXPLICIT_EMPTY and empty.value is None
    assert failed.outcome is OutputOutcome.DECLARED_FAILURE and failed.value is None


@pytest.mark.anyio
async def test_a_pending_output_is_a_conflict() -> None:
    wf = await _workflow()
    assert await _status(_get(wf, "summarize")) == (
        status.HTTP_409_CONFLICT,
        "output_pending",
    )
    scope = (await _list(wf, output="fanout")).entries[0].scope
    assert await _status(_get(wf, "fanout", scope=scope, key="99")) == (
        status.HTTP_409_CONFLICT,
        "output_pending",
    )


@pytest.mark.anyio
async def test_a_collection_member_needs_its_scope_and_key() -> None:
    wf = await _workflow()
    assert (await _status(_get(wf, "fanout", key="0")))[0] == 400
    assert (await _status(_get(wf, "fanout")))[0] == 400


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["internal", "reviewer", "nothing", "legacy:x"])
async def test_an_unknown_or_unpublished_output_is_not_found(name: str) -> None:
    wf = await _workflow()
    assert await _status(_get(wf, name)) == (404, "output_not_found")


@pytest.mark.anyio
async def test_an_unreachable_store_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wf = await _workflow()
    scope = (await _list(wf, output="fanout")).entries[0].scope

    def _away(reference: ContentReference) -> bytes:
        raise ContentUnavailable("store down")

    wf.runtime._results._cache.clear()
    monkeypatch.setattr(wf.runtime._results._store, "fetch", _away)
    assert await _status(_get(wf, "fanout", scope=scope, key="0")) == (
        503,
        "content_unavailable",
    )


@pytest.mark.anyio
async def test_corrupt_bound_content_is_an_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wf = await _workflow()
    scope = (await _list(wf, output="fanout")).entries[0].scope
    wf.runtime._results._cache.clear()
    monkeypatch.setattr(
        wf.runtime._results._store, "fetch", lambda reference: b"not the bytes"
    )
    assert await _status(_get(wf, "fanout", scope=scope, key="0")) == (
        500,
        "output_unreadable",
    )


@pytest.mark.anyio
async def test_another_org_learns_nothing() -> None:
    wf = await _workflow()
    outsider = _principal("p-9", "other-org")
    assert await _status(_get(wf, "fanout", principal=outsider)) == (
        404,
        "output_not_found",
    )
    assert await _status(_list(wf, principal=outsider)) == (404, "output_not_found")


class _DenyingChecker:
    """Denies the named principal one resource kind, before anything is looked up."""

    name = "deny"

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.checked: list[tuple[str, str | None]] = []

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        self.checked.append((resource.kind, resource.id))
        if principal.principal_id == "p-2" and resource.kind == self.kind:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="denied")

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return None


@pytest.fixture(params=["workflow", "result"])
def denying(request: pytest.FixtureRequest) -> Iterator[_DenyingChecker]:
    checker = _DenyingChecker(request.param)
    PERMISSION_CHECKERS.append(checker)
    try:
        yield checker
    finally:
        PERMISSION_CHECKERS.clear()


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["fanout", "summarize", "nothing"])
async def test_a_forbidden_caller_learns_nothing(
    denying: _DenyingChecker, name: str
) -> None:
    wf = await _workflow()
    reads: list[str] = []
    published = wf.runtime.published_outputs

    def _recorded(workflow_id: str) -> Any:
        reads.append(workflow_id)
        return published(workflow_id)

    wf.runtime.published_outputs = _recorded  # type: ignore[method-assign]
    denied = _principal("p-2")

    assert (await _status(_get(wf, name, principal=denied)))[0] == 403
    assert (await _status(_list(wf, principal=denied)))[0] == 403
    assert reads == []


@pytest.mark.anyio
async def test_both_checks_run_for_an_allowed_caller(
    denying: _DenyingChecker,
) -> None:
    wf = await _workflow()
    await _list(wf)
    assert ("workflow", wf.workflow_id) in denying.checked
    assert ("result", None) in denying.checked
