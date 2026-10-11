"""The workflow registry and the Redis clients keep sync and async twins, and each
twin does what its counterpart does."""

import asyncio
import inspect
from typing import Any

import fakeredis
import pytest
from starlette.datastructures import QueryParams

from server.clients.redis import (
    AsyncRedisClient,
    SyncRedisClient,
    workflow_sources_key,
)
from server.orchestration.ledger_fields import (
    LedgerChanges,
    encode_ledger,
    scalar_field,
)
from server.registries.workflow import WorkflowRegistry, WorkflowSched
from server.utils.query import QueryFilter
from tests.server.redis_helpers import fake_redis_client
from tests.server.task.test_v2_orchestration import AUTORESEARCH, _bundle
from tests.server.test_workflow_listing import _Fabric, _seed

# Both clients hand out their pipeline synchronously; the caller awaits its execute.
_CLIENT_SYNC_ON_BOTH = {"control_pipeline"}


def _public(cls: type) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def test_every_registry_method_has_a_sync_and_an_async_twin() -> None:
    names = _public(WorkflowRegistry)
    asyncs = {name for name in names if name.endswith("_async")}
    syncs = names - asyncs

    assert {f"{name}_async" for name in syncs} - asyncs == set()
    assert {name.removesuffix("_async") for name in asyncs} - syncs == set()
    for name in asyncs:
        assert inspect.iscoroutinefunction(getattr(WorkflowRegistry, name)), name
    for name in syncs:
        assert not inspect.iscoroutinefunction(getattr(WorkflowRegistry, name)), name


def test_the_redis_clients_share_their_method_names() -> None:
    syncs, asyncs = _public(SyncRedisClient), _public(AsyncRedisClient)

    assert syncs == asyncs
    for name in asyncs - _CLIENT_SYNC_ON_BOTH:
        assert inspect.iscoroutinefunction(getattr(AsyncRedisClient, name)), name


def _dump(server: fakeredis.FakeServer) -> dict[str, Any]:
    """The store's contents, leaving out when each workflow was last updated."""
    # redis-py types a sync reply as possibly awaitable.
    raw: Any = fakeredis.FakeRedis(server=server, decode_responses=True)
    dump: dict[str, Any] = {}
    for key in raw.keys():
        match raw.type(key):
            case "string":
                dump[key] = raw.get(key)
            case "hash":
                dump[key] = {
                    field: value
                    for field, value in raw.hgetall(key).items()
                    if field != "updated_at"
                }
            case "set":
                dump[key] = raw.smembers(key)
            case "zset":
                dump[key] = raw.zrange(key, 0, -1, withscores=True)
    return dump


def _copy(source: fakeredis.FakeServer) -> fakeredis.FakeServer:
    target = fakeredis.FakeServer()
    reader: Any = fakeredis.FakeRedis(server=source)
    writer = fakeredis.FakeRedis(server=target)
    for key in reader.keys():
        writer.restore(key, 0, reader.dump(key))
    return target


class _Twins:
    """Two registries over identical stores, one driven sync, one async."""

    def __init__(self) -> None:
        fabric = _Fabric(0)
        asyncio.run(
            fabric.runtime.register("owner", "org", AUTORESEARCH, format="native")
        )
        _seed(fabric.registry, 6)
        self.sync = fabric.registry
        self.asyncio = WorkflowRegistry(fake_redis_client(_copy(_server(self.sync))))
        self.workflow_id = sorted(self.sync.get_workflow_ids() - set(_seed_ids(6)))[0]

    def stores(self) -> list[dict[str, Any]]:
        return [_dump(_server(registry)) for registry in (self.sync, self.asyncio)]


def _seed_ids(count: int) -> list[str]:
    return [f"wfl-{index:05d}" for index in range(count)]


def _server(registry: WorkflowRegistry) -> fakeredis.FakeServer:
    return registry._rds.sync._control.connection_pool.connection_kwargs["server"]


@pytest.fixture
def twins() -> _Twins:
    return _Twins()


def test_the_workflow_reads_agree(twins: _Twins) -> None:
    registry = twins.sync
    ids = [*_seed_ids(3), twins.workflow_id, "wfl-missing"]

    assert registry.get_workflows(ids) == asyncio.run(registry.get_workflows_async(ids))
    assert registry.get_workflow(twins.workflow_id) == asyncio.run(
        registry.get_workflow_async(twins.workflow_id)
    )
    assert registry.get_workflow("wfl-missing") is None
    assert registry.get_dynamic_task_ids(twins.workflow_id) == asyncio.run(
        registry.get_dynamic_task_ids_async(twins.workflow_id)
    )


@pytest.mark.parametrize(
    "page",
    [
        {"limit": 3},
        {"limit": 2, "after": (0, "")},
        {"limit": 2, "before": (1_700_000_004_000_000, "wfl-00004")},
        {"limit": 4, "candidates": ["wfl-00001", "wfl-00003", "wfl-00005"]},
        {
            "limit": 2,
            "query": QueryFilter.parse(QueryParams("status=DONE"), {"status"}),
        },
    ],
)
def test_the_workflow_pages_agree(twins: _Twins, page: dict[str, Any]) -> None:
    registry = twins.sync
    kwargs = {"query": QueryFilter.parse(QueryParams(""), {"status"}), **page}

    assert registry.workflow_page(**kwargs) == asyncio.run(
        registry.workflow_page_async(**kwargs)
    )
    assert registry.workflow_page(**kwargs)


def test_the_index_backfills_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    registries = [
        WorkflowRegistry(fake_redis_client(fakeredis.FakeServer())) for _ in range(2)
    ]
    for registry in registries:
        _seed(registry, 7, indexed=False)

    assert registries[0].index_submissions() == 7
    assert asyncio.run(registries[1].index_submissions_async()) == 7
    assert registries[0].index_submissions() == 0
    assert _dump(_server(registries[0])) == _dump(_server(registries[1]))


def test_the_durable_writes_agree(twins: _Twins) -> None:
    sync, async_ = twins.sync, twins.asyncio
    workflow_id = twins.workflow_id
    task_ids = sorted(sync.get_workflow_record(workflow_id).task_ids)  # type: ignore[union-attr]
    records = [
        state for state in sync.load_task_states(workflow_id, *task_ids) if state
    ]
    stored = sync.load_ledger(workflow_id)
    assert records and stored is not None
    assert asyncio.run(async_.load_ledger_async(workflow_id)) == stored
    assert records == [
        state
        for state in asyncio.run(async_.load_task_states_async(workflow_id, *task_ids))
        if state
    ]
    rewrite = LedgerChanges(encode_ledger(stored), reset=True)
    dropped = next(name for name in rewrite.fields if name.startswith("work_items:"))
    delta = LedgerChanges({scalar_field("next_seq"): "7"}, deleted=(dropped,))

    sync.save_task_states(records)
    asyncio.run(async_.save_task_states_async(records))
    for twin in (sync, async_):
        twin._rds.sync.delete(workflow_sources_key(workflow_id))
    sync.keep_sources(workflow_id, records)
    asyncio.run(async_.keep_sources_async(workflow_id, records))
    sync.commit_transition(
        workflow_id, records=records[:1], dispatched=task_ids[:1], failed=task_ids[1:2]
    )
    asyncio.run(
        async_.commit_transition_async(
            workflow_id,
            records=records[:1],
            dispatched=task_ids[:1],
            failed=task_ids[1:2],
        )
    )
    children: dict[str, Any] = {
        "retire": task_ids[:1],
        "dispatched": task_ids[:1],
        "failed": task_ids[1:2],
        "sched": WorkflowSched(in_epoch_order=True, epoch_frontier=2),
    }
    sync.commit_dynamic_tasks(workflow_id, records[:2], rewrite, **children)
    asyncio.run(
        async_.commit_dynamic_tasks_async(workflow_id, records[:2], rewrite, **children)
    )
    sync.save_ledger(workflow_id, delta)
    asyncio.run(async_.save_ledger_async(workflow_id, delta))
    sync.save_workflow_sched(workflow_id, True, 3)
    asyncio.run(async_.save_workflow_sched_async(workflow_id, True, 3))
    registration: dict[str, Any] = {
        "v2": _bundle(AUTORESEARCH, "wfl-new"),
        "ledger": rewrite,
        "submitted_at": "2026-10-08T00:00:00+00:00",
        "blueprints": records[:1],
    }
    sync.register_workflow("wfl-new", records, WorkflowSched(), **registration)
    asyncio.run(
        async_.register_workflow_async(
            "wfl-new", records, WorkflowSched(), **registration
        )
    )

    assert twins.stores()[0] == twins.stores()[1]
    blueprints = sync.load_blueprints("wfl-new")
    assert [b.record.task_id for b in blueprints] == [records[0].record.task_id]
    assert blueprints == asyncio.run(async_.load_blueprints_async("wfl-new"))
    sched = sync.load_workflow_sched(workflow_id)
    assert sched is not None and sched.epoch_frontier == 3
    assert sched == asyncio.run(async_.load_workflow_sched_async(workflow_id))

    sync.unregister_workflows(workflow_id, "wfl-00002")
    asyncio.run(async_.unregister_workflows_async(workflow_id, "wfl-00002"))
    assert twins.stores()[0] == twins.stores()[1]
    assert sync.get_workflow(workflow_id) is None


def test_unregistering_reads_off_the_sync_client(
    twins: _Twins, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a sync Redis read on the event loop")

    for name in ("hash_getall", "set_members"):
        monkeypatch.setattr(twins.asyncio._rds.sync, name, _refuse)

    asyncio.run(twins.asyncio.unregister_workflows_async(twins.workflow_id))

    assert not twins.asyncio.workflow_exists(twins.workflow_id)


def test_the_stream_range_reads_agree() -> None:
    client = fake_redis_client(fakeredis.FakeServer())
    for index in range(5):
        client.sync._telemetry.xadd("logs:task:tsk-1", {"n": str(index)})

    cases: list[dict[str, Any]] = [
        {},
        {"count": 2},
        {"max_id": "+", "min_id": "-", "count": 3},
    ]
    for kwargs in cases:
        expected = asyncio.run(
            client.asyncio.xrevrange_telemetry("logs:task:tsk-1", **kwargs)
        )
        assert client.sync.xrevrange_telemetry("logs:task:tsk-1", **kwargs) == expected
    assert [
        fields["n"] for _, fields in client.sync.xrevrange_telemetry("logs:task:tsk-1")
    ] == ["4", "3", "2", "1", "0"]


def test_the_set_writes_count_alike() -> None:
    sync = fake_redis_client(fakeredis.FakeServer()).sync
    async_ = fake_redis_client(fakeredis.FakeServer()).asyncio

    async def async_counts() -> list[int]:
        return [
            await async_.sadd("s", "a", "b"),
            await async_.sadd("s", "a"),
            await async_.sadd("s"),
            await async_.srem("s", "a", "c"),
            await async_.srem("s"),
        ]

    expected = [2, 0, 0, 1, 0]
    assert asyncio.run(async_counts()) == expected
    assert [
        sync.sadd("s", "a", "b"),
        sync.sadd("s", "a"),
        sync.sadd("s"),
        sync.srem("s", "a", "c"),
        sync.srem("s"),
    ] == expected
