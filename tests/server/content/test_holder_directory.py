"""The holder directory: records follow a registered worker's reports and leave with
it, and every method has a sync and an async twin that behave alike."""

import asyncio
from types import SimpleNamespace
from typing import Any

import fakeredis
import pytest

from server.clients.redis import (
    CONTENT_HELD_WORKERS_KEY,
    WORKERS_SET_KEY,
    content_held_key,
    content_holders_key,
)
from server.content import ContentHolderDirectory
from server.content import directory as directory_module
from tests.server.redis_helpers import fake_redis_client

_A = ("org", "a" * 64)
_B = ("org", "b" * 64)
_C = ("tsk-1:x", "c" * 64)


def _directory(*registered: str) -> tuple[ContentHolderDirectory, Any]:
    client = fake_redis_client(fakeredis.FakeServer())
    if registered:
        client.sync.sadd(WORKERS_SET_KEY, *registered)
    return ContentHolderDirectory(client), client


def _holders(directory: ContentHolderDirectory, copy: tuple[str, str]) -> list[str]:
    return [record.worker_id for record in directory.holders(*copy)]


def test_a_stale_report_is_forgotten_but_a_fresher_one_in_its_place_stays() -> None:
    directory, _ = _directory("wkr-1", "wkr-2")
    directory.apply("wkr-1", node_id="nde-1", generation=3, added=[_A])
    directory.apply("wkr-2", node_id="nde-2", generation=1, added=[_A])

    directory.forget(*_A, "wkr-1", 2)
    directory.forget(*_A, "wkr-2", 1)

    assert _holders(directory, _A) == ["wkr-1"]


def test_a_report_for_a_worker_no_longer_registered_writes_nothing() -> None:
    directory, client = _directory()

    assert not directory.apply("wkr-9", node_id="nde-1", generation=9, added=[_A])
    assert directory.holders(*_A) == []
    assert not client.sync.exists(content_held_key("wkr-9"))
    assert directory.applied_seq("wkr-9") == 0


def test_a_kept_copy_s_record_does_not_inherit_an_expiring_key() -> None:
    directory, client = _directory("wkr-1")
    key = content_holders_key(*_A)
    client.sync.hash_set(key, {"wkr-0": "{}"})
    client.sync.expire(key, 600)

    directory.apply("wkr-1", node_id="nde-1", generation=1, added=[_A])

    assert client.sync.ttl(key) == -1


def test_dropping_a_copy_removes_its_record_and_notes_the_report() -> None:
    directory, _ = _directory("wkr-1")
    directory.apply("wkr-1", node_id="nde-1", generation=1, added=[_A, _B], seq=1)

    directory.apply("wkr-1", node_id="nde-1", generation=1, dropped=[_A], seq=2)

    assert _holders(directory, _A) == []
    assert _holders(directory, _B) == ["wkr-1"]
    assert directory.applied_seq("wkr-1") == 2


def test_a_whole_report_leaves_exactly_the_copies_it_names() -> None:
    directory, _ = _directory("wkr-1")
    directory.apply("wkr-1", node_id="nde-1", generation=1, added=[_A, _B], seq=4)

    directory.replace("wkr-1", [_B, _C], node_id="nde-1", generation=1, seq=1)

    assert [_holders(directory, copy) for copy in (_A, _B, _C)] == [
        [],
        ["wkr-1"],
        ["wkr-1"],
    ]
    assert directory.applied_seq("wkr-1") == 1


def test_a_sweep_removes_every_record_a_worker_left_and_nothing_else() -> None:
    directory, client = _directory("wkr-1", "wkr-2")
    held = [("org", f"{index:064x}") for index in range(600)]
    directory.apply("wkr-1", node_id="nde-1", generation=1, added=held, seq=1)
    directory.apply("wkr-2", node_id="nde-2", generation=2, added=[held[0]], seq=1)
    client.sync.srem(WORKERS_SET_KEY, "wkr-1")

    assert directory.unregistered() == ["wkr-1"]
    assert directory.sweep("wkr-1") == 600

    assert all(_holders(directory, copy) == [] for copy in held[1:])
    assert _holders(directory, held[0]) == ["wkr-2"]
    assert not client.sync.exists(content_held_key("wkr-1"))
    assert directory.applied_seq("wkr-1") == 0
    assert client.sync.set_members(CONTENT_HELD_WORKERS_KEY) == {"wkr-2"}
    assert directory.unregistered() == []


def test_holders_are_listed_most_recently_recorded_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _ = _directory("wkr-1", "wkr-2")
    clock = iter([100.0, 200.0])
    monkeypatch.setattr(
        directory_module, "time", SimpleNamespace(time=lambda: next(clock))
    )
    directory.apply("wkr-1", node_id="nde-1", generation=1, added=[_A])
    directory.apply("wkr-2", node_id="nde-2", generation=1, added=[_A])

    assert _holders(directory, _A) == ["wkr-2", "wkr-1"]


def _scenario_sync(directory: ContentHolderDirectory, client: Any) -> list[Any]:
    out: list[Any] = [
        directory.apply("wkr-1", node_id="n", generation=1, added=[_A, _B], seq=1),
        directory.apply("wkr-2", node_id="n", generation=2, added=[_C], seq=1),
        directory.apply("wkr-1", node_id="n", generation=1, dropped=[_A], seq=2),
        directory.replace("wkr-2", [_A], node_id="n", generation=2, seq=1),
    ]
    directory.forget(*_B, "wkr-1", 1)
    out.append([(r.worker_id, r.generation) for r in directory.holders(*_A)])
    out.append([directory.applied_seq(w) for w in ("wkr-1", "wkr-2")])
    client.sync.srem(WORKERS_SET_KEY, "wkr-2")
    out.append(directory.unregistered())
    out.append(directory.sweep("wkr-2"))
    out.append(directory.holders(*_A))
    return out


async def _scenario_async(directory: ContentHolderDirectory, client: Any) -> list[Any]:
    out: list[Any] = [
        await directory.apply_async(
            "wkr-1", node_id="n", generation=1, added=[_A, _B], seq=1
        ),
        await directory.apply_async(
            "wkr-2", node_id="n", generation=2, added=[_C], seq=1
        ),
        await directory.apply_async(
            "wkr-1", node_id="n", generation=1, dropped=[_A], seq=2
        ),
        await directory.replace_async("wkr-2", [_A], node_id="n", generation=2, seq=1),
    ]
    await directory.forget_async(*_B, "wkr-1", 1)
    out.append(
        [(r.worker_id, r.generation) for r in await directory.holders_async(*_A)]
    )
    out.append([await directory.applied_seq_async(w) for w in ("wkr-1", "wkr-2")])
    await client.asyncio.srem(WORKERS_SET_KEY, "wkr-2")
    out.append(await directory.unregistered_async())
    out.append(await directory.sweep_async("wkr-2"))
    out.append(await directory.holders_async(*_A))
    return out


def test_the_sync_and_async_twins_behave_alike() -> None:
    sync_directory, sync_client = _directory("wkr-1", "wkr-2")
    async_directory, async_client = _directory("wkr-1", "wkr-2")

    expected = _scenario_sync(sync_directory, sync_client)

    assert asyncio.run(_scenario_async(async_directory, async_client)) == expected
    assert expected[4] == [("wkr-2", 2)]
    assert expected[5] == [2, 1]
    assert expected[6] == ["wkr-2"]
    assert expected[7] == 1
    assert expected[8] == []
