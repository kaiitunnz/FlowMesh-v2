"""The holder directory: forgetting a stale report never takes a fresher one with it,
and every method has a sync and an async twin.

A request that finds a holder's report stale forgets it, and a restarted holder can
re-report under the same worker in between; only the stale incarnation's report goes.
"""

import asyncio

import fakeredis

from server.content import ContentHolderDirectory
from server.content import directory as directory_module
from tests.server.redis_helpers import fake_redis_client


def _directory() -> ContentHolderDirectory:
    return ContentHolderDirectory(
        fake_redis_client(fakeredis.FakeServer()), record_ttl_sec=300.0
    )


def test_a_stale_report_is_forgotten_but_a_fresher_one_in_its_place_stays() -> None:
    async def run() -> list[tuple[str, int]]:
        directory = _directory()
        await directory.record_many_async(
            [("org", "d" * 64)], worker_id="wkr-1", node_id="nde-1", generation=3
        )
        await directory.record_many_async(
            [("org", "d" * 64)], worker_id="wkr-2", node_id="nde-2", generation=1
        )
        # wkr-1 restarted and re-reported before the stale incarnation is forgotten.
        await directory.record_many_async(
            [("org", "d" * 64)], worker_id="wkr-1", node_id="nde-1", generation=4
        )
        await directory.forget_async("org", "d" * 64, "wkr-1", 3)
        await directory.forget_async("org", "d" * 64, "wkr-2", 1)
        return [
            (record.worker_id, record.generation)
            for record in await directory.holders_async("org", "d" * 64)
        ]

    assert asyncio.run(run()) == [("wkr-1", 4)]


def test_the_sync_twins_record_and_forget_alike() -> None:
    directory = _directory()
    directory.record_many(
        [("org", "d" * 64)], worker_id="wkr-1", node_id="nde-1", generation=3
    )
    directory.record("org", "d" * 64, worker_id="wkr-2", node_id="nde-2", generation=1)
    directory.record_many(
        [("org", "d" * 64)], worker_id="wkr-1", node_id="nde-1", generation=4
    )
    directory.forget("org", "d" * 64, "wkr-1", 3)

    assert sorted(
        (record.worker_id, record.generation)
        for record in directory.holders("org", "d" * 64)
    ) == [("wkr-1", 4), ("wkr-2", 1)]
    assert directory.holders("org", "d" * 64) == asyncio.run(
        directory.holders_async("org", "d" * 64)
    )


def test_a_report_larger_than_one_round_trip_records_every_object() -> None:
    async def run() -> list[int]:
        directory = _directory()
        held = [
            ("org", f"{i:064x}") for i in range(directory_module._RECORD_CHUNK * 2 + 1)
        ]
        await directory.record_many_async(
            held, worker_id="wkr-1", node_id="nde-1", generation=1
        )
        return [
            len(await directory.holders_async(scope, digest)) for scope, digest in held
        ]

    assert set(asyncio.run(run())) == {1}


def test_every_public_method_has_its_twin() -> None:
    public = {
        name
        for name in vars(ContentHolderDirectory)
        if not name.startswith("_") and callable(getattr(ContentHolderDirectory, name))
    }
    sync = {name for name in public if not name.endswith("_async")}

    assert {f"{name}_async" for name in sync} == public - sync
