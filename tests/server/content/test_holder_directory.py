"""Forgetting a stale holder report never takes a fresher one with it.

A request that finds a holder's report stale forgets it, and a restarted holder can
re-report under the same worker in between; only the stale incarnation's report goes.
"""

import asyncio

import fakeredis

from server.content import ContentHolderDirectory
from tests.server.redis_helpers import fake_redis_client


def test_a_stale_report_is_forgotten_but_a_fresher_one_in_its_place_stays() -> None:
    async def run() -> list[tuple[str, int]]:
        directory = ContentHolderDirectory(
            fake_redis_client(fakeredis.FakeServer()), record_ttl_sec=300.0
        )
        await directory.record_many(
            [("org", "d" * 64)], worker_id="wkr-1", node_id="nde-1", generation=3
        )
        await directory.record_many(
            [("org", "d" * 64)], worker_id="wkr-2", node_id="nde-2", generation=1
        )
        # wkr-1 restarted and re-reported before the stale incarnation is forgotten.
        await directory.record_many(
            [("org", "d" * 64)], worker_id="wkr-1", node_id="nde-1", generation=4
        )
        await directory.forget("org", "d" * 64, "wkr-1", 3)
        await directory.forget("org", "d" * 64, "wkr-2", 1)
        return [
            (record.worker_id, record.generation)
            for record in await directory.holders("org", "d" * 64)
        ]

    assert asyncio.run(run()) == [("wkr-1", 4)]
