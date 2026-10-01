"""A node's relay namespaces follow the node when it re-registers under a new id."""

import asyncio
from typing import Any

import pytest

from server.network.reverse_relay import (
    RelayDirection,
    RelayFrameKind,
    RelayLease,
    RelaySessionStore,
    RelayStreamStore,
)
from server.supervisor.services.node_relays import (
    CONTENT_NAMESPACE,
    RESIDENT_NAMESPACE,
    SSH_NAMESPACE,
    NodeRelays,
    RelayNamespace,
)
from tests.server.network._relay_fakes import FakeBinaryRedis, relay_frame


@pytest.mark.parametrize(
    "namespace",
    [RESIDENT_NAMESPACE, CONTENT_NAMESPACE, SSH_NAMESPACE],
    ids=lambda namespace: namespace.event_type,
)
def test_a_re_registered_node_carries_each_namespace_under_its_new_id(
    namespace: RelayNamespace,
) -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        streams = RelayStreamStore(redis, namespace.keyspace)
        delivered: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def enqueue_local(worker_id: str, payload: dict[str, Any]) -> bool:
            await delivered.put(payload)
            return True

        relays = NodeRelays(
            redis, "nde-old", enqueue_local, [namespace], owner="supervisor"
        )
        relays.start(asyncio.get_running_loop())
        try:
            relays.rebind("nde-new")
            await RelaySessionStore(redis, namespace.keyspace).update(
                "rly-1",
                origin_node="ssh-edge",
                target_node="nde-new",
                target_worker="wkr-1",
            )
            await streams.publish_down(
                "nde-new",
                relay_frame(
                    RelayFrameKind.DATA,
                    direction=RelayDirection.ORIGIN_TO_TARGET,
                    seq=1,
                ),
            )
            payload = await asyncio.wait_for(delivered.get(), 5)
            assert payload["frame_kind"] == namespace.frame_kind

            bridge = relays.bridges_by_event_type()[namespace.event_type]
            await bridge.publish_up(relay_frame(RelayFrameKind.DATA, seq=1))
            new_up, _ = await streams.read_up("nde-new", "0", count=10, block_ms=None)
            old_up, _ = await streams.read_up("nde-old", "0", count=10, block_ms=None)
            assert (len(new_up), len(old_up)) == (1, 0)
        finally:
            await relays.stop()

    asyncio.run(run())


def test_a_re_registered_node_hands_its_lease_to_its_new_id() -> None:
    """The old id's lease is released, so whatever holds that id next can consume
    it, and the new id's lease is taken."""

    async def run() -> None:
        namespace = SSH_NAMESPACE
        redis = FakeBinaryRedis()
        lease = RelayLease(redis, keyspace=namespace.keyspace)
        owner = f"supervisor{namespace.owner_suffix}"
        delivered: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def enqueue_local(worker_id: str, payload: dict[str, Any]) -> bool:
            await delivered.put(payload)
            return True

        relays = NodeRelays(
            redis, "nde-old", enqueue_local, [namespace], owner="supervisor"
        )
        relays.start(asyncio.get_running_loop())
        try:
            async with asyncio.timeout(5):
                while not await lease.owns("nde-old", "down", owner):
                    await asyncio.sleep(0.01)

            relays.rebind("nde-new")
            await RelaySessionStore(redis, namespace.keyspace).update(
                "rly-1",
                origin_node="ssh-edge",
                target_node="nde-new",
                target_worker="wkr-1",
            )
            await RelayStreamStore(redis, namespace.keyspace).publish_down(
                "nde-new",
                relay_frame(
                    RelayFrameKind.DATA,
                    direction=RelayDirection.ORIGIN_TO_TARGET,
                    seq=1,
                ),
            )
            await asyncio.wait_for(delivered.get(), 5)

            assert not await lease.owns("nde-old", "down", owner)
            assert await lease.acquire("nde-old", "down", "next-holder")
            assert await lease.owns("nde-new", "down", owner)
        finally:
            await relays.stop()

    asyncio.run(run())
