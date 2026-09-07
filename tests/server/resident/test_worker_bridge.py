"""The node-local resident bridge forwards frames opaquely by session and direction.

A down frame toward an origin session goes to the origin worker; one toward a replica
session goes to the replica worker; a worker's produced frame publishes to the up
stream. The bridge never decodes the resident wire body it carries.
"""

import asyncio
from typing import Any

from server.network.reverse_relay import RelaySessionStore, RelayStreamStore
from server.resident.worker_bridge import ResidentWorkerBridge
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from tests.server.network._relay_fakes import FakeBinaryRedis


def _frame(direction: RelayDirection) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="s1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=direction,
        seq=1,
        payload=b'{"kind": "chunk", "data": "opaque"}',
    )


def test_forwards_by_direction_to_the_named_local_worker() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        await RelaySessionStore(redis).update(
            "s1",
            origin_node="nde-o",
            target_node="nde-t",
            origin_worker="wrk-origin",
            target_worker="wrk-replica",
        )
        sent: list[tuple[str, dict[str, Any]]] = []

        async def enqueue(worker_id: str, payload: dict[str, Any]) -> bool:
            sent.append((worker_id, payload))
            return True

        bridge = ResidentWorkerBridge(redis, "nde-t", enqueue)

        await bridge.on_frame(_frame(RelayDirection.ORIGIN_TO_TARGET))
        await bridge.on_frame(_frame(RelayDirection.TARGET_TO_ORIGIN))

        assert [w for w, _ in sent] == ["wrk-replica", "wrk-origin"]
        for _worker, payload in sent:
            assert payload["kind"] == "mediated_op"
            assert payload["frame_kind"] == "resident_frame"
            # The bridge carried the frame opaquely: the wire payload round-trips whole.
            assert (
                RelayFrame.from_wire(payload["payload"]).payload
                == _frame(RelayDirection.TARGET_TO_ORIGIN).payload
            )

    asyncio.run(run())


def test_publish_up_writes_the_node_up_stream() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()

        async def enqueue(_worker: str, _payload: dict[str, Any]) -> bool:
            return True

        bridge = ResidentWorkerBridge(redis, "nde-o", enqueue)
        await bridge.publish_up(_frame(RelayDirection.TARGET_TO_ORIGIN))
        entries, _ = await RelayStreamStore(redis).read_up("nde-o", "0", 10, None)
        assert len(entries) == 1
        assert entries[0].frame.session_id == "s1"

    asyncio.run(run())


def test_unknown_session_is_dropped() -> None:
    async def run() -> None:
        redis = FakeBinaryRedis()
        sent: list[str] = []

        async def enqueue(worker_id: str, _payload: dict[str, Any]) -> bool:
            sent.append(worker_id)
            return True

        bridge = ResidentWorkerBridge(redis, "nde-t", enqueue)
        await bridge.on_frame(_frame(RelayDirection.ORIGIN_TO_TARGET))
        assert sent == []

    asyncio.run(run())
