"""The node's target-leg listener: who it serves, and how a session reaches a worker."""

import asyncio
import ssl
from typing import Any

import pytest

from server.network.reverse_relay import RelaySessionStore
from server.resident.worker_bridge import ResidentWorkerBridge
from server.supervisor.services.target_leg_listener import NodeTargetLegListener
from shared.network.frame_stream import read_relay_frame, write_relay_frame
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from tests.server.network._relay_fakes import FakeBinaryRedis
from tests.support.certs import new_ca

_ROOT = "root.flowmesh"
_CA = new_ca()


def _material(identity: str) -> MutualTlsMaterial:
    issued = _CA.issue(identity)
    return MutualTlsMaterial.from_b64(
        ca_b64=_CA.ca_b64,
        cert_b64=issued.cert_b64,
        key_b64=issued.key_b64,
        root_identity=_ROOT,
    )


def _frame(
    direction: RelayDirection = RelayDirection.ORIGIN_TO_TARGET,
    payload: bytes = b"opaque",
) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=direction,
        seq=1,
        payload=payload,
    )


class _Node:
    """A node listener over a bridge whose local enqueues and streams are recorded."""

    def __init__(self) -> None:
        self.redis = FakeBinaryRedis()
        self.enqueued: list[tuple[str, dict[str, Any]]] = []
        self.bridge = ResidentWorkerBridge(self.redis, "nde-target", self._enqueue)
        self.listener = NodeTargetLegListener(
            endpoint="127.0.0.1:0",
            material=_material("node.flowmesh"),
            bridge=self.bridge,
        )
        self.port = 0

    async def _enqueue(self, worker_id: str, payload: dict[str, Any]) -> bool:
        self.enqueued.append((worker_id, payload))
        return True

    async def start(self) -> None:
        await RelaySessionStore(self.redis).update(
            "rly-1",
            origin_node="nde-origin",
            target_node="nde-target",
            origin_worker="wrk-origin",
            target_worker="wrk-replica",
        )
        await self.listener.start()
        self.port = self.listener.port

    async def await_enqueue(self) -> None:
        for _ in range(100):
            if self.enqueued:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the listener never reached the local worker")


def test_the_root_reaches_the_local_worker_and_its_reply_returns_on_the_socket() -> (
    None
):
    async def scenario() -> tuple[_Node, RelayFrame]:
        node = _Node()
        await node.start()
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", node.port, ssl=client_context(_material(_ROOT))
        )
        await write_relay_frame(writer, _frame())
        await node.await_enqueue()
        # The worker's produced frame answers on the connection the root opened.
        await node.bridge.publish_up(_frame(RelayDirection.TARGET_TO_ORIGIN, b"reply"))
        answer = await asyncio.wait_for(read_relay_frame(reader), timeout=5)
        writer.close()
        await node.listener.stop()
        return node, answer

    node, answer = asyncio.run(scenario())
    assert [worker for worker, _ in node.enqueued] == ["wrk-replica"]
    assert node.enqueued[0][1]["frame_kind"] == "resident_frame"
    assert answer.payload == b"reply"
    # Nothing about this session touched the node's reverse-relay up stream.
    assert not node.redis.streams


def test_a_released_session_returns_to_the_up_stream() -> None:
    async def scenario() -> _Node:
        node = _Node()
        await node.start()
        _, writer = await asyncio.open_connection(
            "127.0.0.1", node.port, ssl=client_context(_material(_ROOT))
        )
        await write_relay_frame(writer, _frame())
        await node.await_enqueue()
        writer.close()
        await asyncio.sleep(0.1)
        await node.bridge.publish_up(_frame(RelayDirection.TARGET_TO_ORIGIN, b"reply"))
        await node.listener.stop()
        return node

    node = asyncio.run(scenario())
    assert "rr:node:nde-target:up" in node.redis.streams


def test_a_ca_signed_dialer_that_is_not_the_root_reaches_no_worker() -> None:
    async def scenario() -> _Node:
        node = _Node()
        await node.start()
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", node.port, ssl=client_context(_material("impostor.flowmesh"))
        )
        await write_relay_frame(writer, _frame())
        with pytest.raises(
            (asyncio.IncompleteReadError, ConnectionError, ssl.SSLError, OSError)
        ):
            await asyncio.wait_for(read_relay_frame(reader), timeout=5)
        writer.close()
        await node.listener.stop()
        return node

    assert asyncio.run(scenario()).enqueued == []
