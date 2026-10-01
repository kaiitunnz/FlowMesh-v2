"""A node's reverse-relay namespaces: one worker bridge and attachment per keyspace."""

import asyncio
import logging
from dataclasses import dataclass

from shared.network.relay_frame import (
    CONTENT_FRAME_KIND,
    RESIDENT_FRAME_KIND,
    SSH_FRAME_KIND,
)

from ...network.reverse_relay import (
    CONTENT_RELAY_KEYSPACE,
    RESIDENT_RELAY_KEYSPACE,
    SSH_RELAY_KEYSPACE,
    BinaryRedis,
    RelayKeyspace,
)
from ...network.worker_bridge import LocalEnqueue, RelayWorkerBridge
from .reverse_relay_attachment import ReverseRelayAttachment


@dataclass(frozen=True)
class RelayNamespace:
    """One relay namespace as a node carries it.

    A worker pushes the namespace's frames up as ``event_type`` events and receives
    them down as ``frame_kind`` mediated ops.
    """

    event_type: str
    frame_kind: str
    keyspace: RelayKeyspace
    owner_suffix: str


RESIDENT_NAMESPACE = RelayNamespace(
    "RESIDENT_FRAME", RESIDENT_FRAME_KIND, RESIDENT_RELAY_KEYSPACE, ""
)
CONTENT_NAMESPACE = RelayNamespace(
    "CONTENT_FRAME", CONTENT_FRAME_KIND, CONTENT_RELAY_KEYSPACE, ":content"
)
SSH_NAMESPACE = RelayNamespace("SSH_FRAME", SSH_FRAME_KIND, SSH_RELAY_KEYSPACE, ":ssh")


@dataclass(frozen=True)
class NodeRelay:
    bridge: RelayWorkerBridge
    attachment: ReverseRelayAttachment


class NodeRelays:
    """The bridge and attachment of each namespace this node carries."""

    def __init__(
        self,
        redis: BinaryRedis,
        node_id: str,
        enqueue_local: LocalEnqueue,
        namespaces: list[RelayNamespace],
        *,
        owner: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._relays: dict[RelayNamespace, NodeRelay] = {}
        for namespace in namespaces:
            bridge = RelayWorkerBridge(
                redis,
                node_id,
                enqueue_local,
                keyspace=namespace.keyspace,
                frame_kind=namespace.frame_kind,
                logger=logger,
            )
            attachment = ReverseRelayAttachment(
                redis,
                node_id,
                bridge,
                owner=f"{owner}{namespace.owner_suffix}",
                keyspace=namespace.keyspace,
                logger=logger,
            )
            self._relays[namespace] = NodeRelay(bridge, attachment)

    def bridge(self, namespace: RelayNamespace) -> RelayWorkerBridge | None:
        relay = self._relays.get(namespace)
        return None if relay is None else relay.bridge

    def bridges_by_event_type(self) -> dict[str, RelayWorkerBridge]:
        return {
            namespace.event_type: relay.bridge
            for namespace, relay in self._relays.items()
        }

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        for relay in self._relays.values():
            relay.attachment.start(loop)

    async def stop(self) -> None:
        for relay in reversed(self._relays.values()):
            await relay.attachment.stop()

    def rebind(self, node_id: str) -> None:
        """Carry every namespace under the node's new id; callable from any thread."""
        for relay in self._relays.values():
            relay.bridge.rebind(node_id)
            relay.attachment.rebind(node_id)


__all__ = [
    "CONTENT_NAMESPACE",
    "RESIDENT_NAMESPACE",
    "SSH_NAMESPACE",
    "NodeRelays",
    "RelayNamespace",
]
