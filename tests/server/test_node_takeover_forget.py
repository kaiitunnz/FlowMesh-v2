"""A node record a takeover removes is forgotten by the network plane, as a node that
stopped is."""

import json
import logging
from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis

from server.clients.redis import NODE_EVENT_CHANNEL
from server.config import NetworkPlaneConfig
from server.network.service import NetworkPlane
from server.registries.node import NodeRegistry
from server.services.monitoring import EventMonitor
from shared.schemas.event import NodeEvent, parse_event
from tests.server.redis_helpers import fake_redis_client
from tests.server.registries.test_node_alias_lease import LEASE, TTL_SEC, _info

_LOGGER = logging.getLogger("test.node_takeover_forget")


def test_a_taken_over_node_is_forgotten_by_the_network_plane() -> None:
    server = fakeredis.FakeServer()
    rds = fakeredis.FakeRedis(server=server, decode_responses=True)
    registry = NodeRegistry(fake_redis_client(server), _LOGGER, TTL_SEC)
    plane = NetworkPlane(NetworkPlaneConfig(enabled=True), registry, _LOGGER)
    monitor = EventMonitor(
        redis_client=MagicMock(),
        logger=_LOGGER,
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=registry,
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        on_node_removed=plane.forget_node,
    )
    old_id = registry.register_node(_info())
    plane._seen_generation[old_id] = 1
    rds.delete(LEASE)
    pubsub = rds.pubsub()
    pubsub.subscribe(NODE_EVENT_CHANNEL)
    pubsub.get_message(timeout=1)

    registry.register_node(_info())
    message = pubsub.get_message(timeout=1)
    assert message is not None
    event = parse_event(json.loads(message["data"]))
    assert isinstance(event, NodeEvent)
    monitor._handle_node_event(cast(Any, event))

    assert old_id not in plane._seen_generation
