"""A node registered through the alias-lease script reads back as it registered."""

import logging

import fakeredis

from server.network.state import ReachabilityClass
from server.registries.node import NodeRegistry
from shared.schemas.network import NetworkEndpointAdvertisement
from tests.server.redis_helpers import fake_redis_client
from tests.server.registries.test_node_alias_lease import TTL_SEC, _info


def test_a_registered_node_reads_back_with_its_tags_and_endpoint() -> None:
    registry = NodeRegistry(
        fake_redis_client(fakeredis.FakeServer()),
        logging.getLogger("test.node_record_roundtrip"),
        TTL_SEC,
    )
    endpoint = NetworkEndpointAdvertisement(
        endpoint_id="ep-1",
        url="127.0.0.1:9001",
        peer_url="127.0.0.1:9101",
        generation=3,
        trust_domain="fm",
        reachability_class=ReachabilityClass.ROUTABLE,
        relay_attachment_id="att-1",
    )
    info = _info().model_copy(
        update={"tags": ["gpu", "fast"], "network_endpoint": endpoint}
    )

    node = registry.get_node(registry.register_node(info))

    assert node is not None
    assert node.tags == ["gpu", "fast"]
    assert node.network_endpoint == endpoint
