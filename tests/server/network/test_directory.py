"""The network endpoint directory: latest-generation-wins and node lookup."""

from server.network import NetworkEndpointDirectory
from server.network.state import NetworkEndpointAdvertisement, ReachabilityClass


def _adv(
    endpoint_id: str, *, node_id: str, generation: int
) -> NetworkEndpointAdvertisement:
    return NetworkEndpointAdvertisement(
        endpoint_id=endpoint_id,
        node_id=node_id,
        url=endpoint_id,
        generation=generation,
        trust_domain="fm",
        reachability_class=ReachabilityClass.ROUTABLE,
    )


def test_upsert_new_and_advance() -> None:
    directory = NetworkEndpointDirectory()
    assert directory.upsert(_adv("e1", node_id="nde-1", generation=1)) is True
    assert directory.upsert(_adv("e1", node_id="nde-1", generation=2)) is True
    latest = directory.get("e1")
    assert latest is not None and latest.generation == 2


def test_generation_regression_rejected() -> None:
    directory = NetworkEndpointDirectory()
    directory.upsert(_adv("e1", node_id="nde-1", generation=5))
    assert directory.upsert(_adv("e1", node_id="nde-1", generation=5)) is False
    assert directory.upsert(_adv("e1", node_id="nde-1", generation=3)) is False
    latest = directory.get("e1")
    assert latest is not None and latest.generation == 5


def test_lookup_by_node_and_remove() -> None:
    directory = NetworkEndpointDirectory()
    directory.upsert(_adv("e1", node_id="nde-1", generation=1))
    by_node = directory.by_node("nde-1")
    assert by_node is not None and by_node.endpoint_id == "e1"
    directory.remove_node("nde-1")
    assert directory.by_node("nde-1") is None


def test_rebuild_replaces_index() -> None:
    directory = NetworkEndpointDirectory()
    directory.upsert(_adv("stale", node_id="nde-9", generation=1))
    directory.rebuild([_adv("e2", node_id="nde-2", generation=1)])
    assert directory.get("stale") is None
    assert directory.get("e2") is not None
