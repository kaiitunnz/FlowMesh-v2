"""A network-plane node advertises so the resolver can carry the universal control_relay
base between it and any peer — including an outbound-only node with no inbound URL, for
which the server derives the relay attachment identity from the node."""

from server.config import NetworkPlaneConfig, TrustedPeerConfig
from server.supervisor.supervisor import _endpoint_advertisement_provider
from shared.schemas.network import PEER_PROTOCOL


def test_carries_the_inbound_url_when_configured() -> None:
    provider = _endpoint_advertisement_provider(
        NetworkPlaneConfig(enabled=True, endpoint_url="127.0.0.1:9101")
    )
    first = provider()
    assert first is not None
    assert first.url == "127.0.0.1:9101"
    # A re-registration bumps the generation so the prior advertisement and its stale
    # relay evidence are superseded together.
    second = provider()
    assert second is not None
    assert second.generation > first.generation


def test_advertises_when_enabled_even_without_an_inbound_url() -> None:
    # An outbound-only node still attaches outward, so it advertises (empty url) and is
    # relay attach-eligible; the server derives its attachment identity from the node.
    provider = _endpoint_advertisement_provider(NetworkPlaneConfig(enabled=True))
    ad = provider()
    assert ad is not None
    assert ad.url == ""


def test_no_advertisement_when_the_plane_is_disabled() -> None:
    provider = _endpoint_advertisement_provider(NetworkPlaneConfig(enabled=False))
    assert provider() is None


def test_a_peer_node_without_a_listener_can_dial_but_is_not_dialed() -> None:
    provider = _endpoint_advertisement_provider(
        NetworkPlaneConfig(enabled=True, peer=TrustedPeerConfig(enabled=True))
    )
    ad = provider()
    assert ad is not None
    assert ad.protocols == (PEER_PROTOCOL,)
    assert ad.peer_url == ""


def test_a_peer_node_with_a_listener_is_also_dialed_through_it() -> None:
    provider = _endpoint_advertisement_provider(
        NetworkPlaneConfig(
            enabled=True,
            peer=TrustedPeerConfig(enabled=True, node_listener_url="10.0.0.2:9102"),
        )
    )
    ad = provider()
    assert ad is not None
    assert ad.protocols == (PEER_PROTOCOL,)
    assert ad.peer_url == "10.0.0.2:9102"


def test_a_node_without_the_peer_plane_neither_dials_nor_is_dialed() -> None:
    provider = _endpoint_advertisement_provider(
        NetworkPlaneConfig(
            enabled=True, peer=TrustedPeerConfig(node_listener_url="10.0.0.2:9102")
        )
    )
    ad = provider()
    assert ad is not None
    assert ad.protocols == ()
    assert ad.peer_url == ""
