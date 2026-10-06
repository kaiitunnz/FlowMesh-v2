"""A node and a worker refuse to serve peers in plaintext unless an operator says so.

Mutual TLS is on by default, so material that is missing or unusable fails start-up
rather than falling back: the listener is advertised either way, and serving it in
plaintext would carry resident payloads over a wire the deployment asked to protect.
Only the explicit disable flag, an operator attesting a trusted network, runs without
it. The root's serve ingress, which only dials, carries its requests over the relay
when its material is unusable, and its node fails start-up.
"""

import dataclasses
import logging
import ssl
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from server.config import NetworkPlaneConfig, TrustedPeerConfig
from server.network.peer_tls import load_peer_material as _node_material
from server.network.peer_tls import root_peer_dialer
from shared.network.mtls import MutualTlsMaterialError, server_context
from tests.support.certs import new_ca
from worker.config import WorkerConfig
from worker.main import _peer_material as _worker_material

_LOGGER = logging.getLogger("test-peer-posture")


def _node_config(**overrides) -> TrustedPeerConfig:
    material = {
        "tls_ca_file": "/etc/ssl/peer/peer-ca.pem",
        "tls_cert_file": "/etc/ssl/peer/peer.pem",
        "tls_key_file": "/etc/ssl/peer/peer.key",
    }
    return TrustedPeerConfig(
        enabled=True,
        trust_domain="td",
        classes=("same_cluster",),
        **{**material, **overrides},
    )


def _worker_config(material: str | None, disable_mtls: bool = False) -> WorkerConfig:
    # The loader reads only the peer posture, so the rest of a worker's environment
    # is not what is under test here.
    return cast(
        WorkerConfig,
        SimpleNamespace(
            peer_enabled=True,
            peer_disable_mtls=disable_mtls,
            peer_tls_ca_b64=material,
            peer_tls_cert_b64=material,
            peer_tls_key_b64=material,
        ),
    )


def test_a_node_whose_material_is_missing_refuses_to_start():
    with pytest.raises(MutualTlsMaterialError):
        _node_material(_node_config(), _LOGGER)


def test_material_that_is_not_a_certificate_refuses_to_serve(tmp_path):
    # Loading only reads the bytes; a file that is not an identity fails where the
    # listener builds its context, which is still before it serves a connection.
    unusable = tmp_path / "peer.pem"
    unusable.write_text("not a certificate")
    material = _node_material(
        _node_config(
            tls_ca_file=unusable.as_posix(),
            tls_cert_file=unusable.as_posix(),
            tls_key_file=unusable.as_posix(),
        ),
        _LOGGER,
    )
    assert material is not None
    with pytest.raises(ssl.SSLError):
        server_context(material)


def test_a_node_attesting_a_trusted_network_serves_without_material():
    assert _node_material(_node_config(disable_mtls=True), _LOGGER) is None


def test_a_worker_given_no_material_refuses_to_start():
    # The supervisor reads the files and fails on its own side first, so a worker that
    # reaches this state is misconfigured rather than attesting anything.
    with pytest.raises(MutualTlsMaterialError):
        _worker_material(_worker_config(None), _LOGGER)


def test_a_worker_handed_unreadable_material_refuses_to_start():
    with pytest.raises(MutualTlsMaterialError):
        _worker_material(_worker_config("not base64"), _LOGGER)


def test_a_worker_on_an_attested_network_dials_without_material():
    assert _worker_material(_worker_config(None, disable_mtls=True), _LOGGER) is None


def _network(peer: TrustedPeerConfig, enabled: bool = True) -> NetworkPlaneConfig:
    return NetworkPlaneConfig(enabled=enabled, peer=peer, connect_budget_sec=3.0)


def _write_material(tmp_path: Path) -> TrustedPeerConfig:
    material = new_ca().material("root-node")
    paths = {}
    for name, content in (
        ("tls_ca_file", material.ca_pem),
        ("tls_cert_file", material.cert_pem),
        ("tls_key_file", material.key_pem),
    ):
        path = tmp_path / name
        path.write_bytes(content)
        paths[name] = path.as_posix()
    return _node_config(**paths)


def test_the_root_dials_with_its_nodes_identity(tmp_path):
    dialer = root_peer_dialer(_network(_write_material(tmp_path)), _LOGGER)
    assert dialer is not None
    assert isinstance(dialer.ssl_context, ssl.SSLContext)
    assert dialer.ssl_context.verify_mode is ssl.CERT_REQUIRED
    assert dialer.ssl_context.check_hostname
    assert dialer.connect_budget_sec == 3.0


@pytest.mark.parametrize(
    ("network_enabled", "peer_enabled"), [(True, False), (False, True)]
)
def test_a_root_without_the_peer_plane_dials_nothing(network_enabled, peer_enabled):
    peer = dataclasses.replace(_node_config(), enabled=peer_enabled)
    assert root_peer_dialer(_network(peer, network_enabled), _LOGGER) is None


def test_a_root_with_unusable_material_rides_the_relay_rather_than_plaintext(
    tmp_path, caplog
):
    unusable = tmp_path / "peer.pem"
    unusable.write_text("not a certificate")
    for config in (
        _node_config(),
        _node_config(
            tls_ca_file=unusable.as_posix(),
            tls_cert_file=unusable.as_posix(),
            tls_key_file=unusable.as_posix(),
        ),
    ):
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            assert root_peer_dialer(_network(config), _LOGGER) is None
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1 and "control_relay" in errors[0].getMessage()


def test_a_root_attesting_a_trusted_network_dials_without_tls(caplog):
    with caplog.at_level(logging.WARNING):
        dialer = root_peer_dialer(_network(_node_config(disable_mtls=True)), _LOGGER)
    assert dialer is not None and dialer.ssl_context is None
    assert any("without mutual TLS" in r.getMessage() for r in caplog.records)
