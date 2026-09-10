"""A node and a worker refuse to serve peers in plaintext unless an operator says so.

Mutual TLS is on by default, so material that is missing or unusable fails start-up
rather than falling back: the listener is advertised either way, and serving it in
plaintext would carry resident payloads over a wire the deployment asked to protect.
Only the explicit disable flag, an operator attesting a trusted network, runs without
it.
"""

import logging
import ssl
from types import SimpleNamespace
from typing import cast

import pytest

from server.config import TrustedPeerConfig
from server.supervisor.supervisor import _peer_material as _node_material
from shared.network.mtls import MutualTlsMaterialError, server_context
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
