"""A node and a worker refuse to serve offloads in plaintext under a mutual-TLS posture.

Material an operator configured and the process cannot read is fatal at start-up. The
listener is advertised either way, so returning no material instead would carry resident
payloads over a wire the deployment asked to protect.
"""

import logging
from types import SimpleNamespace
from typing import cast

import pytest

from server.config import TrustedOffloadConfig
from server.supervisor.supervisor import _offload_material as _node_material
from shared.network.mtls import MutualTlsMaterialError
from worker.config import WorkerConfig
from worker.main import _offload_material as _worker_material

_LOGGER = logging.getLogger("test-offload-posture")


def _node_config(**overrides) -> TrustedOffloadConfig:
    return TrustedOffloadConfig(
        enabled=True,
        trust_domain="td",
        classes=("same_cluster",),
        require_mtls=True,
        tls_ca_file="/absent/ca.pem",
        tls_cert_file="/absent/cert.pem",
        tls_key_file="/absent/key.pem",
        **overrides,
    )


def test_a_node_that_cannot_read_its_configured_material_refuses_to_start():
    with pytest.raises(MutualTlsMaterialError):
        _node_material(_node_config(), _LOGGER)


def test_a_node_attesting_a_trusted_network_serves_without_material():
    config = TrustedOffloadConfig(
        enabled=True, trust_domain="td", classes=("same_cluster",), require_mtls=False
    )
    assert _node_material(config, _LOGGER) is None


def _worker_config(material: str | None) -> WorkerConfig:
    # The loader reads only the offload posture, so the rest of a worker's environment
    # is not what is under test here.
    return cast(
        WorkerConfig,
        SimpleNamespace(
            offload_enabled=True,
            offload_tls_ca_b64=material,
            offload_tls_cert_b64=material,
            offload_tls_key_b64=material,
        ),
    )


def test_a_worker_handed_unreadable_material_refuses_to_start():
    with pytest.raises(MutualTlsMaterialError):
        _worker_material(_worker_config("not base64"), _LOGGER)


def test_a_worker_without_material_dials_on_the_attested_network():
    assert _worker_material(_worker_config(None), _LOGGER) is None
