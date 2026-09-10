"""The offload transports' material loading and peer identity reading."""

import base64

import pytest

from shared.network.mtls import (
    MutualTlsMaterial,
    MutualTlsMaterialError,
    peer_identities,
)
from tests.support.certs import new_ca


def _material(tmp_path, identity: str) -> MutualTlsMaterial:
    ca = new_ca()
    issued = ca.issue(identity)
    files = {}
    for name, value in (
        ("ca.pem", ca.ca_b64),
        ("cert.pem", issued.cert_b64),
        ("key.pem", issued.key_b64),
    ):
        path = tmp_path / name
        path.write_bytes(base64.b64decode(value))
        files[name] = path.as_posix()
    return MutualTlsMaterial.from_files(
        ca_file=files["ca.pem"], cert_file=files["cert.pem"], key_file=files["key.pem"]
    )


def test_file_material_round_trips_through_the_transient_encoding(tmp_path):
    material = _material(tmp_path, "wkr-1.node-a")
    ca_b64, cert_b64, key_b64 = material.to_b64()

    assert (
        MutualTlsMaterial.from_b64(ca_b64=ca_b64, cert_b64=cert_b64, key_b64=key_b64)
        == material
    )


def test_an_unconfigured_file_is_refused(tmp_path):
    with pytest.raises(MutualTlsMaterialError):
        MutualTlsMaterial.from_files(ca_file="", cert_file="x", key_file="y")


def test_an_unreadable_file_is_refused(tmp_path):
    with pytest.raises(MutualTlsMaterialError):
        MutualTlsMaterial.from_files(
            ca_file=(tmp_path / "absent.pem").as_posix(),
            cert_file=(tmp_path / "absent.pem").as_posix(),
            key_file=(tmp_path / "absent.pem").as_posix(),
        )


def test_undecodable_transient_material_is_refused():
    with pytest.raises(MutualTlsMaterialError):
        MutualTlsMaterial.from_b64(ca_b64="not base64", cert_b64="a", key_b64="a")


def _cert(*names: str) -> dict:
    return {
        "subject": ((("commonName", names[0]),),),
        "subjectAltName": tuple(("DNS", name) for name in names[1:]),
    }


def test_peer_identities_reads_the_common_name_and_alternatives():
    assert peer_identities(_cert("wkr-1", "node-a", "wkr-1.fabric")) == frozenset(
        {"wkr-1", "node-a", "wkr-1.fabric"}
    )
