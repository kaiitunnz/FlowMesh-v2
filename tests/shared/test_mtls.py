"""The offload transports' material loading and peer identity checks."""

import base64

import pytest

from shared.network.mtls import (
    MutualTlsMaterial,
    MutualTlsMaterialError,
    peer_identities,
    peer_matches,
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


def test_a_peer_control_named_is_admitted():
    assert peer_matches(_cert("wkr-1"), ["wkr-1"])
    assert peer_matches(_cert("wkr-1", "node-a"), ["node-a"])


def test_a_ca_signed_peer_control_did_not_name_is_refused():
    # The whole point of the identity check: a certificate the deployment CA signed for
    # some other party verifies, so the CA alone must not admit it.
    assert not peer_matches(_cert("wkr-9"), ["wkr-1"])


def test_an_unnamed_expectation_never_matches():
    assert not peer_matches(_cert("wkr-1"), [])
    assert not peer_matches(_cert("wkr-1"), [""])
    assert not peer_matches(None, ["wkr-1"])
