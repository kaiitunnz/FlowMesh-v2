"""Mutual-TLS material for the trusted target-leg transports.

Both ends of a forward-dialed target leg authenticate each other: the root presents a
client certificate and the target requires one, and each validates the peer against the
deployment's configured CA. The target additionally pins the root's certificate
identity, so a certificate the CA signed for some other party is refused — a CA bundle
alone does not prove to a target that its dialer is the root.

Material is carried as base64 PEM so it travels the same way as the rest of a node's
configured credentials, and is written to a private temporary file only because
``ssl`` loads a chain from a path.
"""

import base64
import binascii
import ssl
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MutualTlsMaterialError(ValueError):
    """The configured certificate material is missing or undecodable."""


@dataclass(frozen=True)
class MutualTlsMaterial:
    """The CA bundle, own certificate chain and key, and the pinned root identity."""

    ca_pem: bytes
    cert_pem: bytes
    key_pem: bytes
    root_identity: str

    @classmethod
    def from_b64(
        cls, *, ca_b64: str, cert_b64: str, key_b64: str, root_identity: str
    ) -> "MutualTlsMaterial":
        return cls(
            ca_pem=_decode(ca_b64, "CA bundle"),
            cert_pem=_decode(cert_b64, "certificate"),
            key_pem=_decode(key_b64, "private key"),
            root_identity=root_identity,
        )


def _decode(value: str, what: str) -> bytes:
    if not value.strip():
        raise MutualTlsMaterialError(f"target-leg {what} is not configured")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MutualTlsMaterialError(f"target-leg {what} is not valid base64") from exc


@contextmanager
def _chain_files(material: MutualTlsMaterial) -> Iterator[tuple[Path, Path, Path]]:
    with tempfile.TemporaryDirectory(prefix="flowmesh-target-leg-") as directory:
        root = Path(directory)
        ca, cert, key = root / "ca.pem", root / "cert.pem", root / "key.pem"
        for path, content in ((ca, material.ca_pem), (cert, material.cert_pem)):
            path.write_bytes(content)
        key.write_bytes(material.key_pem)
        key.chmod(0o600)
        yield ca, cert, key


def client_context(material: MutualTlsMaterial) -> ssl.SSLContext:
    """The root's dialing context: present its certificate, verify the target's.

    Hostname verification is off because a target is named by its control-plane
    advertisement rather than by DNS, and its address moves with the replica; the CA
    and the certificate the target presents are what the root validates.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    with _chain_files(material) as (ca, cert, key):
        context.load_verify_locations(cafile=ca.as_posix())
        context.load_cert_chain(certfile=cert.as_posix(), keyfile=key.as_posix())
    return context


def server_context(material: MutualTlsMaterial) -> ssl.SSLContext:
    """A target listener's context: require and verify the dialer's certificate."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.verify_mode = ssl.CERT_REQUIRED
    with _chain_files(material) as (ca, cert, key):
        context.load_verify_locations(cafile=ca.as_posix())
        context.load_cert_chain(certfile=cert.as_posix(), keyfile=key.as_posix())
    return context


def peer_identities(peer_cert: dict[str, Any] | None) -> frozenset[str]:
    """The subject common names and subject-alternative names a peer presented."""
    if not peer_cert:
        return frozenset()
    names: set[str] = set()
    for field in peer_cert.get("subject", ()):
        for key, value in field:
            if key == "commonName":
                names.add(str(value))
    for kind, value in peer_cert.get("subjectAltName", ()):
        if kind in ("DNS", "URI", "IP Address"):
            names.add(str(value))
    return frozenset(names)


def is_pinned_root(peer_cert: dict[str, Any] | None, root_identity: str) -> bool:
    """Whether the verified peer certificate carries the pinned root identity."""
    return bool(root_identity) and root_identity in peer_identities(peer_cert)


__all__ = [
    "MutualTlsMaterial",
    "MutualTlsMaterialError",
    "client_context",
    "is_pinned_root",
    "peer_identities",
    "server_context",
]
