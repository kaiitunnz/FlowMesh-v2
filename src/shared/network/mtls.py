"""Mutual-TLS material for the peer transports.

Both ends of a dialed peer session authenticate each other against the deployment's
configured CA. The two ends check different things with that proof: an origin requires
the certificate to cover the endpoint host it dialed, so the peer is the target its
route selected, while a target admits any peer the CA vouched for and leaves the
question of which invocation may be carried to the replica's claim gate. A target that
must know which node dialed it checks that the peer's certificate covers that node's
advertised host, by the same subject-alternative-name rule an origin applies.

Material is configured as operator files and is base64-encoded only when it is loaded
into a transient worker attachment.
"""

import base64
import binascii
import contextlib
import ipaddress
import ssl
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self


class MutualTlsMaterialError(ValueError):
    """The configured certificate material is missing, unreadable, or undecodable."""


@dataclass(frozen=True)
class MutualTlsMaterial:
    """A CA bundle and the holder's own certificate chain and private key."""

    ca_pem: bytes
    cert_pem: bytes
    key_pem: bytes

    @classmethod
    def from_files(cls, *, ca_file: str, cert_file: str, key_file: str) -> Self:
        return cls(
            ca_pem=_read(ca_file, "CA bundle"),
            cert_pem=_read(cert_file, "certificate"),
            key_pem=_read(key_file, "private key"),
        )

    @classmethod
    def from_b64(cls, *, ca_b64: str, cert_b64: str, key_b64: str) -> Self:
        return cls(
            ca_pem=_decode(ca_b64, "CA bundle"),
            cert_pem=_decode(cert_b64, "certificate"),
            key_pem=_decode(key_b64, "private key"),
        )

    def to_b64(self) -> tuple[str, str, str]:
        """The CA, certificate, and key encoded for a transient worker attachment."""
        return (
            base64.b64encode(self.ca_pem).decode("ascii"),
            base64.b64encode(self.cert_pem).decode("ascii"),
            base64.b64encode(self.key_pem).decode("ascii"),
        )


def _read(path: str, what: str) -> bytes:
    if not path.strip():
        raise MutualTlsMaterialError(f"peer {what} file is not configured")
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise MutualTlsMaterialError(f"peer {what} file is unreadable: {exc}") from exc


def _decode(value: str, what: str) -> bytes:
    if not value.strip():
        raise MutualTlsMaterialError(f"peer {what} is not configured")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MutualTlsMaterialError(f"peer {what} is not valid base64") from exc


@contextmanager
def _chain_files(material: MutualTlsMaterial) -> Iterator[tuple[Path, Path, Path]]:
    with tempfile.TemporaryDirectory(prefix="flowmesh-peer-tls-") as directory:
        root = Path(directory)
        ca, cert, key = root / "ca.pem", root / "cert.pem", root / "key.pem"
        for path, content in ((ca, material.ca_pem), (cert, material.cert_pem)):
            path.write_bytes(content)
        key.write_bytes(material.key_pem)
        key.chmod(0o600)
        yield ca, cert, key


def client_context(material: MutualTlsMaterial) -> ssl.SSLContext:
    """A dialing origin's context: present its certificate, verify the target's.

    A route names its target as the endpoint host the origin dials, so the origin
    validates that the peer's certificate covers that host: the operator lists a node's
    reachable address among its certificate's subject-alternative names, and issuing
    only to registered nodes is what makes holding such a certificate meaningful.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    with _chain_files(material) as (ca, cert, key):
        context.load_verify_locations(cafile=ca.as_posix())
        context.load_cert_chain(certfile=cert.as_posix(), keyfile=key.as_posix())
    return context


def server_context(material: MutualTlsMaterial) -> ssl.SSLContext:
    """A target listener's context: require and verify the dialing origin's chain."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.verify_mode = ssl.CERT_REQUIRED
    with _chain_files(material) as (ca, cert, key):
        context.load_verify_locations(cafile=ca.as_posix())
        context.load_cert_chain(certfile=cert.as_posix(), keyfile=key.as_posix())
    return context


@dataclass(frozen=True)
class PeerNames:
    """The hosts a verified peer certificate's subject alternative names cover.

    Only DNS and IP-address names count; a subject common name never names a host here.
    """

    dns: frozenset[str] = frozenset()
    ips: frozenset[str] = frozenset()

    @classmethod
    def of(cls, peer_cert: dict[str, Any] | None) -> Self:
        dns: set[str] = set()
        ips: set[str] = set()
        for kind, value in (peer_cert or {}).get("subjectAltName", ()):
            match kind:
                case "DNS":
                    dns.add(str(value).lower())
                case "IP Address":
                    # Python renders an IP SAN it cannot decode as "<invalid>".
                    with contextlib.suppress(ValueError):
                        ips.add(_ip_text(str(value)))
        return cls(frozenset(dns), frozenset(ips))

    def covers(self, host: str) -> bool:
        """Whether the names cover ``host``, as TLS hostname verification decides it.

        An IP literal matches only an IP-address name. A hostname matches a DNS name
        exactly or by a wildcard standing for its whole leftmost label.
        """
        try:
            return _ip_text(host) in self.ips
        except ValueError:
            pass
        host = host.lower().rstrip(".")
        if host in self.dns:
            return True
        label, _, rest = host.partition(".")
        return bool(label and rest) and f"*.{rest}" in self.dns


def _ip_text(value: str) -> str:
    return ipaddress.ip_address(value.strip()).compressed


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


__all__ = [
    "MutualTlsMaterial",
    "MutualTlsMaterialError",
    "PeerNames",
    "client_context",
    "peer_identities",
    "server_context",
]
