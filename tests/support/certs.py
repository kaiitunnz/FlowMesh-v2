"""Issue a throwaway CA and leaf certificates for mutual-TLS tests."""

import base64
import datetime
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.x509.oid import NameOID

_DAY = datetime.timedelta(days=1)


@dataclass(frozen=True)
class Issued:
    """One issued identity: its certificate and key, base64 PEM."""

    cert_b64: str
    key_b64: str


@dataclass(frozen=True)
class TestCa:
    """A throwaway CA that signs leaf certificates for a given identity."""

    ca_b64: str
    signer: ed25519.Ed25519PrivateKey
    issuer_cert: x509.Certificate

    def issue(self, identity: str) -> Issued:
        key = ed25519.Ed25519PrivateKey.generate()
        now = datetime.datetime.now(datetime.UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, identity)])
            )
            .issuer_name(self.issuer_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _DAY)
            .not_valid_after(now + _DAY)
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(identity)]), critical=False
            )
            .sign(self.signer, None)
        )
        return Issued(_b64(_pem(cert)), _b64(_key_pem(key)))


def new_ca(name: str = "flowmesh-test-ca") -> TestCa:
    key = ed25519.Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _DAY)
        .not_valid_after(now + _DAY)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, None)
    )
    return TestCa(_b64(_pem(cert)), key, cert)


def _pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _key_pem(key: ed25519.Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


__all__ = ["Issued", "TestCa", "new_ca"]
