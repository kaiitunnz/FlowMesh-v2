"""The transparent HTTP envelope a task-addressed serve request is frozen into.

A gated serve ingress freezes the client's request — its method, origin-form path and
query, ordered end-to-end header fields, and raw body — before admission, and derives
the canonical descriptor digest bound into the claim fence. The selected replica's
sidecar recomputes that digest from the frozen envelope it received and forwards the
same method, path, query, headers, and body to its co-located engine verbatim, so
FlowMesh applies no endpoint semantics of its own. The header travels as the relay
message's JSON and the raw body as the bytes that follow it in the same opaque payload,
so a binary body survives unchanged without a second encoding, and the servers between
the two ends relay it without decoding.

Client credentials and framing headers never reach the engine: the ingress strips
``Authorization`` and ``Proxy-Authorization`` (the sidecar owns the engine credential),
``Host`` and ``Content-Length`` (the upstream host and body length are the sidecar's),
and every hop-by-hop or ``Connection``-nominated field. An ambiguously framed request is
refused before it can reach an engine at all.
"""

import hashlib
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

# The transparent method set. A serve binding may narrow it, but no method outside it
# is proxied: CONNECT and protocol upgrades are not HTTP request/response exchanges
# the claim-gated relay can carry under one fenced terminal.
TRANSPARENT_METHODS = ("GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD")

# Hop-by-hop fields belong to one transport hop, so neither direction forwards them.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",  # codespell:ignore te
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# End-to-end request fields the ingress still never forwards: the client credential is
# not the engine credential, and the upstream host and body length are the sidecar's.
_NEVER_FORWARDED_REQUEST_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "host", "content-length"}
)

_HEADER_SEP = b"\x1f"
_FIELD_SEP = b"\x1e"


class EnvelopeRejected(Exception):
    """The request cannot be frozen into an unambiguous transparent envelope."""


def connection_nominated_headers(values: Iterable[str]) -> frozenset[str]:
    """The extra per-hop field names a ``Connection`` header nominates.

    ``Connection: X-Foo`` makes ``X-Foo`` hop-by-hop for this hop only, so it is dropped
    alongside the fixed set.
    """
    nominated: set[str] = set()
    for raw in values:
        nominated.update(
            token.strip().lower() for token in raw.split(",") if token.strip()
        )
    return frozenset(nominated)


def _values(headers: Iterable[tuple[str, str]], name: str) -> list[str]:
    return [value for header, value in headers if header.lower() == name]


def filter_response_headers(
    headers: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Keep every non-hop-by-hop response field, including repeated ones.

    The engine's own response metadata reaches the client unchanged apart from the
    fields that describe this hop's transport.
    """
    items = list(headers)
    nominated = connection_nominated_headers(_values(items, "connection"))
    return tuple(
        (name, value)
        for name, value in items
        if name.lower() not in _HOP_BY_HOP_HEADERS and name.lower() not in nominated
    )


class ServeRequestEnvelope(BaseModel):
    """One frozen task-addressed request, relayed verbatim to the engine."""

    model_config = ConfigDict(frozen=True)

    method: str
    path: str
    query: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b""

    @property
    def target(self) -> str:
        """The origin-form request target the sidecar dials on the engine."""
        return f"{self.path}?{self.query}" if self.query else self.path

    def digest(self) -> str:
        """The canonical descriptor digest bound into the claim fence.

        It covers the method, request target, ordered filtered headers, and the body's
        length and content digest, so the fence binds the whole request rather than
        its body alone and the sidecar can prove the envelope it holds is the
        admitted one.
        """
        parts = [
            self.method.encode(),
            self.path.encode(),
            self.query.encode(),
        ]
        for name, value in self.headers:
            parts.append(name.lower().encode() + _HEADER_SEP + value.encode())
        parts.append(str(len(self.body)).encode())
        parts.append(hashlib.sha256(self.body).hexdigest().encode())
        return hashlib.sha256(_FIELD_SEP.join(parts)).hexdigest()

    def header_fields(self) -> dict[str, Any]:
        """The envelope's JSON header; the raw body rides after it on the wire."""
        return {
            "method": self.method,
            "path": self.path,
            "query": self.query,
            "headers": [[name, value] for name, value in self.headers],
        }

    @classmethod
    def from_parts(cls, header: Any, body: bytes) -> "ServeRequestEnvelope":
        """Rebuild the frozen envelope from the relayed header and raw body."""
        if not isinstance(header, dict):
            raise EnvelopeRejected("serve request envelope missing")
        headers = tuple(
            (str(item[0]), str(item[1])) for item in header.get("headers") or () if item
        )
        return cls(
            method=str(header.get("method") or ""),
            path=str(header.get("path") or ""),
            query=str(header.get("query") or ""),
            headers=headers,
            body=body,
        )


def freeze_request_envelope(
    *,
    method: str,
    upstream_path: str,
    query: str,
    headers: Iterable[tuple[str, str]],
    body: bytes,
) -> ServeRequestEnvelope:
    """Freeze a client request into its transparent envelope, or refuse it.

    The request target must be an unambiguous origin-form path: an absolute-form URL or
    a leading ``//`` authority would let the target name a host, and ambiguous framing
    (duplicate or conflicting ``Content-Length``/``Transfer-Encoding``) or a protocol
    upgrade is refused before admission rather than resolved by guessing.
    """
    items = list(headers)
    lowered = [(name.lower(), value) for name, value in items]

    if "://" in upstream_path or upstream_path.lower().startswith(("http:", "https:")):
        raise EnvelopeRejected(
            "request target must be origin-form, not an absolute URL"
        )
    # A leading slash here would make the target ``//authority``: refuse it rather than
    # normalize it away, so a caller cannot smuggle a host past the upstream resolution.
    path = "/" + upstream_path
    if path.startswith("//"):
        raise EnvelopeRejected("request target must not carry an authority")

    content_lengths = _values(lowered, "content-length")
    transfer_encodings = _values(lowered, "transfer-encoding")
    if len(content_lengths) > 1 or len(transfer_encodings) > 1:
        raise EnvelopeRejected("ambiguous request framing")
    if content_lengths and transfer_encodings:
        raise EnvelopeRejected("conflicting request framing")
    if content_lengths and not content_lengths[0].strip().isdigit():
        raise EnvelopeRejected("invalid content-length")

    nominated = connection_nominated_headers(_values(lowered, "connection"))
    if any(name == "upgrade" for name, _ in lowered) or "upgrade" in nominated:
        raise EnvelopeRejected("protocol upgrades are not proxied")

    forwarded = tuple(
        (name, value)
        for name, value in items
        if name.lower() not in _HOP_BY_HOP_HEADERS
        and name.lower() not in _NEVER_FORWARDED_REQUEST_HEADERS
        and name.lower() not in nominated
    )
    return ServeRequestEnvelope(
        method=method.upper(),
        path=path,
        query=query,
        headers=forwarded,
        body=body,
    )
