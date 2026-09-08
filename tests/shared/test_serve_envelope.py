"""Freezing a task-addressed serve request into its transparent envelope.

The envelope carries the client's request unchanged apart from the fields that
describe one transport hop, refuses an ambiguously framed or authority-bearing request
before it can reach an engine, and digests the whole request so the claim fence binds
more than the body.
"""

import pytest

from shared.resident.envelope import (
    EnvelopeRejected,
    ServeRequestEnvelope,
    filter_response_headers,
    freeze_request_envelope,
)


def _freeze(
    method: str = "POST",
    upstream_path: str = "v1/chat/completions",
    query: str = "",
    headers: list[tuple[str, str]] | None = None,
    body: bytes = b"{}",
) -> ServeRequestEnvelope:
    return freeze_request_envelope(
        method=method,
        upstream_path=upstream_path,
        query=query,
        headers=[] if headers is None else headers,
        body=body,
    )


def test_the_client_request_is_carried_unchanged() -> None:
    envelope = _freeze(
        method="get",
        upstream_path="v1/models",
        query="limit=2",
        headers=[("Content-Type", "application/json"), ("X-Trace", "t1")],
        body=b"",
    )
    assert envelope.method == "GET"
    assert envelope.path == "/v1/models"
    assert envelope.target == "/v1/models?limit=2"
    assert envelope.headers == (
        ("Content-Type", "application/json"),
        ("X-Trace", "t1"),
    )
    assert envelope.body == b""


def test_a_binary_body_survives_the_round_trip() -> None:
    blob = bytes(range(256))
    envelope = _freeze(body=blob)
    restored = ServeRequestEnvelope.from_parts(envelope.header_fields(), envelope.body)
    assert restored == envelope
    assert restored.digest() == envelope.digest()


def test_repeated_request_headers_keep_their_order_and_duplicates() -> None:
    envelope = _freeze(headers=[("Accept", "a"), ("X-K", "1"), ("Accept", "b")])
    assert envelope.headers == (("Accept", "a"), ("X-K", "1"), ("Accept", "b"))


def test_the_client_credential_and_hop_fields_never_reach_the_engine() -> None:
    envelope = _freeze(
        headers=[
            ("Authorization", "Bearer client-token"),
            ("Proxy-Authorization", "Basic x"),
            ("Host", "evil.example"),
            ("Content-Length", "2"),
            ("Connection", "keep-alive, X-Hop"),
            ("X-Hop", "dropped"),
            ("Keep-Alive", "timeout=5"),
            ("X-Kept", "kept"),
        ]
    )
    assert envelope.headers == (("X-Kept", "kept"),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upstream_path": "http://evil.example/v1/models"},
        {"upstream_path": "https://evil.example/v1/models"},
        {"upstream_path": "/evil.example/v1/models"},
    ],
)
def test_an_authority_bearing_request_target_is_refused(kwargs: dict) -> None:
    # A target that can name a host would let the caller redirect the upstream request.
    with pytest.raises(EnvelopeRejected):
        _freeze(**kwargs)


def test_ambiguous_or_conflicting_framing_is_refused() -> None:
    with pytest.raises(EnvelopeRejected):
        _freeze(headers=[("Content-Length", "2"), ("Content-Length", "3")])
    with pytest.raises(EnvelopeRejected):
        _freeze(headers=[("Transfer-Encoding", "chunked"), ("Content-Length", "2")])
    with pytest.raises(EnvelopeRejected):
        _freeze(headers=[("Transfer-Encoding", "chunked"), ("Transfer-Encoding", "x")])
    with pytest.raises(EnvelopeRejected):
        _freeze(headers=[("Content-Length", "not-a-number")])


def test_a_protocol_upgrade_is_refused() -> None:
    with pytest.raises(EnvelopeRejected):
        _freeze(headers=[("Upgrade", "websocket")])
    with pytest.raises(EnvelopeRejected):
        _freeze(headers=[("Connection", "Upgrade")])


def test_the_digest_covers_the_whole_request_not_only_its_body() -> None:
    base = _freeze()
    for changed in (
        _freeze(method="PUT"),
        _freeze(upstream_path="v1/embeddings"),
        _freeze(query="stream=1"),
        _freeze(headers=[("X-K", "1")]),
        _freeze(body=b"{ }"),
    ):
        assert changed.digest() != base.digest()
    assert _freeze().digest() == base.digest()


def test_the_digest_separates_header_boundaries() -> None:
    # Concatenating the fields would let two different requests share one digest.
    a = _freeze(headers=[("X-A", "1"), ("X-B", "2")])
    b = _freeze(headers=[("X-A", "1"), ("X-B", "2")][:1] + [("X-A1", "2")])
    assert a.digest() != b.digest()


def test_response_headers_keep_repeats_and_drop_hop_fields() -> None:
    kept = filter_response_headers(
        [
            ("Content-Type", "application/json"),
            ("Set-Cookie", "a=1"),
            ("Set-Cookie", "b=2"),
            ("Connection", "keep-alive, X-Hop"),
            ("X-Hop", "dropped"),
            ("Transfer-Encoding", "chunked"),
        ]
    )
    assert kept == (
        ("Content-Type", "application/json"),
        ("Set-Cookie", "a=1"),
        ("Set-Cookie", "b=2"),
    )
