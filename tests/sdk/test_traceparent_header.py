"""Contract P1: the SDK forwards the caller's ambient trace context as a
``traceparent`` header, without a hard OpenTelemetry dependency.

The SDK soft-imports the OTel propagator, so both paths must be proven: with the API
importable the ambient context is injected; without it ``_build_headers`` still returns
valid headers and does not raise.
"""

from unittest import mock

from flowmesh import _base_client


def test_build_headers_injects_the_ambient_traceparent_when_otel_is_available() -> None:
    def fake_inject(headers: dict[str, str]) -> None:
        headers["traceparent"] = (
            "00-11111111111111111111111111111111-2222222222222222-01"
        )

    with mock.patch.object(_base_client, "_otel_inject", fake_inject):
        headers = _base_client._build_headers("key")

    assert headers["traceparent"] == (
        "00-11111111111111111111111111111111-2222222222222222-01"
    )
    assert headers["Authorization"] == "Bearer key"


def test_build_headers_is_a_noop_when_otel_is_absent() -> None:
    with mock.patch.object(_base_client, "_otel_inject", None):
        headers = _base_client._build_headers("key")

    assert headers == {
        "Accept": "application/json",
        "User-Agent": _base_client.USER_AGENT,
        "Authorization": "Bearer key",
    }
    assert "traceparent" not in headers


def test_build_headers_without_api_key_omits_authorization() -> None:
    with mock.patch.object(_base_client, "_otel_inject", None):
        headers = _base_client._build_headers(None)

    assert "Authorization" not in headers
    assert "traceparent" not in headers
