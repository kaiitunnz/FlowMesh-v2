"""The SDK forwards the caller's ambient trace context as a ``traceparent`` header.

The context belongs to the request, not to the client: a client outlives the span it
was constructed under, so injecting once at construction stamps whatever happened to be
active then onto every later request. The SDK soft-imports the OTel propagator, so the
absent-OTel path must be proven too.
"""

from unittest import mock

from flowmesh import _base_client


def _injector(value: str):
    def fake_inject(headers: dict[str, str]) -> None:
        headers["traceparent"] = value

    return fake_inject


_FIRST = "00-11111111111111111111111111111111-2222222222222222-01"
_SECOND = "00-33333333333333333333333333333333-4444444444444444-01"


def test_a_request_carries_the_context_active_when_it_is_made() -> None:
    with mock.patch.object(_base_client, "_otel_inject", _injector(_FIRST)):
        headers = _base_client._traced_headers(None)

    assert headers is not None
    assert headers["traceparent"] == _FIRST


def test_the_callers_own_headers_are_kept_alongside_it() -> None:
    with mock.patch.object(_base_client, "_otel_inject", _injector(_FIRST)):
        headers = _base_client._traced_headers({"X-Caller": "value"})

    assert headers is not None
    assert headers["X-Caller"] == "value"
    assert headers["traceparent"] == _FIRST


def test_two_requests_under_two_contexts_carry_different_values() -> None:
    """The regression guard: one value frozen for the client's life is the defect."""
    with mock.patch.object(_base_client, "_otel_inject", _injector(_FIRST)):
        first = _base_client._traced_headers(None)
    with mock.patch.object(_base_client, "_otel_inject", _injector(_SECOND)):
        second = _base_client._traced_headers(None)

    assert first is not None and second is not None
    assert first["traceparent"] != second["traceparent"]


def test_a_request_is_untouched_when_otel_is_absent() -> None:
    with mock.patch.object(_base_client, "_otel_inject", None):
        assert _base_client._traced_headers(None) is None
        assert _base_client._traced_headers({"X-Caller": "v"}) == {"X-Caller": "v"}


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
