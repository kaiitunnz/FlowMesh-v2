"""Contract §4.2 H3: ``ServeRequestEnvelope.digest()`` is identical whether telemetry
is on or off.

The envelope's headers are digested into the claim fence (``freeze_request_envelope``),
and ``traceparent`` is on neither strip list, so writing one there would land inside the
digest and break the resident path -- nothing would fail locally, which is exactly why
this needs a test rather than a read of the code. Serve-request tracing takes its
context from the serve trace root and rides ``RelayFrame.tp`` outside the digested
envelope (contract §1.1), so the serve path must never inject, rewrite, or strip a
``traceparent`` in ``ServeRequestEnvelope.headers``.

This drives ``freeze_request_envelope`` -- exactly what ``routers/v1/serve.py``'s
``serve_gated`` calls with the client's raw header list -- once with no active span
(telemetry off) and once inside a real, active OTel span with a real ``traceparent``
available to inject (telemetry on, ambient context present). If anything in the serve
path ever called ``inject_ambient_traceparent`` on the header list before freezing, the
"on" case would carry an extra header the "off" case does not, and the digests would
differ. They must not.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from shared.resident.envelope import freeze_request_envelope
from shared.telemetry.propagation import inject_ambient_traceparent

_HEADERS = [
    ("Content-Type", "application/json"),
    ("Authorization", "Bearer client-token"),
    ("X-Custom", "kept"),
]
_BODY = b'{"model": "m", "messages": []}'


def _freeze():
    return freeze_request_envelope(
        method="POST",
        upstream_path="v1/chat/completions",
        query="stream=1",
        headers=_HEADERS,
        body=_BODY,
    )


def test_digest_is_identical_with_no_active_span() -> None:
    off = _freeze()
    # A second call with no telemetry involved at all -- the baseline.
    baseline = _freeze()
    assert off.digest() == baseline.digest()
    assert off.headers == baseline.headers


def test_digest_is_identical_inside_a_real_active_span() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    off = _freeze()

    with tracer.start_as_current_span("flowmesh.boundary"):
        # A real traceparent is genuinely available to inject here -- confirm it would
        # actually inject into an unrelated header dict, so this test would catch a
        # regression rather than passing by construction.
        probe: dict[str, str] = {}
        inject_ambient_traceparent(probe)
        assert (
            "traceparent" in probe
        ), "no ambient context available; this test cannot prove anything"

        on = _freeze()

    assert on.digest() == off.digest()
    assert on.headers == off.headers
    assert not any(name.lower() == "traceparent" for name, _ in on.headers)


def test_freeze_never_reads_or_writes_a_traceparent_header() -> None:
    # A client-sent traceparent is an ordinary header like any other -- forwarded
    # verbatim, never added, never stripped, never treated specially.
    with_client_tp = freeze_request_envelope(
        method="POST",
        upstream_path="v1/chat/completions",
        query="",
        headers=[
            *_HEADERS,
            ("traceparent", "00-11111111111111111111111111111111-2222222222222222-01"),
        ],
        body=_BODY,
    )
    without = _freeze()
    assert with_client_tp.digest() != without.digest()
    assert (
        "traceparent",
        "00-11111111111111111111111111111111-2222222222222222-01",
    ) in (with_client_tp.headers)
