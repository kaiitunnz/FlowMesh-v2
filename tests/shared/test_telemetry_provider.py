"""build_tracer / build_meter: off constructs no SDK provider and allocates nothing.

The null twin must return the *same* context-manager object on every call — not merely
something falsy — since a fresh object per call is exactly the allocation the
zero-overhead ``off`` discipline forbids.
"""

import pytest
from opentelemetry.metrics import NoOpMeter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider

from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.telemetry.provider import build_meter, build_tracer


def _config(
    level: TelemetryLevel = TelemetryLevel.OFF,
    *,
    traces_enabled: bool = True,
    metrics_enabled: bool = True,
) -> TelemetryConfig:
    return TelemetryConfig(
        level=level,
        traces_enabled=traces_enabled,
        metrics_enabled=metrics_enabled,
        sample_ratio=1.0,
        otlp_endpoint=None,
    )


def test_off_constructs_no_tracer_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_init(*args: object, **kwargs: object) -> None:
        raise AssertionError("TracerProvider must not be constructed when off")

    monkeypatch.setattr(TracerProvider, "__init__", _fail_init)

    build_tracer(_config(TelemetryLevel.OFF), {})


def test_off_constructs_no_meter_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_init(*args: object, **kwargs: object) -> None:
        raise AssertionError("MeterProvider must not be constructed when off")

    monkeypatch.setattr(MeterProvider, "__init__", _fail_init)

    build_meter(_config(TelemetryLevel.OFF), {})


def test_traces_disabled_returns_null_tracer_even_at_full_level() -> None:
    tracer = build_tracer(_config(TelemetryLevel.FULL, traces_enabled=False), {})

    first = tracer.start_as_current_span("a")
    second = tracer.start_as_current_span("b")

    assert first is second


def test_null_tracer_reuses_one_context_manager_across_many_calls() -> None:
    tracer = build_tracer(_config(TelemetryLevel.OFF), {})

    contexts = {tracer.start_as_current_span(f"span-{i}") for i in range(1000)}

    assert len(contexts) == 1


def test_null_tracer_start_span_returns_invalid_span_without_recording() -> None:
    tracer = build_tracer(_config(TelemetryLevel.OFF), {})

    span = tracer.start_span("noop")

    assert span.is_recording() is False


def test_null_meter_is_the_shared_no_op_meter() -> None:
    meter = build_meter(_config(TelemetryLevel.OFF), {})

    assert isinstance(meter, NoOpMeter)


def test_enabled_tracer_builds_a_real_sdk_tracer_provider() -> None:
    tracer = build_tracer(_config(TelemetryLevel.COARSE), {"service.name": "test"})

    with tracer.start_as_current_span("real-span") as span:
        assert span.is_recording() is True
