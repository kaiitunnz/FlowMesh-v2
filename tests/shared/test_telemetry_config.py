"""TelemetryConfig.emits: level ordering, never true when off."""

import pytest

from shared.telemetry.config import TelemetryConfig, TelemetryLevel


def _config(level: TelemetryLevel) -> TelemetryConfig:
    return TelemetryConfig(
        level=level,
        traces_enabled=True,
        metrics_enabled=True,
        sample_ratio=1.0,
        otlp_endpoint=None,
    )


@pytest.mark.parametrize(
    "minimum", [TelemetryLevel.COARSE, TelemetryLevel.FINE, TelemetryLevel.FULL]
)
def test_off_never_emits_above_off(minimum: TelemetryLevel) -> None:
    assert _config(TelemetryLevel.OFF).emits(minimum) is False


@pytest.mark.parametrize(
    ("level", "minimum", "expected"),
    [
        (TelemetryLevel.OFF, TelemetryLevel.OFF, True),
        (TelemetryLevel.COARSE, TelemetryLevel.COARSE, True),
        (TelemetryLevel.COARSE, TelemetryLevel.FINE, False),
        (TelemetryLevel.FINE, TelemetryLevel.COARSE, True),
        (TelemetryLevel.FINE, TelemetryLevel.FULL, False),
        (TelemetryLevel.FULL, TelemetryLevel.FULL, True),
        (TelemetryLevel.FULL, TelemetryLevel.COARSE, True),
    ],
)
def test_emits_respects_level_ordering(
    level: TelemetryLevel, minimum: TelemetryLevel, expected: bool
) -> None:
    assert _config(level).emits(minimum) is expected
