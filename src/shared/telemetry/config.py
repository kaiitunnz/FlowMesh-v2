"""The telemetry knob every producer checks before emitting a span or metric."""

from dataclasses import dataclass
from enum import StrEnum


class TelemetryLevel(StrEnum):
    OFF = "off"
    COARSE = "coarse"
    FINE = "fine"
    FULL = "full"


_LEVEL_ORDER: dict[TelemetryLevel, int] = {
    TelemetryLevel.OFF: 0,
    TelemetryLevel.COARSE: 1,
    TelemetryLevel.FINE: 2,
    TelemetryLevel.FULL: 3,
}


@dataclass(frozen=True)
class TelemetryConfig:
    level: TelemetryLevel
    traces_enabled: bool
    metrics_enabled: bool
    sample_ratio: float
    otlp_endpoint: str | None

    def emits(self, minimum: TelemetryLevel) -> bool:
        """Whether this config's level is at least as verbose as ``minimum``.

        A call site gates actual emission on ``config.traces_enabled and
        config.emits(...)`` (or the ``metrics_enabled`` twin), never on this alone, so
        ``off`` still constructs no provider even though ``off.emits(off)`` is true.
        """
        return _LEVEL_ORDER[self.level] >= _LEVEL_ORDER[minimum]
