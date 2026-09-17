"""The telemetry knob every producer checks before emitting a span or metric."""

import os
from dataclasses import dataclass
from enum import StrEnum

from shared.utils.parsing import parse_bool_env, parse_float_env


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

    @staticmethod
    def from_env() -> "TelemetryConfig":
        """Parse the ``SERVER_METRICS_*`` telemetry vars, shared by every process.

        Root, supervisor and worker all read this one parser so they agree on the
        level: two independent copies of the level default and validation is how they
        would end up disagreeing.
        """
        level_raw = (
            (os.getenv("SERVER_METRICS_TELEMETRY_LEVEL") or TelemetryLevel.OFF.value)
            .strip()
            .lower()
        )
        try:
            level = TelemetryLevel(level_raw)
        except ValueError as exc:
            allowed = ", ".join(value.value for value in TelemetryLevel)
            raise SystemExit(
                f"SERVER_METRICS_TELEMETRY_LEVEL must be one of: {allowed} "
                f"(got {level_raw!r})"
            ) from exc
        return TelemetryConfig(
            level=level,
            traces_enabled=parse_bool_env("SERVER_METRICS_TRACES_ENABLED", True),
            metrics_enabled=parse_bool_env("SERVER_METRICS_METRICS_ENABLED", True),
            sample_ratio=parse_float_env("SERVER_METRICS_TRACE_SAMPLE_RATIO", 1.0),
            otlp_endpoint=(os.getenv("SERVER_METRICS_OTLP_ENDPOINT") or "").strip()
            or None,
        )
