"""The telemetry knob every producer checks before emitting a span or metric."""

import os
from dataclasses import dataclass
from enum import StrEnum

from shared.utils.parsing import parse_bool_env, parse_float_env, parse_int_env


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
    otlp_timeout_sec: int = 10
    resource_sample_sec: int = 15

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
            sample_ratio=_sample_ratio(),
            otlp_endpoint=_otlp_endpoint(),
            otlp_timeout_sec=max(
                1, parse_int_env("SERVER_METRICS_OTLP_TIMEOUT_SEC", 10)
            ),
            resource_sample_sec=max(
                1, parse_int_env("SERVER_METRICS_RESOURCE_SAMPLE_SEC", 15)
            ),
        )


DISABLED_TELEMETRY_CONFIG = TelemetryConfig(
    level=TelemetryLevel.OFF,
    traces_enabled=False,
    metrics_enabled=False,
    sample_ratio=1.0,
    otlp_endpoint=None,
)
"""The config a process holds when telemetry is off, shared by every null twin."""


def _otlp_endpoint() -> str | None:
    """The configured collector endpoint, or ``None`` when export is not configured.

    The endpoint must name its scheme. Without one the OTLP exporter negotiates TLS
    against a collector that is almost always serving plaintext, and the export fails
    where nothing reports it -- on the exporter's own background thread.
    """
    endpoint = (os.getenv("SERVER_METRICS_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return None
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(
            "SERVER_METRICS_OTLP_ENDPOINT must start with http:// or https:// "
            f"(got {endpoint!r}); use http://<host>:4317 for a plaintext collector."
        )
    return endpoint


def _sample_ratio() -> float:
    """The configured trace sample ratio, refused outside ``[0.0, 1.0]``.

    Every process parses this independently, so a value one of them would clamp and
    another would reject is a deployment where the root and its workers disagree about
    which workflows are traced. Refusing at the edge keeps that disagreement impossible.
    """
    ratio = parse_float_env("SERVER_METRICS_TRACE_SAMPLE_RATIO", 1.0)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "SERVER_METRICS_TRACE_SAMPLE_RATIO must be between 0.0 and 1.0 "
            f"(got {ratio!r})."
        )
    return ratio
