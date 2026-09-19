"""Shared OpenTelemetry primitives: identity, config, and the provider factory.

Attribute keys and span names live in :mod:`shared.telemetry.semconv` and are imported
from there directly rather than re-exported here, since the set is large.
"""

from .config import TelemetryConfig, TelemetryLevel
from .ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from .provider import build_meter, build_tracer

__all__ = [
    "SpanIdKind",
    "TelemetryConfig",
    "TelemetryLevel",
    "build_meter",
    "build_tracer",
    "derived_span_id",
    "workflow_to_trace_id_int",
]
