"""The server's own telemetry surfaces: the control-plane spans and the store read port.

``tracing`` is the write side, opening a span at each control-plane stage. The read side
is the ``TelemetryStore`` protocol, its ClickHouse adapter, and the factory that builds
one from an injected ``server.config.TelemetryStoreConfig``; the OTel Collector
(``cli/stack/.../otel-collector-config.yaml``) is the store's sole writer.
"""

from .clickhouse import ClickHouseTelemetryStore, build_telemetry_store
from .store import (
    AggregateBucket,
    AggregateStat,
    MetricPoint,
    SpanRow,
    TelemetryStore,
    TelemetryStoreError,
)
from .tracing import (
    NULL_CONTROL_TRACER,
    ControlPlaneTracer,
    format_traceparent,
    serve_trace_id_int,
)

__all__ = [
    "AggregateBucket",
    "AggregateStat",
    "ClickHouseTelemetryStore",
    "ControlPlaneTracer",
    "MetricPoint",
    "NULL_CONTROL_TRACER",
    "SpanRow",
    "TelemetryStore",
    "TelemetryStoreError",
    "build_telemetry_store",
    "format_traceparent",
    "serve_trace_id_int",
]
