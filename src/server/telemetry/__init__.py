"""Server-side read port over the telemetry store.

The OTel Collector (``cli/stack/.../otel-collector-config.yaml``) is the store's sole
writer. Everything in this package only ever reads: the ``TelemetryStore`` protocol, its
ClickHouse adapter, and the factory that builds one from environment configuration.
"""

from .clickhouse import ClickHouseTelemetryStore, build_telemetry_store
from .store import (
    AggregateBucket,
    AggregateStat,
    MetricPoint,
    SpanRow,
    TelemetryStore,
    TelemetryStoreConfig,
)

__all__ = [
    "AggregateBucket",
    "AggregateStat",
    "ClickHouseTelemetryStore",
    "MetricPoint",
    "SpanRow",
    "TelemetryStore",
    "TelemetryStoreConfig",
    "build_telemetry_store",
]
