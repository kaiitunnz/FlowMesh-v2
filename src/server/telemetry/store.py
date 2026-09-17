"""The ``TelemetryStore`` read port: protocol and row shapes.

A narrow, backend-agnostic surface over the telemetry store's two queries -- a
workflow's whole trace, and an aggregate over one metric. Backends implement the
protocol; callers (the trace/aggregate REST routes) depend on nothing more than it, so
swapping the backing store is an adapter change with no change to a producer or to the
CLI/SDK surface built on top.

Read-only: nothing here writes a span or a metric, and nothing here is a second ingest
path. The OTel Collector is the store's sole writer.

The read-side connection settings live in ``server.config.TelemetryStoreConfig`` (read
at the config edge and injected down); this module never reads the environment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

AggregateStat = Literal["count", "sum", "avg", "min", "max", "p50", "p95", "p99"]
MetricKind = Literal["gauge", "histogram"]


class TelemetryStoreError(RuntimeError):
    """A store could not answer a query.

    Part of the port, not of any one adapter: a caller programs against the protocol
    and handles this without knowing which store is behind it.
    """


@dataclass(frozen=True)
class SpanRow:
    """One span as stored, with its logical/physical attribute views already split.

    The split happens here -- in the read path -- rather than in the store's schema
    (the store's own attribute map is not namespace-split); see
    ``ClickHouseTelemetryStore`` for why.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time: datetime
    end_time: datetime
    duration_ns: int
    status_code: str
    service_name: str
    logical: dict[str, str] = field(default_factory=dict)
    physical: dict[str, str] = field(default_factory=dict)
    resource: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricPoint:
    metric_name: str
    timestamp: datetime
    value: float
    attributes: dict[str, str] = field(default_factory=dict)
    resource: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregateBucket:
    group_value: str
    stat: AggregateStat
    value: float
    sample_count: int


class TelemetryStore(Protocol):
    """Read-only port over the telemetry store.

    Never writes a span or a metric, and is never a second ingest path -- the Collector
    is the store's sole writer: the write path and the read path never touch.
    """

    def fetch_trace(self, workflow_id: str) -> list[SpanRow]:
        """Every span belonging to the workflow's trace, unordered.

        The caller assembles the span tree from ``parent_span_id``; this only resolves
        the flat row set a primary-key scan on the derived trace id produces.
        """
        ...

    def aggregate(
        self,
        *,
        metric: str,
        group_by: str,
        stat: AggregateStat = "avg",
        kind: MetricKind = "gauge",
        workflow_id: str | None = None,
    ) -> list[AggregateBucket]:
        """Aggregate one metric's datapoints, grouped by one attribute key.

        ``kind`` selects which physical metric table the point lives in (gauges land
        separately from the spanmetrics-derived histogram); callers that don't know a
        metric's kind ahead of time try ``"gauge"`` first, the default for every
        counter/gauge FlowMesh emits today.
        """
        ...
