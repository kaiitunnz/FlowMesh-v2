"""ClickHouse adapter for the ``TelemetryStore`` read port.

Queries the store over ClickHouse's plain HTTP interface -- no new dependency: the
server already carries ``httpx``, and the HTTP interface avoids pulling in a native
ClickHouse driver for what is a handful of read queries. Every query is parameterized
(ClickHouse's own ``{name:Type}`` binding, never string interpolation) so a
caller-supplied workflow id or attribute key cannot inject SQL.

The store's own schema is not namespace-split (``flowmesh_spans.SpanAttributes`` is one
combined map -- see ``cli/stack/.../clickhouse-init.sql`` for why the Collector's
ClickHouse exporter cannot write a schema that is). The logical/physical split the
``TelemetryStore`` protocol exposes happens here, at read time, by attribute-key prefix.

The two metric tables are shaped differently and are queried differently. A gauge row
holds one ``Value``. A histogram row holds ``Count``/``Sum``/``BucketCounts``/
``ExplicitBounds`` for a whole series, exported at cumulative temporality: every export
interval re-reports the series' running totals, so each interval's row subsumes the last
and adding rows up multiplies the answer by the number of intervals. A histogram query
therefore reduces each series to its latest point first, and only then combines series.
"""

import json
from datetime import datetime, timedelta

import httpx

from server.config import TelemetryStoreConfig
from shared.telemetry.semconv import LOGICAL_ATTRIBUTE_PREFIX, PHYSICAL_ATTRIBUTE_PREFIX

from .store import (
    AggregateBucket,
    AggregateStat,
    MetricKind,
    SpanRow,
    TelemetryStore,
    TelemetryStoreError,
    UnsupportedAggregateError,
)

_TRACES_TABLE = "flowmesh_spans"
_GAUGE_TABLE = "flowmesh_metrics_gauge"
_HISTOGRAM_TABLE = "flowmesh_metrics_histogram"

_GAUGE_STAT_EXPR: dict[AggregateStat, str] = {
    "count": "count()",
    "sum": "sum(Value)",
    "avg": "avg(Value)",
    "min": "min(Value)",
    "max": "max(Value)",
    "p50": "quantile(0.5)(Value)",
    "p95": "quantile(0.95)(Value)",
    "p99": "quantile(0.99)(Value)",
}

_HISTOGRAM_STAT_EXPR: dict[AggregateStat, str] = {
    "count": "total_count",
    "sum": "total_sum",
    "avg": "if(total_count = 0, 0, total_sum / total_count)",
}
_HISTOGRAM_QUANTILES: dict[AggregateStat, float] = {
    "p50": 0.5,
    "p95": 0.95,
    "p99": 0.99,
}


def _gauge_sql(stat: AggregateStat) -> str:
    stat_expr = _GAUGE_STAT_EXPR.get(stat)
    if stat_expr is None:
        raise TelemetryStoreError(f"unknown aggregate stat {stat!r}")
    return f"""
        SELECT
            Attributes[{{group_by_key:String}}] AS group_value,
            {stat_expr} AS stat_value,
            count() AS sample_count
        FROM {_GAUGE_TABLE}
        WHERE MetricName = {{metric:String}}
        GROUP BY group_value
        ORDER BY group_value
    """  # nosec B608 - table is a module constant, stat_expr a closed-map value


def _histogram_series_sql() -> str:
    """Each series reduced to its latest cumulative point, then combined per group.

    A series is one attribute set reported by one producer over one uninterrupted
    lifetime, so the key includes ``StartTimeUnix``: a producer restart opens a fresh
    cumulative run whose counts start again from zero, and keying on it keeps both runs'
    totals instead of letting the later run's smaller numbers win the ``argMax``.
    """
    return f"""
        SELECT
            group_value,
            sum(series_count) AS total_count,
            sum(series_sum) AS total_sum,
            sumForEach(series_buckets) AS bucket_counts,
            any(series_bounds) AS bounds
        FROM (
            SELECT
                Attributes[{{group_by_key:String}}] AS group_value,
                argMax(Count, TimeUnix) AS series_count,
                argMax(Sum, TimeUnix) AS series_sum,
                argMax(BucketCounts, TimeUnix) AS series_buckets,
                argMax(ExplicitBounds, TimeUnix) AS series_bounds
            FROM {_HISTOGRAM_TABLE}
            WHERE MetricName = {{metric:String}}
            GROUP BY
                group_value, ServiceName, ResourceAttributes, Attributes, StartTimeUnix
        )
        GROUP BY group_value
    """  # nosec B608 - table is a module constant; metric and group key are bound params


def _histogram_sql(stat: AggregateStat) -> str:
    stat_expr = _HISTOGRAM_STAT_EXPR.get(stat)
    if stat_expr is not None:
        return _histogram_point_sql(stat_expr)
    quantile = _HISTOGRAM_QUANTILES.get(stat)
    if quantile is not None:
        return _histogram_quantile_sql(quantile)
    if stat not in _GAUGE_STAT_EXPR:
        raise TelemetryStoreError(f"unknown aggregate stat {stat!r}")
    raise UnsupportedAggregateError(
        f"the histogram table cannot answer stat {stat!r}: a histogram point carries "
        "bucket counts rather than the observations behind them, and leaves its "
        "minimum and maximum columns unpopulated, so there is no value to report -- "
        "use avg or a percentile"
    )


def _histogram_point_sql(stat_expr: str) -> str:
    return f"""
        SELECT
            group_value,
            {stat_expr} AS stat_value,
            total_count AS sample_count
        FROM ({_histogram_series_sql()})
        ORDER BY group_value
    """  # nosec B608 - stat_expr is a closed-map value; the inner query is generated


def _histogram_quantile_sql(quantile: float) -> str:
    """The quantile's position inside the bucket that holds it, linearly interpolated.

    A bucket records only how many observations fell in it, so the answer is resolved to
    the width of that bucket and no finer. A quantile landing in the unbounded overflow
    bucket has no upper edge to interpolate towards and reports the largest explicit
    bound, which the true value is at or above.

    ``lower_bounds`` and ``cumulative_before`` are the bound and running count each
    prepended with a zero, so every index the expression takes is in range for any
    bucket layout, including an empty one -- ClickHouse rejects index 0 outright and
    evaluates both arms of an ``if``, so guarding by index arithmetic rather than by
    branch is what keeps a degenerate row from raising.
    """
    return f"""
        SELECT
            group_value,
            if(
                total_count = 0,
                0,
                if(
                    (bucket_index = 0) OR (bucket_index > length(bounds)),
                    lower_bounds[length(bounds) + 1],
                    lower_bounds[safe_index]
                        + (bounds[safe_index] - lower_bounds[safe_index])
                        * (target_rank - cumulative_before[safe_index])
                        / greatest(bucket_counts[safe_index], 1)
                )
            ) AS stat_value,
            total_count AS sample_count
        FROM (
            SELECT
                group_value,
                total_count,
                bounds,
                bucket_counts,
                arrayCumSum(arrayMap(c -> toFloat64(c), bucket_counts)) AS cumulative,
                arrayPushFront(bounds, 0.) AS lower_bounds,
                arrayPushFront(cumulative, 0.) AS cumulative_before,
                {quantile} * total_count AS target_rank,
                arrayFirstIndex(c -> c >= target_rank, cumulative) AS bucket_index,
                greatest(least(bucket_index, length(bounds)), 1) AS safe_index
            FROM ({_histogram_series_sql()})
        )
        ORDER BY group_value
    """  # nosec B608 - quantile is a closed-map float; the inner query is generated


def _trace_id_hex(workflow_id: str) -> str:
    from shared.telemetry.ids import workflow_to_trace_id_int

    return format(workflow_to_trace_id_int(workflow_id), "032x")


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TelemetryStoreError(f"expected a string column value, got {value!r}")
    return value


def _as_int(value: object) -> int:
    # ClickHouse's JSONEachRow encodes UInt64 columns as strings, so accept both.
    if isinstance(value, bool):
        raise TelemetryStoreError(f"expected an integer column value, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise TelemetryStoreError(
                f"expected an integer column value, got {value!r}"
            ) from exc
    raise TelemetryStoreError(f"expected an integer column value, got {value!r}")


def _as_float(value: object) -> float:
    # ClickHouse's JSONEachRow encodes Float64 columns as strings, so accept both.
    if isinstance(value, bool):
        raise TelemetryStoreError(f"expected a numeric column value, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise TelemetryStoreError(
                f"expected a numeric column value, got {value!r}"
            ) from exc
    raise TelemetryStoreError(f"expected a numeric column value, got {value!r}")


def _split_namespaces(
    attributes: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    logical: dict[str, str] = {}
    physical: dict[str, str] = {}
    for key, value in attributes.items():
        if key.startswith(LOGICAL_ATTRIBUTE_PREFIX):
            logical[key.removeprefix(LOGICAL_ATTRIBUTE_PREFIX)] = value
        elif key.startswith(PHYSICAL_ATTRIBUTE_PREFIX):
            physical[key.removeprefix(PHYSICAL_ATTRIBUTE_PREFIX)] = value
    return logical, physical


class ClickHouseTelemetryStore(TelemetryStore):
    """Read-only ClickHouse adapter. Never issues an INSERT."""

    def __init__(
        self, config: TelemetryStoreConfig, client: httpx.Client | None = None
    ) -> None:
        if not config.url:
            raise TelemetryStoreError("TelemetryStoreConfig.url is not set")
        self._database = config.database
        self._client = client or httpx.Client(
            base_url=config.url,
            timeout=config.timeout_sec,
            auth=(config.username, config.password) if config.username else None,
        )

    def close(self) -> None:
        self._client.close()

    def _query_rows(self, sql: str, params: dict[str, str]) -> list[dict[str, object]]:
        query_params = {f"param_{k}": v for k, v in params.items()}
        query_params["database"] = self._database
        try:
            response = self._client.post(
                "/", params=query_params, content=f"{sql}\nFORMAT JSONEachRow"
            )
        except httpx.HTTPError as exc:
            raise TelemetryStoreError(
                f"telemetry store at {self._client.base_url} is unreachable: {exc}"
            ) from exc
        if response.status_code != httpx.codes.OK:
            raise TelemetryStoreError(
                f"telemetry store query failed "
                f"({response.status_code}): {response.text}"
            )
        text = response.text.strip()
        if not text:
            return []
        return [json.loads(line) for line in text.splitlines()]

    def fetch_trace(self, workflow_id: str) -> list[SpanRow]:
        trace_id = _trace_id_hex(workflow_id)
        rows = self._query_rows(
            f"""
            SELECT
                TraceId, SpanId, ParentSpanId, SpanName, SpanKind,
                Timestamp, Duration, StatusCode, ServiceName,
                ResourceAttributes, SpanAttributes
            FROM {_TRACES_TABLE} FINAL
            WHERE TraceId = {{trace_id:String}}
            """,  # nosec B608 - table name is a module constant; trace_id is a bound param
            {"trace_id": trace_id},
        )
        spans: list[SpanRow] = []
        for row in rows:
            span_attrs = row["SpanAttributes"]
            if not isinstance(span_attrs, dict):
                raise TelemetryStoreError(
                    f"unexpected SpanAttributes shape: {span_attrs!r}"
                )
            resource_attrs = row["ResourceAttributes"]
            if not isinstance(resource_attrs, dict):
                raise TelemetryStoreError(
                    f"unexpected ResourceAttributes shape: {resource_attrs!r}"
                )
            logical, physical = _split_namespaces(span_attrs)
            start = datetime.fromisoformat(_as_str(row["Timestamp"]))
            duration_ns = _as_int(row["Duration"])
            spans.append(
                SpanRow(
                    trace_id=_as_str(row["TraceId"]),
                    span_id=_as_str(row["SpanId"]),
                    parent_span_id=_as_str(row["ParentSpanId"]) or None,
                    name=_as_str(row["SpanName"]),
                    kind=_as_str(row["SpanKind"]),
                    start_time=start,
                    end_time=start + timedelta(microseconds=duration_ns / 1000),
                    duration_ns=duration_ns,
                    status_code=_as_str(row["StatusCode"]),
                    service_name=_as_str(row["ServiceName"]),
                    logical=logical,
                    physical=physical,
                    resource={
                        _as_str(k): _as_str(v) for k, v in resource_attrs.items()
                    },
                )
            )
        return spans

    def aggregate(
        self,
        *,
        metric: str,
        group_by: str,
        stat: AggregateStat = "avg",
        kind: MetricKind = "gauge",
        workflow_id: str | None = None,
    ) -> list[AggregateBucket]:
        if kind == "histogram":
            sql = _histogram_sql(stat)
        elif kind == "gauge":
            sql = _gauge_sql(stat)
        else:
            raise TelemetryStoreError(f"unknown metric kind {kind!r}")
        if workflow_id is not None:
            raise UnsupportedAggregateError(
                "a workflow-scoped aggregate is not answerable: no metric this store "
                "holds carries a workflow id, so the filter can only ever match "
                "nothing -- read the workflow's span tree for per-workflow telemetry"
            )
        rows = self._query_rows(sql, {"metric": metric, "group_by_key": group_by})
        return [
            AggregateBucket(
                group_value=_as_str(row["group_value"]),
                stat=stat,
                value=_as_float(row["stat_value"]),
                sample_count=_as_int(row["sample_count"]),
            )
            for row in rows
        ]


def build_telemetry_store(config: TelemetryStoreConfig) -> TelemetryStore | None:
    """Build the configured ``TelemetryStore``, or ``None`` when unconfigured.

    Mirrors the house pattern for an optional subsystem (``resident/wiring.py``'s
    ``build_resident_capacity``): construction is a pure function of config, and the
    caller decides what an absent store means for its route.
    """
    if not config.url:
        return None
    return ClickHouseTelemetryStore(config)
