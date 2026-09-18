"""The metric-aggregate queries, against the shape the Collector's exporter creates.

Both metric tables run with the ClickHouse exporter's ``create_schema: true`` (see
``otel-collector-config.yaml``), so the exporter -- not this repo -- owns their columns.
``_GAUGE_COLUMNS`` / ``_HISTOGRAM_COLUMNS`` are the column sets ``SHOW CREATE TABLE``
returns from a ClickHouse the pinned collector image (0.161.0) has written to, and every
column the generated SQL reads is checked against the table it reads it from: the two
tables share only some of their columns, and a gauge-only column named in a histogram
query is a run-time "Unknown identifier", not a wrong number.

The gauge and histogram tables differ in more than columns. A gauge row is one datapoint
carrying one ``Value``. A histogram row is a whole series at cumulative temporality --
the collector re-exports each series' running totals every interval -- so every
statistic has to reduce each series to its latest point before combining series, and the
tests below lock that reduction in as well.

Setting ``FLOWMESH_TEST_CLICKHOUSE_URL`` (plus the optional ``..._USERNAME`` /
``..._PASSWORD``) runs the arithmetic itself against a live ClickHouse: the tests build
their own tables from the locked columns, insert the cumulative points a collector
export produces, and assert the values that come back. Without it they skip, and the
offline tests above still hold the contract.
"""

import json
import os
import re
import uuid
from collections.abc import Iterator

import httpx
import pytest

from server.config import TelemetryStoreConfig
from server.telemetry.clickhouse import (
    ClickHouseTelemetryStore,
    _gauge_sql,
    _histogram_sql,
)
from server.telemetry.store import (
    AggregateStat,
    MetricKind,
    TelemetryStoreError,
    UnsupportedAggregateError,
)

_GAUGE_COLUMNS: dict[str, str] = {
    "ResourceAttributes": "Map(LowCardinality(String), String)",
    "ResourceSchemaUrl": "String",
    "ScopeName": "String",
    "ScopeVersion": "String",
    "ScopeAttributes": "Map(LowCardinality(String), String)",
    "ScopeDroppedAttrCount": "UInt32",
    "ScopeSchemaUrl": "String",
    "ServiceName": "LowCardinality(String)",
    "MetricName": "LowCardinality(String)",
    "MetricDescription": "String",
    "MetricUnit": "String",
    "Attributes": "Map(LowCardinality(String), String)",
    "StartTimeUnix": "DateTime",
    "TimeUnix": "DateTime",
    "Value": "Float64",
    "Flags": "UInt32",
}

_HISTOGRAM_COLUMNS: dict[str, str] = {
    "ResourceAttributes": "Map(LowCardinality(String), String)",
    "ResourceSchemaUrl": "String",
    "ScopeName": "String",
    "ScopeVersion": "String",
    "ScopeAttributes": "Map(LowCardinality(String), String)",
    "ScopeDroppedAttrCount": "UInt32",
    "ScopeSchemaUrl": "String",
    "ServiceName": "LowCardinality(String)",
    "MetricName": "LowCardinality(String)",
    "MetricDescription": "String",
    "MetricUnit": "String",
    "Attributes": "Map(LowCardinality(String), String)",
    "StartTimeUnix": "DateTime",
    "TimeUnix": "DateTime",
    "Count": "UInt64",
    "Sum": "Float64",
    "BucketCounts": "Array(UInt64)",
    "ExplicitBounds": "Array(Float64)",
    "Flags": "UInt32",
    "Min": "Float64",
    "Max": "Float64",
    "AggregationTemporality": "Int32",
}

_HISTOGRAM_STATS: tuple[AggregateStat, ...] = ("count", "sum", "avg", "p50", "p95")
_GAUGE_STATS: tuple[AggregateStat, ...] = (
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "p50",
    "p95",
    "p99",
)


def _columns_read(sql: str) -> set[str]:
    """Every store column the SQL names.

    A ``{name:Type}`` binding is stripped first so its type token is not mistaken for a
    column; what is left that starts with an upper-case letter followed by a lower-case
    one is a column reference -- SQL keywords here are upper-case throughout, and every
    function, table and alias is lower-case or camelCase.
    """
    return set(re.findall(r"\b[A-Z][a-z][A-Za-z0-9]*\b", re.sub(r"\{[^}]*\}", "", sql)))


@pytest.mark.parametrize("stat", _HISTOGRAM_STATS)
def test_histogram_query_reads_only_histogram_columns(stat: AggregateStat) -> None:
    unknown = _columns_read(_histogram_sql(stat)) - set(_HISTOGRAM_COLUMNS)
    assert not unknown, (
        f"stat {stat!r} reads {sorted(unknown)} from the histogram table, which the "
        "pinned exporter does not create there"
    )


@pytest.mark.parametrize("stat", _GAUGE_STATS)
def test_gauge_query_reads_only_gauge_columns(stat: AggregateStat) -> None:
    unknown = _columns_read(_gauge_sql(stat)) - set(_GAUGE_COLUMNS)
    assert not unknown


def test_no_histogram_query_reads_the_gauge_value_column() -> None:
    assert "Value" not in _HISTOGRAM_COLUMNS
    for stat in _HISTOGRAM_STATS:
        assert "Value" not in _columns_read(_histogram_sql(stat))


@pytest.mark.parametrize("stat", _HISTOGRAM_STATS)
def test_histogram_query_reduces_each_series_before_combining(
    stat: AggregateStat,
) -> None:
    sql = _histogram_sql(stat)
    assert "argMax(Count, TimeUnix)" in sql
    assert "argMax(Sum, TimeUnix)" in sql
    assert "argMax(BucketCounts, TimeUnix)" in sql
    # A cumulative point restates the series' totals, so adding rows up multiplies the
    # answer by the number of export intervals.
    assert re.search(r"sum\(\s*Count\s*\)", sql) is None
    assert re.search(r"sum\(\s*Sum\s*\)", sql) is None
    assert re.search(r"count\(\)", sql) is None
    # A producer restart opens a fresh cumulative run; keying series on its start keeps
    # the earlier run's totals instead of letting the restarted run's argMax win.
    assert "StartTimeUnix" in sql


def test_histogram_refuses_min_and_max() -> None:
    for stat in ("min", "max"):
        with pytest.raises(UnsupportedAggregateError, match="bucket counts"):
            _histogram_sql(stat)


def test_histogram_sample_count_is_observations_not_rows() -> None:
    for stat in _HISTOGRAM_STATS:
        assert "total_count AS sample_count" in _histogram_sql(stat)


def test_unknown_stat_is_not_an_unsupported_aggregate() -> None:
    for builder in (_gauge_sql, _histogram_sql):
        with pytest.raises(TelemetryStoreError, match="unknown aggregate stat") as exc:
            builder("median")  # type: ignore[arg-type]
        assert not isinstance(exc.value, UnsupportedAggregateError)


def test_generated_sql_binds_every_caller_supplied_value() -> None:
    for sql in (_gauge_sql("count"), _histogram_sql("count"), _histogram_sql("p95")):
        assert "{metric:String}" in sql
        assert "{group_by_key:String}" in sql


_LIVE_URL = os.getenv("FLOWMESH_TEST_CLICKHOUSE_URL")
_live = pytest.mark.skipif(
    not _LIVE_URL, reason="FLOWMESH_TEST_CLICKHOUSE_URL is not set"
)

# One collector export interval of a spanmetrics histogram: two stages, the buckets the
# collector config declares, at cumulative temporality. `dispatch` observed 6ms, 7ms and
# 300ms; `admission` observed 12ms and one 40s that lands in the unbounded overflow
# bucket. The second row of each pair is the next interval restating the same totals,
# and the `dispatch` pair at a later StartTimeUnix is the run a collector restart opens,
# adding observations of 6ms and 6ms.
_BOUNDS = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
_RUN_ONE = "2026-09-17 00:00:00"
_RUN_TWO = "2026-09-17 01:00:00"


def _histogram_row(
    stage: str, start: str, at: str, count: int, total: float, buckets: list[int]
) -> dict[str, object]:
    return {
        "ServiceName": "flowmesh-server",
        "MetricName": "flowmesh.duration",
        "ResourceAttributes": {"flowmesh.node_id": "nde-1"},
        "Attributes": {"flowmesh.physical.stage": stage},
        "StartTimeUnix": start,
        "TimeUnix": at,
        "Count": count,
        "Sum": total,
        "BucketCounts": buckets,
        "ExplicitBounds": _BOUNDS,
        "Min": 0.0,
        "Max": 0.0,
        "AggregationTemporality": 2,
    }


def _dispatch_buckets() -> list[int]:
    counts = [0] * (len(_BOUNDS) + 1)
    counts[1] = 2  # 6ms and 7ms, in (5, 10]
    counts[6] = 1  # 300ms, in (250, 500]
    return counts


def _admission_buckets() -> list[int]:
    counts = [0] * (len(_BOUNDS) + 1)
    counts[2] = 1  # 12ms, in (10, 25]
    counts[-1] = 1  # 40s, past the last bound
    return counts


def _restart_buckets() -> list[int]:
    counts = [0] * (len(_BOUNDS) + 1)
    counts[1] = 2  # 6ms and 6ms
    return counts


_HISTOGRAM_ROWS = (
    [
        _histogram_row("dispatch", _RUN_ONE, t, 3, 313.0, _dispatch_buckets())
        for t in ("2026-09-17 00:01:00", "2026-09-17 00:02:00")
    ]
    + [
        _histogram_row("admission", _RUN_ONE, t, 2, 40012.0, _admission_buckets())
        for t in ("2026-09-17 00:01:00", "2026-09-17 00:02:00")
    ]
    + [
        _histogram_row(
            "dispatch", _RUN_TWO, "2026-09-17 01:01:00", 2, 12.0, _restart_buckets()
        )
    ]
)

_GAUGE_ROWS = [
    {
        "ServiceName": "flowmesh-worker",
        "MetricName": "flowmesh.gpu.utilization",
        "ResourceAttributes": {"flowmesh.node_id": "nde-1"},
        "Attributes": {"flowmesh.gpu.index": index},
        "StartTimeUnix": _RUN_ONE,
        "TimeUnix": "2026-09-17 00:01:00",
        "Value": value,
        "Flags": 0,
    }
    for index, value in (("0", 10.0), ("0", 30.0), ("1", 50.0))
]


def _create_table(
    client: httpx.Client, database: str, table: str, columns: dict
) -> None:
    body = ", ".join(f"`{name}` {type_}" for name, type_ in columns.items())
    client.post(
        "/",
        content=(
            f"CREATE TABLE {database}.{table} ({body}) "
            "ENGINE = MergeTree ORDER BY (ServiceName, MetricName, TimeUnix)"
        ),
    ).raise_for_status()


def _insert(
    client: httpx.Client, database: str, table: str, rows: list[dict[str, object]]
) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    client.post(
        "/",
        content=f"INSERT INTO {database}.{table} FORMAT JSONEachRow\n{payload}",
    ).raise_for_status()


@pytest.fixture
def live_store() -> Iterator[ClickHouseTelemetryStore]:
    database = f"flowmesh_test_{uuid.uuid4().hex[:12]}"
    admin = httpx.Client(
        base_url=_LIVE_URL or "",
        timeout=30.0,
        auth=(
            os.getenv("FLOWMESH_TEST_CLICKHOUSE_USERNAME", "default"),
            os.getenv("FLOWMESH_TEST_CLICKHOUSE_PASSWORD", ""),
        ),
    )
    admin.post("/", content=f"CREATE DATABASE {database}").raise_for_status()
    try:
        _create_table(admin, database, "flowmesh_metrics_gauge", _GAUGE_COLUMNS)
        _create_table(admin, database, "flowmesh_metrics_histogram", _HISTOGRAM_COLUMNS)
        _insert(admin, database, "flowmesh_metrics_gauge", _GAUGE_ROWS)
        _insert(admin, database, "flowmesh_metrics_histogram", _HISTOGRAM_ROWS)
        store = ClickHouseTelemetryStore(
            TelemetryStoreConfig(
                url=_LIVE_URL,
                database=database,
                username=os.getenv("FLOWMESH_TEST_CLICKHOUSE_USERNAME", "default"),
                password=os.getenv("FLOWMESH_TEST_CLICKHOUSE_PASSWORD", ""),
                timeout_sec=30.0,
            )
        )
        try:
            yield store
        finally:
            store.close()
    finally:
        admin.post("/", content=f"DROP DATABASE IF EXISTS {database}")
        admin.close()


def _values(
    store: ClickHouseTelemetryStore, stat: AggregateStat, kind: MetricKind = "histogram"
) -> dict[str, tuple[float, int]]:
    return {
        bucket.group_value: (round(bucket.value, 6), bucket.sample_count)
        for bucket in store.aggregate(
            metric=(
                "flowmesh.duration"
                if kind == "histogram"
                else "flowmesh.gpu.utilization"
            ),
            group_by=(
                "flowmesh.physical.stage"
                if kind == "histogram"
                else "flowmesh.gpu.index"
            ),
            stat=stat,
            kind=kind,
        )
    }


@_live
def test_live_histogram_count_and_sum_survive_cumulative_restatement(
    live_store,
) -> None:
    # Five dispatch observations reported across three cumulative rows, two of which
    # restate the first run's totals: a row-wise sum would answer 8, a row count 3.
    assert _values(live_store, "count") == {"dispatch": (5.0, 5), "admission": (2.0, 2)}
    assert _values(live_store, "sum") == {
        "dispatch": (325.0, 5),
        "admission": (40012.0, 2),
    }


@_live
def test_live_histogram_avg_is_the_mean_of_the_observations(
    live_store: ClickHouseTelemetryStore,
) -> None:
    assert _values(live_store, "avg") == {
        "dispatch": (65.0, 5),
        "admission": (20006.0, 2),
    }


@_live
def test_live_histogram_quantile_interpolates_inside_its_bucket(
    live_store: ClickHouseTelemetryStore,
) -> None:
    # Five dispatch observations: four in (5, 10] and one in (250, 500]. The median's
    # rank of 2.5 lands 2.5/4 of the way into the first of those.
    assert _values(live_store, "p50")["dispatch"] == (8.125, 5)
    # The 95th's rank of 4.75 lands three quarters into the single-observation bucket.
    assert _values(live_store, "p95")["dispatch"] == (437.5, 5)


@_live
def test_live_histogram_quantile_past_the_last_bound_reports_that_bound(
    live_store: ClickHouseTelemetryStore,
) -> None:
    # One of admission's two observations is past the largest explicit bound, so the
    # 95th falls in the unbounded bucket, which has no upper edge to interpolate to.
    assert _values(live_store, "p95")["admission"] == (max(_BOUNDS), 2)


@_live
def test_live_gauge_statistics_are_unchanged(
    live_store: ClickHouseTelemetryStore,
) -> None:
    assert _values(live_store, "avg", kind="gauge") == {"0": (20.0, 2), "1": (50.0, 1)}
    assert _values(live_store, "min", kind="gauge") == {"0": (10.0, 2), "1": (50.0, 1)}
    assert _values(live_store, "max", kind="gauge") == {"0": (30.0, 2), "1": (50.0, 1)}
    assert _values(live_store, "count", kind="gauge") == {"0": (2.0, 2), "1": (1.0, 1)}


@_live
def test_live_empty_metric_returns_no_buckets(
    live_store: ClickHouseTelemetryStore,
) -> None:
    assert live_store.aggregate(metric="absent", group_by="x", kind="histogram") == []
