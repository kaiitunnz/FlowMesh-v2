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
)

_TRACES_TABLE = "flowmesh_spans"
_METRIC_TABLES: dict[MetricKind, str] = {
    "gauge": "flowmesh_metrics_gauge",
    "histogram": "flowmesh_metrics_histogram",
}
_STAT_EXPR: dict[AggregateStat, str] = {
    "count": "count()",
    "sum": "sum(Value)",
    "avg": "avg(Value)",
    "min": "min(Value)",
    "max": "max(Value)",
    "p50": "quantile(0.5)(Value)",
    "p95": "quantile(0.95)(Value)",
    "p99": "quantile(0.99)(Value)",
}


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
        table = _METRIC_TABLES.get(kind)
        if table is None:
            raise TelemetryStoreError(f"unknown metric kind {kind!r}")
        stat_expr = _STAT_EXPR.get(stat)
        if stat_expr is None:
            raise TelemetryStoreError(f"unknown aggregate stat {stat!r}")
        where = ["MetricName = {metric:String}"]
        params = {"metric": metric, "group_by_key": group_by}
        if workflow_id is not None:
            where.append("Attributes[{workflow_key:String}] = {workflow_value:String}")
            params["workflow_key"] = LOGICAL_ATTRIBUTE_PREFIX + "workflow_id"
            params["workflow_value"] = workflow_id
        rows = self._query_rows(
            f"""
            SELECT
                Attributes[{{group_by_key:String}}] AS group_value,
                {stat_expr} AS stat_value,
                count() AS sample_count
            FROM {table}
            WHERE {" AND ".join(where)}
            GROUP BY group_value
            ORDER BY group_value
            """,  # nosec B608 - table and stat come from closed maps; values are bound params
            params,
        )
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
