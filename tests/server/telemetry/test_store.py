"""The ClickHouse TelemetryStore adapter: query construction, parsing, config edge.

No live ClickHouse: every HTTP call goes through an ``httpx.MockTransport`` handler that
asserts the request shape (parameterized query, correct params) before returning a
canned JSONEachRow body shaped exactly like real ClickHouse output -- verified against a
live ClickHouse instance during this change (string-encoded UInt64, ``"YYYY-MM-DD
HH:MM:SS.nnnnnnnnn"`` timestamps, a combined attribute Map) rather than assumed.
"""

import json

import httpx
import pytest
from flowmesh_cli_stack.env_schema import STACK_ENV_SCHEMA

from server.config import TelemetryStoreConfig
from server.telemetry import build_telemetry_store
from server.telemetry.clickhouse import ClickHouseTelemetryStore, TelemetryStoreError
from server.telemetry.store import UnsupportedAggregateError
from shared.telemetry.ids import workflow_to_trace_id_int


def _config(url: str | None = "http://clickhouse.test:8123") -> TelemetryStoreConfig:
    return TelemetryStoreConfig(
        url=url,
        database="flowmesh",
        username="default",
        password="secret",
        timeout_sec=5.0,
    )


def _store(handler) -> ClickHouseTelemetryStore:
    client = httpx.Client(
        base_url="http://clickhouse.test:8123", transport=httpx.MockTransport(handler)
    )
    return ClickHouseTelemetryStore(_config(), client=client)


def test_fetch_trace_derives_the_same_trace_id_producers_use() -> None:
    workflow_id = "wfl-0102030405060708090a0b0c0d0e0f10"
    expected_trace_id = format(workflow_to_trace_id_int(workflow_id), "032x")
    seen_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.update(dict(request.url.params))
        row = {
            "TraceId": expected_trace_id,
            "SpanId": "0102030405060708",
            "ParentSpanId": "",
            "SpanName": "flowmesh.control.ledger_snapshot",
            "SpanKind": "SPAN_KIND_INTERNAL",
            "Timestamp": "2026-09-17 00:00:00.123456789",
            "Duration": "100000000",
            "StatusCode": "STATUS_CODE_OK",
            "ServiceName": "flowmesh-server",
            "ResourceAttributes": {"flowmesh.node_id": "nde-1"},
            "SpanAttributes": {
                "flowmesh.logical.workflow_id": workflow_id,
                "flowmesh.physical.stage": "ledger_snapshot",
                "flowmesh.gpu.index": "0",  # not logical or physical -- must be dropped
            },
        }
        return httpx.Response(200, text=json.dumps(row) + "\n")

    store = _store(handler)
    spans = store.fetch_trace(workflow_id)

    assert seen_params["param_trace_id"] == expected_trace_id
    assert seen_params["database"] == "flowmesh"
    assert len(spans) == 1
    span = spans[0]
    assert span.trace_id == expected_trace_id
    assert span.parent_span_id is None  # empty string normalizes to None
    assert span.duration_ns == 100_000_000
    assert span.logical == {"workflow_id": workflow_id}
    assert span.physical == {"stage": "ledger_snapshot"}
    assert "flowmesh.gpu.index" not in span.logical
    assert "flowmesh.gpu.index" not in span.physical
    assert span.resource == {"flowmesh.node_id": "nde-1"}
    assert span.start_time.isoformat().startswith("2026-09-17T00:00:00.123456")


def test_fetch_trace_empty_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    store = _store(handler)
    assert store.fetch_trace("wfl-nonexistent") == []


def test_fetch_trace_raises_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="table does not exist")

    store = _store(handler)
    with pytest.raises(TelemetryStoreError, match="table does not exist"):
        store.fetch_trace("wfl-x")


def test_aggregate_binds_the_metric_and_grouping_key_and_parses_buckets() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        rows = [
            {"group_value": "0", "stat_value": "1.5", "sample_count": "10"},
            {"group_value": "1", "stat_value": "0.2", "sample_count": "4"},
        ]
        body = "\n".join(json.dumps(r) for r in rows)
        return httpx.Response(200, text=body)

    store = _store(handler)
    buckets = store.aggregate(
        metric="flowmesh.gpu.utilization", group_by="flowmesh.gpu.index", stat="avg"
    )

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["param_metric"] == "flowmesh.gpu.utilization"
    assert params["param_group_by_key"] == "flowmesh.gpu.index"

    assert [b.group_value for b in buckets] == ["0", "1"]
    assert buckets[0].value == 1.5
    assert buckets[0].sample_count == 10
    assert buckets[0].stat == "avg"


def test_aggregate_histogram_sample_count_is_the_observation_total() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # What the histogram query returns: `sample_count` is the summed point count
        # of the bucket's series, not the number of rows they were spread over.
        row = {"group_value": "dispatch", "stat_value": "437.5", "sample_count": "5"}
        return httpx.Response(200, text=json.dumps(row))

    store = _store(handler)
    buckets = store.aggregate(
        metric="flowmesh.duration",
        group_by="flowmesh.physical.stage",
        stat="p95",
        kind="histogram",
    )
    assert buckets[0].sample_count == 5
    assert buckets[0].value == 437.5


def test_aggregate_unknown_kind_raises() -> None:
    store = _store(lambda request: httpx.Response(200, text=""))
    with pytest.raises(TelemetryStoreError, match="unknown metric kind"):
        store.aggregate(metric="m", group_by="g", kind="sum")  # type: ignore[arg-type]


def _refusing_store() -> ClickHouseTelemetryStore:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an unanswerable aggregate must not reach the store")

    return _store(handler)


def test_aggregate_refuses_histogram_min_and_max() -> None:
    for stat in ("min", "max"):
        with pytest.raises(UnsupportedAggregateError, match="bucket counts"):
            _refusing_store().aggregate(
                metric="flowmesh.duration",
                group_by="flowmesh.physical.stage",
                stat=stat,  # type: ignore[arg-type]
                kind="histogram",
            )


def test_unsupported_aggregate_stays_a_telemetry_store_error() -> None:
    assert issubclass(UnsupportedAggregateError, TelemetryStoreError)


def test_read_port_defaults_match_the_values_the_env_chain_declares(
    monkeypatch,
) -> None:
    """A deployment whose ``.env`` predates a variable must land on its declared value.

    The schema is what renders ``.env.example`` and documents the knob, so a code
    default that disagrees with it silently authenticates a fresh server differently
    from the one an operator reads about.
    """
    declared = {
        var.key: var.default
        for section in STACK_ENV_SCHEMA.sections
        for var in section.vars
        if var.key.startswith("SERVER_METRICS_CLICKHOUSE_")
    }
    for key in declared:
        monkeypatch.delenv(key, raising=False)

    config = TelemetryStoreConfig.from_env()

    assert config.database == declared["SERVER_METRICS_CLICKHOUSE_DATABASE"]
    assert config.username == declared["SERVER_METRICS_CLICKHOUSE_USERNAME"]
    assert config.password == declared["SERVER_METRICS_CLICKHOUSE_PASSWORD"]
    assert config.timeout_sec == float(
        declared["SERVER_METRICS_CLICKHOUSE_TIMEOUT_SEC"]
    )


def test_build_telemetry_store_returns_none_when_unconfigured() -> None:
    assert build_telemetry_store(_config(url=None)) is None


def test_build_telemetry_store_returns_adapter_when_configured() -> None:
    store = build_telemetry_store(_config())
    assert isinstance(store, ClickHouseTelemetryStore)
    store.close()
