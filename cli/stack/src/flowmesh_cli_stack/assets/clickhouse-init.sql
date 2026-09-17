-- Telemetry store schema for the OTel Collector's ClickHouse exporter.
--
-- The exporter is configured with `create_schema: false` for the traces table (see
-- otel-collector-config.yaml), so this DDL is the exporter's actual schema authority for
-- flowmesh_spans. Column names and types are copied verbatim from the exporter's own
-- native traces DDL template (open-telemetry/opentelemetry-collector-contrib, exporter/
-- clickhouseexporter/internal/sqltemplates/traces_table.sql, verified byte-identical
-- (column names, types and order) from the exporter's v0.129.0 release through its
-- v0.161.0 release, which is the collector image tag compose.yml pins -- the exporter's
-- INSERT statement addresses columns by name and fails if they drift. Only ENGINE and
-- ORDER BY are changed from the native template: a trace-id-leading key is what makes a
-- lookup by workflow id a primary-key scan, and ReplacingMergeTree is only a safe dedup
-- backstop when paired with a key that uniquely identifies a span -- the native
-- (ServiceName, SpanName, toDateTime(Timestamp)) key does not (two distinct same-second,
-- same-name spans from the same service would collapse into one; verified against a real
-- ClickHouse instance during this change -- see tests/server/telemetry).
--
-- The gauge and histogram metrics tables are NOT defined here: they run with
-- `create_schema: true` and the exporter creates and owns their schema, since nothing
-- about them needs the same override.
--
-- Applying this file: the bundled `clickhouse` compose service runs it automatically via
-- `docker-entrypoint-initdb.d`. Against an external / bring-your-own ClickHouse, apply it
-- once with `clickhouse-client --multiquery < clickhouse-init.sql` before pointing the
-- Collector's `TELEMETRY_CLICKHOUSE_DSN` at it.

CREATE DATABASE IF NOT EXISTS flowmesh;

CREATE TABLE IF NOT EXISTS flowmesh.flowmesh_spans (
    Timestamp DateTime64(9) CODEC(Delta, ZSTD(1)),
    TraceId String CODEC(ZSTD(1)),
    SpanId String CODEC(ZSTD(1)),
    ParentSpanId String CODEC(ZSTD(1)),
    TraceState String CODEC(ZSTD(1)),
    SpanName LowCardinality(String) CODEC(ZSTD(1)),
    SpanKind LowCardinality(String) CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    SpanAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    Duration UInt64 CODEC(ZSTD(1)),
    StatusCode LowCardinality(String) CODEC(ZSTD(1)),
    StatusMessage String CODEC(ZSTD(1)),
    Events Nested (
        Timestamp DateTime64(9),
        Name LowCardinality(String),
        Attributes Map(LowCardinality(String), String)
    ) CODEC(ZSTD(1)),
    Links Nested (
        TraceId String,
        SpanId String,
        TraceState String,
        Attributes Map(LowCardinality(String), String)
    ) CODEC(ZSTD(1)),
    INDEX idx_trace_id TraceId TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_span_attr_key mapKeys(SpanAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_span_attr_value mapValues(SpanAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_duration Duration TYPE minmax GRANULARITY 1
) ENGINE = ReplacingMergeTree
PARTITION BY toDate(Timestamp)
ORDER BY (TraceId, Timestamp, SpanId)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;
