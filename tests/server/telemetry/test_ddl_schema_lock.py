"""Schema-lock: the hand-authored traces DDL vs. the pinned Collector exporter version.

``clickhouse-init.sql`` runs with the ClickHouse exporter's ``create_schema: false`` for
the traces table (see that file's header comment for why: a trace-id-leading
``ReplacingMergeTree`` recovers both the primary-key-scan and dedup properties the
exporter's own native schema does not). ``create_schema: false`` means the exporter's
own
INSERT statement -- not this file -- is the authority on column names and types; this
file only supplies ENGINE/ORDER BY/TTL. So the DDL's columns are locked here against the
exact set fetched from ``open-telemetry/opentelemetry-collector-contrib`` at the pinned
image tag (verified byte-identical from v0.129.0 through v0.161.0, the pinned tag, at
the
time of this change) -- a future collector version bump that changes the exporter's
column set fails this test loudly instead of failing INSERTs at 3am.
"""

import re

import yaml
from flowmesh_cli.core.assets import asset_path

_COMPOSE_PATH = asset_path("flowmesh_cli_stack.assets", "compose.yml")
_DDL_PATH = asset_path("flowmesh_cli_stack.assets", "clickhouse-init.sql")

# Top-level columns of the pinned exporter's native `traces_table.sql`, in declared
# order. Nested (`Events`, `Links`) columns are locked by name only -- their subfields
# are addressed by the exporter's INSERT as flattened `Events.Timestamp` etc., which is
# unaffected by anything this DDL controls (ENGINE/ORDER BY/TTL).
_EXPECTED_COLUMNS = [
    ("Timestamp", "DateTime64(9)"),
    ("TraceId", "String"),
    ("SpanId", "String"),
    ("ParentSpanId", "String"),
    ("TraceState", "String"),
    ("SpanName", "LowCardinality(String)"),
    ("SpanKind", "LowCardinality(String)"),
    ("ServiceName", "LowCardinality(String)"),
    ("ResourceAttributes", "Map(LowCardinality(String), String)"),
    ("ScopeName", "String"),
    ("ScopeVersion", "String"),
    ("SpanAttributes", "Map(LowCardinality(String), String)"),
    ("Duration", "UInt64"),
    ("StatusCode", "LowCardinality(String)"),
    ("StatusMessage", "String"),
    ("Events", "Nested"),
    ("Links", "Nested"),
]

_PINNED_IMAGE_TAG = "0.161.0"


def _pinned_collector_image_tag() -> str:
    doc = yaml.safe_load(_COMPOSE_PATH.read_text())
    image = doc["services"]["otel_collector"]["image"]
    assert image.startswith("otel/opentelemetry-collector-contrib:"), image
    return image.rsplit(":", 1)[1]


def _parse_top_level_columns(ddl_text: str) -> list[tuple[str, str]]:
    """Extract (name, type-or-'Nested') for each top-level DDL column.

    Walks paren depth so a `Nested(...)` block's own subfields, and each `INDEX ...`
    clause, are skipped rather than mistaken for top-level columns.
    """
    start = ddl_text.index("CREATE TABLE")
    open_paren = ddl_text.index("(", start)
    depth = 0
    body_start = None
    body_end = None
    for i in range(open_paren, len(ddl_text)):
        if ddl_text[i] == "(":
            if depth == 0 and body_start is None:
                body_start = i + 1
            depth += 1
        elif ddl_text[i] == ")":
            depth -= 1
            if depth == 0:
                body_end = i
                break
    assert body_start is not None and body_end is not None, "unbalanced parens in DDL"
    body = ddl_text[body_start:body_end]

    columns: list[tuple[str, str | None]] = []
    depth = 0
    for raw_line in body.splitlines():
        line = raw_line.strip().rstrip(",")
        opens = line.count("(")
        closes = line.count(")")
        if depth == 0 and line and not line.startswith("INDEX"):
            match = re.match(r"^(\w+)\s+(\w+)", line)
            if match:
                name, type_head = match.group(1), match.group(2)
                columns.append((name, "Nested" if type_head == "Nested" else None))
        depth += opens - closes
    # Fill in full type text for non-nested columns via a second, simpler pass.
    resolved: list[tuple[str, str]] = []
    for name, placeholder in columns:
        if placeholder == "Nested":
            resolved.append((name, "Nested"))
            continue
        m = re.search(rf"^\s*{re.escape(name)}\s+(.+?)\s+CODEC", body, re.MULTILINE)
        assert m, f"could not resolve type for column {name}"
        resolved.append((name, m.group(1).strip()))
    return resolved


def test_pinned_collector_image_tag_matches_the_verified_version() -> None:
    assert _pinned_collector_image_tag() == _PINNED_IMAGE_TAG, (
        "compose.yml's otel_collector image tag moved; re-verify clickhouse-init.sql's "
        "column set against the new tag's traces_table.sql before updating this test"
    )


def test_traces_ddl_columns_match_the_pinned_exporter_version() -> None:
    actual = _parse_top_level_columns(_DDL_PATH.read_text())
    assert actual == _EXPECTED_COLUMNS


def test_traces_table_name_matches_the_collector_config() -> None:
    collector_config = asset_path(
        "flowmesh_cli_stack.assets", "otel-collector-config.yaml"
    ).read_text()
    doc = yaml.safe_load(collector_config)
    assert (
        doc["exporters"]["clickhouse/traces"]["traces_table_name"] == "flowmesh_spans"
    )
    assert "CREATE TABLE IF NOT EXISTS flowmesh.flowmesh_spans" in _DDL_PATH.read_text()
    assert doc["exporters"]["clickhouse/traces"]["create_schema"] is False
