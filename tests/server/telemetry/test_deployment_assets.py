"""Deployment assets: the telemetry Compose profile, and the write/read ownership split.

Static structural checks only -- no Docker dependency, so this runs in the normal unit
suite. The full pipeline (Collector startup, DDL application, dedup semantics,
allow-list
enforcement against a live ClickHouse) was verified interactively against real images
during this change; these tests guard the structural properties that verification can't
re-run on every commit.
"""

import re

import yaml
from flowmesh_cli.core.assets import asset_path

_COMPOSE_PATH = asset_path("flowmesh_cli_stack.assets", "compose.yml")
_COLLECTOR_CONFIG_PATH = asset_path(
    "flowmesh_cli_stack.assets", "otel-collector-config.yaml"
)
_CLICKHOUSE_INIT_PATH = asset_path("flowmesh_cli_stack.assets", "clickhouse-init.sql")

_CORE_SERVICES = {"redis_control", "redis_telemetry", "server"}
_TELEMETRY_SERVICES = {"otel_collector", "clickhouse"}


def _load_compose() -> dict:
    return yaml.safe_load(_COMPOSE_PATH.read_text())


def test_telemetry_services_are_gated_behind_the_telemetry_profile() -> None:
    doc = _load_compose()
    for name in _TELEMETRY_SERVICES:
        assert doc["services"][name]["profiles"] == ["telemetry"]


def test_core_services_do_not_reference_the_telemetry_profile() -> None:
    doc = _load_compose()
    for name in _CORE_SERVICES:
        service = doc["services"][name]
        assert "telemetry" not in service.get(
            "profiles", []
        ), f"{name} must stay reachable under the default (root) profile alone"


def test_core_services_unchanged_by_the_telemetry_addition() -> None:
    """The `server` service's own config is untouched -- only new services were added.

    A prior compose.yml revision is not available in this test's scope, so this asserts
    the documented constraint directly: the server service carries no CLICKHOUSE-named
    or
    the new TELEMETRY_CLICKHOUSE_*-named environment variable, which is what a snuck-in
    cross-boundary edit would look like. (``REDIS_TELEMETRY_URL`` legitimately predates
    this change -- the Redis telemetry pub/sub channel, unrelated to OTel telemetry --
    so
    a bare "TELEMETRY" substring check would false-positive on it.)
    """
    doc = _load_compose()
    server_env = doc["services"]["server"].get("environment", {})
    assert not any("CLICKHOUSE" in k or k.startswith("TELEMETRY_") for k in server_env)


def test_clickhouse_volume_is_persistent() -> None:
    doc = _load_compose()
    assert "clickhouse_data" in doc["volumes"]
    assert "clickhouse_data" in doc["services"]["clickhouse"]["volumes"][0]


def test_collector_and_store_configs_are_compose_file_relative_assets() -> None:
    """Both configs resolve next to compose.yml, not to a user-CWD-relative path.

    Verified empirically during this change: a `./relative` compose volume/config path
    resolves against the *compose file's own directory*, not the invoking shell's CWD --
    unlike `worker_config.yaml`, which opts into CWD-anchoring via `STACK_PATH_KEYS`.
    These two are static infra config shipped with the distribution, not per-deployment
    user content, so compose-file-relative is the correct (and simpler) choice.
    """
    doc = _load_compose()
    configs = doc["configs"]
    assert (
        configs["flowmesh_otel_collector_config"]["file"]
        == "./otel-collector-config.yaml"
    )
    assert configs["flowmesh_clickhouse_init"]["file"] == "./clickhouse-init.sql"
    assert _COLLECTOR_CONFIG_PATH.parent == _COMPOSE_PATH.parent
    assert _CLICKHOUSE_INIT_PATH.parent == _COMPOSE_PATH.parent


def test_collector_config_never_references_the_flowmesh_server_cli_or_sdk() -> None:
    """The write path must never be reachable from the server, CLI or SDK.

    A byte-level guard: the Collector's own config should have no reason to name the
    FlowMesh server at all -- it receives OTLP and writes to ClickHouse, nothing else.
    """
    text = _COLLECTOR_CONFIG_PATH.read_text()
    forbidden_markers = (
        "flowmesh-server",
        "/api/v1",
        "SERVER_APP_PORT",
        "SERVER_HTTP_PORT",
    )
    for marker in forbidden_markers:
        assert marker not in text, f"collector config references {marker!r}"


def test_collector_config_exporters_target_only_clickhouse() -> None:
    doc = yaml.safe_load(_COLLECTOR_CONFIG_PATH.read_text())
    exporter_names = set(doc["exporters"])
    assert exporter_names == {"clickhouse/traces", "clickhouse/metrics"}
    for exporter in doc["exporters"].values():
        assert exporter["endpoint"] == "${env:TELEMETRY_CLICKHOUSE_DSN}"


def test_store_module_defines_no_receiver_or_listen_socket() -> None:
    """The read port must never become a second ingest path.

    A structural guard on the module set T6 owns: no OTLP/receiver-shaped symbol, no
    server socket construction.
    """
    import inspect

    from server.telemetry import clickhouse as clickhouse_module
    from server.telemetry import store as store_module

    source = inspect.getsource(store_module) + inspect.getsource(clickhouse_module)
    lowered = source.lower()
    forbidden = (
        "otlp",
        "receiver",
        "bind(",
        "listen(",
        "grpc.server",
        "add_insecure_port",
    )
    for marker in forbidden:
        assert marker not in lowered, f"TelemetryStore module references {marker!r}"


def test_clickhouse_init_ddl_has_no_reference_to_producers_or_the_server() -> None:
    text = _CLICKHOUSE_INIT_PATH.read_text()
    assert "flowmesh-server" not in text
    assert "/api/v1" not in text


def test_clickhouse_ddl_uses_trace_id_leading_replacing_merge_tree() -> None:
    """The property the ClickHouse-verified dedup/lookup fix depends on."""
    text = _CLICKHOUSE_INIT_PATH.read_text()
    assert re.search(r"ENGINE\s*=\s*ReplacingMergeTree", text)
    assert re.search(r"ORDER BY\s*\(TraceId,\s*Timestamp,\s*SpanId\)", text)
