"""A root node's init writes its Redis, content store and ClickHouse credentials."""

from pathlib import Path

import yaml
from flowmesh_cli_stack import stack
from flowmesh_stack.env import parse_env_file

_PASSWORD_PLACEHOLDER = "<replace-with-strong-password>"
_ACCESS_KEY_PLACEHOLDER = "<replace-with-access-key>"
_TOKEN_PLACEHOLDER = "<replace-with-strong-token>"
_PLACEHOLDERS = (_PASSWORD_PLACEHOLDER, _ACCESS_KEY_PLACEHOLDER, _TOKEN_PLACEHOLDER)
_CREDENTIALS = (
    "REDIS_PASSWORD",
    "CONTENT_STORE_ACCESS_KEY",
    "CONTENT_STORE_SECRET_KEY",
    "TELEMETRY_CLICKHOUSE_PASSWORD",
    "SERVER_METRICS_CLICKHOUSE_PASSWORD",
    "TELEMETRY_OTLP_TOKEN",
    "SERVER_METRICS_OTLP_TOKEN",
)


def _init(tmp_path: Path, role: str, name: str = ".env") -> dict[str, str]:
    env_file = tmp_path / name
    stack.init(env_file=env_file, force=True, role=role, deploy=False)
    return parse_env_file(env_file)


def test_a_root_init_replaces_every_placeholder_with_fresh_credentials(
    tmp_path, capsys
):
    first = _init(tmp_path, "root")
    second = _init(tmp_path, "root", ".env.second")

    printed = capsys.readouterr()
    for key in _CREDENTIALS:
        assert first[key] not in _PLACEHOLDERS
        assert len(first[key]) >= 16
        assert first[key] != second[key]
        assert first[key] not in printed.out + printed.err
    text = (tmp_path / ".env").read_text()
    assert not any(placeholder in text for placeholder in _PLACEHOLDERS)
    assert (
        first["SERVER_METRICS_CLICKHOUSE_PASSWORD"]
        == first["TELEMETRY_CLICKHOUSE_PASSWORD"]
    )
    assert first["SERVER_METRICS_OTLP_TOKEN"] == first["TELEMETRY_OTLP_TOKEN"]


def test_a_root_init_writes_an_env_only_its_owner_reads(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    env_file.chmod(0o644)

    stack.init(env_file=env_file, force=True, role="root", deploy=False)

    assert env_file.stat().st_mode & 0o777 == 0o600


def test_a_worker_init_leaves_the_redis_password_placeholder(tmp_path):
    env = _init(tmp_path, "worker")

    assert env["REDIS_PASSWORD"] == _PASSWORD_PLACEHOLDER
    assert env["SERVER_METRICS_OTLP_TOKEN"] == _TOKEN_PLACEHOLDER
    assert env["TELEMETRY_OTLP_TOKEN"] == _TOKEN_PLACEHOLDER


def test_the_content_store_console_listens_on_loopback():
    compose = yaml.safe_load(
        (Path(stack.__file__).parent / "assets" / "compose.yml").read_text()
    )
    store = compose["services"]["content-store"]

    assert store["ports"] == [
        "${CONTENT_STORE_BIND_HOST:-0.0.0.0}:${CONTENT_STORE_PORT:-9800}:9000",
        "${CONTENT_STORE_CONSOLE_BIND_HOST:-127.0.0.1}:"
        "${CONTENT_STORE_CONSOLE_PORT:-9801}:9001",
    ]
