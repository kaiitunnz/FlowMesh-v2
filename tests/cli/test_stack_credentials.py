"""A node runs its Redis, content store and ClickHouse on credentials of its own."""

from pathlib import Path
from unittest import mock

import pytest
import typer
import yaml
from flowmesh_cli_stack import stack
from flowmesh_cli_stack.env_schema import STACK_ENV_SCHEMA, service_credential_errors
from flowmesh_stack.env import parse_env_file
from flowmesh_stack.env_schema import validate_env_values

_CREDENTIALS = (
    "REDIS_PASSWORD",
    "CONTENT_STORE_ACCESS_KEY",
    "CONTENT_STORE_SECRET_KEY",
    "TELEMETRY_CLICKHOUSE_PASSWORD",
)


def _set(env_file: Path, key: str, value: str) -> None:
    lines = env_file.read_text().splitlines()
    env_file.write_text(
        "\n".join(
            f"{key}={value}" if line.startswith(f"{key}=") else line for line in lines
        )
        + "\n"
    )


def _init(tmp_path: Path, role: str, name: str = ".env") -> dict[str, str]:
    env_file = tmp_path / name
    stack.init(env_file=env_file, force=True, role=role, deploy=False)
    return parse_env_file(env_file)


def test_a_root_init_writes_fresh_credentials_it_never_prints(tmp_path, capsys):
    first = _init(tmp_path, "root")
    second = _init(tmp_path, "root", ".env.second")

    printed = capsys.readouterr()
    for key in _CREDENTIALS:
        assert len(first[key]) >= 16
        assert first[key] != second[key]
        assert first[key] not in printed.out + printed.err
    assert (
        first["SERVER_METRICS_CLICKHOUSE_PASSWORD"]
        == first["TELEMETRY_CLICKHOUSE_PASSWORD"]
    )
    assert service_credential_errors(first) == []
    assert service_credential_errors({**first, "COMPOSE_PROFILES": "telemetry"}) == []


def test_a_root_init_writes_an_env_only_its_owner_reads(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    env_file.chmod(0o644)

    stack.init(env_file=env_file, force=True, role="root", deploy=False)

    assert env_file.stat().st_mode & 0o777 == 0o600


def test_a_worker_takes_the_roots_redis_password(tmp_path):
    env = _init(tmp_path, "worker")

    assert env["REDIS_PASSWORD"] == ""
    assert [
        error for error in service_credential_errors(env) if "REDIS_PASSWORD" in error
    ]
    root = _init(tmp_path, "root", ".env.root")
    assert (
        service_credential_errors({**env, "REDIS_PASSWORD": root["REDIS_PASSWORD"]})
        == []
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("REDIS_PASSWORD", "very-strong-password"),
        ("REDIS_PASSWORD", ""),
        ("CONTENT_STORE_ACCESS_KEY", "flowmesh"),
        ("CONTENT_STORE_SECRET_KEY", "flowmeshcontent"),
        ("CONTENT_STORE_SECRET_KEY", ""),
    ],
)
def test_up_refuses_a_root_on_a_default_credential_and_names_it(tmp_path, key, value):
    env_file = tmp_path / ".env"
    stack.init(env_file=env_file, force=True, role="root", deploy=False)
    env = parse_env_file(env_file)
    env_file.write_text(
        env_file.read_text().replace(f"{key}={env[key]}", f"{key}={value}")
    )

    with mock.patch.object(stack, "_compose") as compose:
        with mock.patch.object(stack.logging, "error") as error:
            with pytest.raises(typer.Exit):
                stack.up(env_file=env_file, image_tag=None)

    compose.assert_not_called()
    assert any(key in call.args[0] for call in error.call_args_list)
    errors, _ = validate_env_values(STACK_ENV_SCHEMA, parse_env_file(env_file))
    assert any(key in message for message in errors)


@pytest.mark.parametrize(
    ("role", "key", "value"),
    [
        (role, "TELEMETRY_CLICKHOUSE_PASSWORD", value)
        for role in ("root", "worker")
        for value in ("flowmesh", "")
    ]
    + [("root", "SERVER_METRICS_CLICKHOUSE_PASSWORD", v) for v in ("flowmesh", "")],
)
def test_up_refuses_the_telemetry_profile_on_a_default_password(
    tmp_path, role, key, value
):
    env_file = tmp_path / ".env"
    stack.init(env_file=env_file, force=True, role=role, deploy=False)
    _set(env_file, "COMPOSE_PROFILES", "telemetry")
    _set(env_file, key, value)

    with mock.patch.object(stack, "_compose") as compose:
        with mock.patch.object(stack.logging, "error") as error:
            with pytest.raises(typer.Exit):
                stack.up(env_file=env_file, image_tag=None)

    compose.assert_not_called()
    assert any(key in call.args[0] for call in error.call_args_list)


@pytest.mark.parametrize("services", [None, ["server"]])
def test_restart_refuses_a_default_credential_before_touching_the_stack(
    tmp_path, services
):
    env_file = tmp_path / ".env"
    stack.init(env_file=env_file, force=True, role="root", deploy=False)
    _set(env_file, "REDIS_PASSWORD", "very-strong-password")

    with (
        mock.patch.object(stack, "_compose") as compose,
        mock.patch.object(stack, "_drain_workers") as drain,
        mock.patch.object(stack.logging, "error") as error,
        pytest.raises(typer.Exit),
    ):
        stack.restart(services=services, env_file=env_file, image_tag=None, pull=False)

    compose.assert_not_called()
    drain.assert_not_called()
    assert any("REDIS_PASSWORD" in call.args[0] for call in error.call_args_list)


def test_restart_of_a_service_checks_only_the_credentials_it_reads(tmp_path):
    env_file = tmp_path / ".env"
    stack.init(env_file=env_file, force=True, role="root", deploy=False)
    _set(env_file, "COMPOSE_PROFILES", "telemetry")
    _set(env_file, "TELEMETRY_CLICKHOUSE_PASSWORD", "flowmesh")

    with (
        mock.patch.object(stack, "_compose") as compose,
        mock.patch.object(stack, "_drain_workers"),
    ):
        stack.restart(
            services=["redis_control"], env_file=env_file, image_tag=None, pull=False
        )
    compose.assert_called_once()

    with (
        mock.patch.object(stack, "_compose") as compose,
        mock.patch.object(stack, "_drain_workers"),
        pytest.raises(typer.Exit),
    ):
        stack.restart(services=None, env_file=env_file, image_tag=None, pull=False)
    compose.assert_not_called()


def test_a_worker_on_the_telemetry_profile_needs_only_the_collector_password():
    env = {
        "NODE_ROLE": "worker",
        "COMPOSE_PROFILES": "telemetry",
        "TELEMETRY_CLICKHOUSE_PASSWORD": "a-password-of-its-own",
        "CONTENT_STORE_ENDPOINT_URL": "https://store.example",
    }
    assert service_credential_errors(env) == []
    assert service_credential_errors({**env, "NODE_ROLE": "root"}) == [
        "SERVER_METRICS_CLICKHOUSE_PASSWORD must be set for the telemetry profile's "
        "ClickHouse"
    ]


def test_the_telemetry_password_is_not_checked_without_its_profile():
    assert (
        service_credential_errors(
            {
                "NODE_ROLE": "worker",
                "REDIS_ACL_ENABLED": "0",
                "TELEMETRY_CLICKHOUSE_PASSWORD": "flowmesh",
                "SERVER_METRICS_CLICKHOUSE_PASSWORD": "flowmesh",
            }
        )
        == []
    )


def test_up_leaves_disabled_acl_and_an_external_store_to_the_operator(tmp_path):
    env_file = tmp_path / ".env"
    stack.init(env_file=env_file, force=True, role="root", deploy=False)
    text = env_file.read_text()
    env = parse_env_file(env_file)
    for key in _CREDENTIALS:
        text = text.replace(f"{key}={env[key]}", f"{key}=")
    text = text.replace("REDIS_ACL_ENABLED=1", "REDIS_ACL_ENABLED=0").replace(
        "CONTENT_STORE_ENDPOINT_URL=",
        "CONTENT_STORE_ENDPOINT_URL=https://s3.example.com",
    )
    env_file.write_text(text)

    with mock.patch.object(stack, "_compose") as compose:
        stack.up(env_file=env_file, image_tag=None)

    compose.assert_called_once()


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
    assert store["environment"] == {
        "MINIO_ROOT_USER": "${CONTENT_STORE_ACCESS_KEY:-}",
        "MINIO_ROOT_PASSWORD": "${CONTENT_STORE_SECRET_KEY:-}",
    }
