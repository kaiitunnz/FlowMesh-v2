"""A root node runs its own Redis and content store on credentials of its own."""

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
    assert service_credential_errors(first) == []


def test_a_worker_init_generates_no_redis_password(tmp_path):
    env = _init(tmp_path, "worker")

    assert env["REDIS_PASSWORD"] == ""
    assert service_credential_errors(env) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("REDIS_PASSWORD", "very-strong-password"),
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
