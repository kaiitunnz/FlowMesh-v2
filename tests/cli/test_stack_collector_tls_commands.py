"""Only a command that starts the collector needs its TLS files to be readable and
its token set."""

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from flowmesh_cli_stack import stack as stack_module
from flowmesh_stack.env import load_env


@pytest.fixture
def broken_tls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A telemetry node whose configured TLS key is missing on the host."""
    tls = tmp_path / "tls"
    tls.mkdir()
    (tls / "server.pem").write_text("cert")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "COMPOSE_PROFILES=telemetry\n"
        f"SERVER_TLS_DIR={tls.as_posix()}\n"
        "SERVER_GRPC_TLS_CERT_FILE=/etc/ssl/server/server.pem\n"
        "SERVER_GRPC_TLS_KEY_FILE=/etc/ssl/server/server.key\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(load_env, "_loaded", None, raising=False)
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        yield env_file


@pytest.fixture
def compose() -> Iterator[MagicMock]:
    with (
        patch(
            "flowmesh_stack.docker.compose",
            return_value=subprocess.CompletedProcess([], 0),
        ) as compose,
        patch.object(stack_module, "_drain_workers"),
    ):
        yield compose


@pytest.mark.parametrize("command", [stack_module.down, stack_module.ps])
def test_stopping_or_listing_the_stack_ignores_unreadable_tls(
    broken_tls: Path, compose: MagicMock, command
) -> None:
    command(env_file=broken_tls)
    compose.assert_called_once()
    assert os.environ["FLOWMESH_COLLECTOR_TLS_KEY"] == os.devnull


def test_starting_the_stack_refuses_unreadable_tls_naming_it(
    broken_tls: Path, compose: MagicMock
) -> None:
    with (
        patch.object(stack_module.logging, "error") as error,
        pytest.raises(typer.Exit) as raised,
    ):
        stack_module.up(env_file=broken_tls, image_tag=None)
    assert raised.value.exit_code == 1
    compose.assert_not_called()
    assert "server.key" in error.call_args.args[0]


def _telemetry_env(tmp_path: Path, token: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(f"COMPOSE_PROFILES=telemetry\nTELEMETRY_OTLP_TOKEN={token}\n")
    return env_file


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(load_env, "_loaded", None, raising=False)
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        yield tmp_path


@pytest.mark.parametrize("token", ["<replace-with-strong-token>", ""])
def test_starting_the_collector_refuses_the_example_token(
    scratch: Path, compose: MagicMock, token: str
) -> None:
    with (
        patch.object(stack_module.logging, "error") as error,
        pytest.raises(typer.Exit),
    ):
        stack_module.up(env_file=_telemetry_env(scratch, token), image_tag=None)
    compose.assert_not_called()
    assert "TELEMETRY_OTLP_TOKEN" in error.call_args.args[0]


def test_starting_the_collector_takes_a_real_token(
    scratch: Path, compose: MagicMock
) -> None:
    stack_module.up(env_file=_telemetry_env(scratch, "a-real-token"), image_tag=None)
    compose.assert_called_once()


def test_stopping_the_stack_ignores_the_example_token(
    scratch: Path, compose: MagicMock
) -> None:
    stack_module.down(
        env_file=_telemetry_env(scratch, "<replace-with-strong-token>"),
        image_tag=None,
    )
    compose.assert_called_once()
