import os
from pathlib import Path
from unittest.mock import patch

import pytest
from flowmesh_cli_stack.utils import (
    _PLUGIN_DATA_ALIAS,
    COLLECTOR_TLS_CERT_ENV,
    COLLECTOR_TLS_CONFIG_ARG_ENV,
    COLLECTOR_TLS_KEY_ENV,
    COLLECTOR_USER_ENV,
    STACK_PATH_DEFAULTS,
    STACK_SLUG_ENV,
    STACK_SUFFIX_ENV,
    WORKER_RESULTS_DIR_ENV,
    apply_collector_tls_env,
    apply_plugin_data_env,
    apply_stack_path_env,
    apply_stack_resource_env,
    ensure_deploy_paths,
    stack_compose_file,
    stack_resource_env_overrides,
)


def test_stack_resource_env_overrides_use_defaults_without_suffix() -> None:
    overrides = stack_resource_env_overrides({})
    assert overrides[STACK_SLUG_ENV] == "flowmesh_node"


def test_stack_resource_env_overrides_append_sanitized_suffix() -> None:
    overrides = stack_resource_env_overrides({STACK_SUFFIX_ENV: "alice.dev"})
    assert overrides[STACK_SLUG_ENV] == "flowmesh_node_alice.dev"


def test_stack_resource_env_overrides_reject_invalid_suffix() -> None:
    with pytest.raises(ValueError, match=STACK_SUFFIX_ENV):
        stack_resource_env_overrides({STACK_SUFFIX_ENV: "!!!"})


def test_apply_stack_resource_env_defaults_results_dirs_from_suffix() -> None:
    with patch.dict(
        os.environ,
        {
            STACK_SUFFIX_ENV: "alice.dev",
            WORKER_RESULTS_DIR_ENV: "",
        },
        clear=True,
    ):
        apply_stack_resource_env()
        assert os.environ[WORKER_RESULTS_DIR_ENV] == "flowmesh_node_alice.dev_results"


def test_apply_plugin_data_env_empty_resolves_default(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"FLOWMESH_PLUGIN_DATA_DIR": ""}, clear=True):
        apply_plugin_data_env(tmp_path)
        expected = (tmp_path / "plugin-data").as_posix()
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == expected
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ
        assert not (tmp_path / "plugin-data").exists()  # routing only; no mkdir


def test_apply_plugin_data_env_relative_path_is_cwd_resolved(tmp_path: Path) -> None:
    env = {"FLOWMESH_PLUGIN_DATA_DIR": "./custom-data"}
    with patch.dict(os.environ, env, clear=True):
        apply_plugin_data_env(tmp_path)
        expected = (tmp_path / "custom-data").as_posix()
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == expected
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ


def test_apply_plugin_data_env_absolute_path_passthrough(tmp_path: Path) -> None:
    abs_path = "/var/lib/flowmesh-plugin"
    with patch.dict(os.environ, {"FLOWMESH_PLUGIN_DATA_DIR": abs_path}, clear=True):
        apply_plugin_data_env(tmp_path)
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == abs_path
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ


def test_apply_plugin_data_env_tilde_is_path(tmp_path: Path) -> None:
    env = {"FLOWMESH_PLUGIN_DATA_DIR": "~/flowmesh-data"}
    with patch.dict(os.environ, env, clear=True):
        apply_plugin_data_env(tmp_path)
        # Resolved against base_dir; the leading ~ keeps it in path mode.
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"]  # set to something resolved


def test_apply_plugin_data_env_bare_name_routes_to_volume(tmp_path: Path) -> None:
    env = {"FLOWMESH_PLUGIN_DATA_DIR": "my_external_vol"}
    with patch.dict(os.environ, env, clear=True):
        apply_plugin_data_env(tmp_path)
        assert os.environ["FLOWMESH_PLUGIN_DATA_VOLUME"] == "my_external_vol"
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == _PLUGIN_DATA_ALIAS


@pytest.mark.parametrize("raw", [None, "", "./custom"])
def test_every_stack_mount_source_is_set_to_an_absolute_path(
    tmp_path: Path, raw: str | None
) -> None:
    env = {} if raw is None else dict.fromkeys(STACK_PATH_DEFAULTS, raw)
    with patch.dict(os.environ, env, clear=True):
        apply_stack_path_env(tmp_path)
        for key, default in STACK_PATH_DEFAULTS.items():
            expected = tmp_path / (raw or default)
            assert os.environ[key] == expected.resolve().as_posix()


def test_an_absolute_stack_mount_source_is_kept(tmp_path: Path) -> None:
    env = {"REDIS_TLS_DIR": "/srv/tls/redis"}
    with patch.dict(os.environ, env, clear=True):
        apply_stack_path_env(tmp_path)
        assert os.environ["REDIS_TLS_DIR"] == "/srv/tls/redis"


def test_deploy_paths_create_each_mount_source_as_compose_mounts_it(
    tmp_path: Path,
) -> None:
    with patch.dict(os.environ, {}, clear=True):
        apply_stack_path_env(tmp_path)
        ensure_deploy_paths(tmp_path)
        for key in STACK_PATH_DEFAULTS:
            path = Path(os.environ[key])
            assert path.is_file() if key == "SERVER_WORKER_CONFIG" else path.is_dir()


def test_compose_requires_every_mount_source_from_the_cli() -> None:
    compose = stack_compose_file().read_text()
    for key in (*STACK_PATH_DEFAULTS, "FLOWMESH_PLUGIN_DATA_DIR"):
        assert f"${{{key}:?" in compose
        assert f"${{{key}:-" not in compose


def _server_tls(tmp_path: Path) -> Path:
    tls = tmp_path / "tls"
    tls.mkdir()
    for name in ("server.pem", "server.key", "server-ca.pem", "server-ca.key"):
        (tls / name).write_text(name)
    (tls / "server.key").chmod(0o600)
    return tls


_TELEMETRY = {"COMPOSE_PROFILES": "root,telemetry"}


def test_the_collector_gets_the_server_cert_and_key_alone(tmp_path: Path) -> None:
    tls = _server_tls(tmp_path)
    env = {
        **_TELEMETRY,
        "SERVER_TLS_DIR": tls.as_posix(),
        "SERVER_GRPC_TLS_CERT_FILE": "/etc/ssl/server/server.pem",
        "SERVER_GRPC_TLS_KEY_FILE": "/etc/ssl/server/server.key",
    }
    with patch.dict(os.environ, env, clear=True):
        apply_collector_tls_env()
        assert os.environ[COLLECTOR_TLS_CONFIG_ARG_ENV] == (
            "--config=/etc/otelcol-contrib/tls.yaml"
        )
        assert os.environ[COLLECTOR_TLS_CERT_ENV] == (tls / "server.pem").as_posix()
        assert os.environ[COLLECTOR_TLS_KEY_ENV] == (tls / "server.key").as_posix()
        key = (tls / "server.key").stat()
        assert os.environ[COLLECTOR_USER_ENV] == f"{key.st_uid}:{key.st_gid}"


def _assert_plaintext_collector() -> None:
    assert os.environ[COLLECTOR_TLS_CONFIG_ARG_ENV] == ""
    assert os.environ[COLLECTOR_TLS_CERT_ENV] == os.devnull
    assert os.environ[COLLECTOR_TLS_KEY_ENV] == os.devnull
    assert os.environ[COLLECTOR_USER_ENV] == f"{os.getuid()}:{os.getgid()}"


@pytest.mark.parametrize(
    ("cert", "key"), [("/etc/ssl/server/server.pem", ""), ("", "")]
)
def test_without_server_material_the_collector_binds_nothing_from_disk(
    tmp_path: Path, cert: str, key: str
) -> None:
    tls = _server_tls(tmp_path)
    env = {
        **_TELEMETRY,
        "SERVER_TLS_DIR": tls.as_posix(),
        "SERVER_GRPC_TLS_CERT_FILE": cert,
        "SERVER_GRPC_TLS_KEY_FILE": key,
    }
    with patch.dict(os.environ, env, clear=True):
        apply_collector_tls_env()
        _assert_plaintext_collector()


def test_a_node_without_the_telemetry_profile_reads_no_tls_file(
    tmp_path: Path,
) -> None:
    env = {
        "COMPOSE_PROFILES": "root",
        "SERVER_TLS_DIR": (tmp_path / "unreadable").as_posix(),
        "SERVER_GRPC_TLS_CERT_FILE": "/etc/ssl/server/server.pem",
        "SERVER_GRPC_TLS_KEY_FILE": "/etc/ssl/server/server.key",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(Path, "is_file", side_effect=PermissionError("denied")),
        patch.object(Path, "stat", side_effect=PermissionError("denied")),
    ):
        apply_collector_tls_env()
        _assert_plaintext_collector()


@pytest.mark.parametrize(
    ("cert", "named"),
    [
        ("/etc/ssl/server/missing.pem", "missing.pem"),
        ("/opt/tls/server.pem", "/opt/tls/server.pem"),
    ],
)
def test_a_tls_file_the_collector_cannot_be_handed_is_an_error(
    tmp_path: Path, cert: str, named: str
) -> None:
    env = {
        **_TELEMETRY,
        "SERVER_TLS_DIR": _server_tls(tmp_path).as_posix(),
        "SERVER_GRPC_TLS_CERT_FILE": cert,
        "SERVER_GRPC_TLS_KEY_FILE": "/etc/ssl/server/server.key",
    }
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError, match=named):
            apply_collector_tls_env()


def test_an_unreadable_tls_directory_is_an_error_naming_it(tmp_path: Path) -> None:
    env = {
        **_TELEMETRY,
        "SERVER_TLS_DIR": _server_tls(tmp_path).as_posix(),
        "SERVER_GRPC_TLS_CERT_FILE": "/etc/ssl/server/server.pem",
        "SERVER_GRPC_TLS_KEY_FILE": "/etc/ssl/server/server.key",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(Path, "is_file", side_effect=PermissionError("denied")),
    ):
        with pytest.raises(ValueError, match="server.pem"):
            apply_collector_tls_env()
