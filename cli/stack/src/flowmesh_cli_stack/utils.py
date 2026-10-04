import os
import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

import typer
from flowmesh import FlowMesh
from flowmesh.models.nodes import NodeRole
from flowmesh_cli.core import logging
from flowmesh_cli.core.assets import asset_path
from flowmesh_stack.env import EnvFileError, load_env, parse_env_file
from flowmesh_stack.node_client import NodeClient
from flowmesh_stack.paths import ensure_dir, ensure_file, resolve_path

from .env_schema import STACK_ENV_SCHEMA, collector_serves_tls, telemetry_profile_on

DEFAULT_ENV_FILE = Path(".env")
_SCHEMA_DEFAULTS = {
    var.key: var.default
    for section in STACK_ENV_SCHEMA.sections
    for var in section.vars
}
# Compose resolves a relative bind source against the packaged compose file, so the CLI
# anchors each of the stack's mount sources to the working directory.
STACK_PATH_DEFAULTS = {
    key: _SCHEMA_DEFAULTS[key]
    for key in (
        "REDIS_TLS_DIR",
        "SERVER_TLS_DIR",
        "NETWORK_PLANE_PEER_TLS_DIR",
        "SERVER_WORKER_CONFIG",
        "FLOWMESH_PLUGIN_DIR",
    )
}
STACK_PATH_KEYS = set(STACK_PATH_DEFAULTS)
_STACK_FILE_KEYS = {"SERVER_WORKER_CONFIG"}
STACK_SUFFIX_ENV = "FLOWMESH_STACK_SUFFIX"
STACK_SLUG_ENV = "FLOWMESH_STACK_SLUG"
WORKER_RESULTS_DIR_ENV = "WORKER_RESULTS_DIR"
_STACK_SLUG_BASE = "flowmesh_node"
_STACK_SUFFIX_MAX_LEN = 48


def _resolve_stack_suffix(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-_.")
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)[:_STACK_SUFFIX_MAX_LEN]
    sanitized = sanitized.rstrip("-_.")
    if not sanitized and value.strip():
        raise ValueError(
            f"{STACK_SUFFIX_ENV} must contain at least one ASCII letter or digit"
        )
    return sanitized


def stack_resource_env_overrides(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = os.environ if env is None else env
    suffix = _resolve_stack_suffix(values.get(STACK_SUFFIX_ENV, ""))
    stack_slug = f"{_STACK_SLUG_BASE}_{suffix}" if suffix else _STACK_SLUG_BASE
    return {STACK_SLUG_ENV: stack_slug}


def apply_stack_resource_env() -> None:
    overrides = stack_resource_env_overrides(os.environ)
    os.environ.update(overrides)
    os.environ["COMPOSE_PROJECT_NAME"] = overrides[STACK_SLUG_ENV]
    results_volume = f"{overrides[STACK_SLUG_ENV]}_results"
    if not os.environ.get(WORKER_RESULTS_DIR_ENV, "").strip():
        os.environ[WORKER_RESULTS_DIR_ENV] = results_volume


_PLUGIN_DATA_PATH_PREFIXES = ("/", "./", "../", "~")
_PLUGIN_DATA_ALIAS = "flowmesh_plugin_data"
_PLUGIN_DATA_DEFAULT = _SCHEMA_DEFAULTS["FLOWMESH_PLUGIN_DATA_DIR"]
_PLUGIN_DATA_VOLUME_ENV = "FLOWMESH_PLUGIN_DATA_VOLUME"


def apply_stack_path_env(base_dir: Path) -> None:
    """Set each stack mount source to an absolute path, its default when unset."""
    for key, default in STACK_PATH_DEFAULTS.items():
        os.environ[key] = resolve_path(os.getenv(key, ""), default, base_dir).as_posix()


COLLECTOR_TLS_CONFIG_ARG_ENV = "FLOWMESH_COLLECTOR_TLS_CONFIG_ARG"
COLLECTOR_TLS_CERT_ENV = "FLOWMESH_COLLECTOR_TLS_CERT"
COLLECTOR_TLS_KEY_ENV = "FLOWMESH_COLLECTOR_TLS_KEY"
COLLECTOR_USER_ENV = "FLOWMESH_COLLECTOR_USER"
_COLLECTOR_TLS_CONFIG_ARG = "--config=/etc/otelcol-contrib/tls.yaml"
_SERVER_TLS_MOUNT = PurePosixPath("/etc/ssl/server")


def apply_collector_tls_env() -> None:
    """Hand the collector the server's TLS certificate and key alone, and run it as
    the key's owner, who alone can read it.

    A node without the telemetry profile, or a stack without that material, runs the
    collector in plaintext as the invoking user, with the null device bound in place of
    the files. Raises ``ValueError`` naming a configured file the collector cannot be
    handed.
    """
    if not (telemetry_profile_on(os.environ) and collector_serves_tls(os.environ)):
        os.environ.update(
            {
                COLLECTOR_TLS_CONFIG_ARG_ENV: "",
                COLLECTOR_TLS_CERT_ENV: os.devnull,
                COLLECTOR_TLS_KEY_ENV: os.devnull,
                COLLECTOR_USER_ENV: f"{os.getuid()}:{os.getgid()}",
            }
        )
        return
    cert = _server_tls_host_file("SERVER_GRPC_TLS_CERT_FILE")
    key = _server_tls_host_file("SERVER_GRPC_TLS_KEY_FILE")
    try:
        owner = key.stat()
    except OSError as exc:
        raise ValueError(f"Cannot read the collector's TLS key {key}: {exc}") from exc
    os.environ.update(
        {
            COLLECTOR_TLS_CONFIG_ARG_ENV: _COLLECTOR_TLS_CONFIG_ARG,
            COLLECTOR_TLS_CERT_ENV: cert.as_posix(),
            COLLECTOR_TLS_KEY_ENV: key.as_posix(),
            COLLECTOR_USER_ENV: f"{owner.st_uid}:{owner.st_gid}",
        }
    )


def _server_tls_host_file(key: str) -> Path:
    """Return the host file behind a server TLS path, which the server reads from its
    ``SERVER_TLS_DIR`` mount."""
    value = os.environ[key].strip()
    try:
        relative = PurePosixPath(value).relative_to(_SERVER_TLS_MOUNT)
    except ValueError:
        raise ValueError(
            f"{key}={value} is outside {_SERVER_TLS_MOUNT}, so the collector cannot "
            "be handed it"
        ) from None
    path = Path(os.environ.get("SERVER_TLS_DIR", ""), relative)
    try:
        is_file = path.is_file()
    except OSError as exc:
        raise ValueError(f"Cannot read {key} at {path}: {exc}") from exc
    if not is_file:
        raise ValueError(f"{key} names {path} on this host, which is not a file")
    return path


def apply_plugin_data_env(base_dir: Path) -> None:
    raw = os.environ.get("FLOWMESH_PLUGIN_DATA_DIR", "").strip()
    # The env file loads once per process, so a second compose call sees this
    # function's own output: the alias, with the volume already recorded.
    if raw == _PLUGIN_DATA_ALIAS and os.environ.get(_PLUGIN_DATA_VOLUME_ENV):
        return
    if not raw or raw.startswith(_PLUGIN_DATA_PATH_PREFIXES):
        resolved = resolve_path(raw, default=_PLUGIN_DATA_DEFAULT, base_dir=base_dir)
        os.environ["FLOWMESH_PLUGIN_DATA_DIR"] = resolved.as_posix()
    else:
        os.environ[_PLUGIN_DATA_VOLUME_ENV] = raw
        os.environ["FLOWMESH_PLUGIN_DATA_DIR"] = _PLUGIN_DATA_ALIAS


def stack_compose_file() -> Path:
    return asset_path("flowmesh_cli_stack.assets", "compose.yml")


def stack_env_example() -> Path:
    return asset_path("flowmesh_cli_stack.assets", ".env.example")


def stack_bake_file() -> Path:
    return asset_path("flowmesh_cli_stack.assets", "docker-bake.hcl")


def load_stack_env(env_file: Path) -> None:
    """Load the stack env file, exiting with its error when Compose would refuse it."""
    try:
        load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)
    except EnvFileError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)


def read_stack_env(env_file: Path) -> dict[str, str]:
    """Read the stack env file, exiting with its error when Compose would refuse it."""
    try:
        return parse_env_file(env_file)
    except EnvFileError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)


def stack_node_client(
    env_file: Path, base_url: str | None, token: str | None
) -> NodeClient:
    load_stack_env(env_file)
    default_base = "http://{}:{}".format(
        os.getenv("SERVER_HOST", "localhost"),
        os.getenv("SERVER_HTTP_PORT", os.getenv("SERVER_APP_PORT", "8000")),
    )
    resolved_base = base_url or default_base
    resolved_token = token or os.getenv("FLOWMESH_API_KEY") or None
    return NodeClient(resolved_base, token=resolved_token)


def flowmesh_client(
    env_file: Path, base_url: str | None, api_key: str | None
) -> FlowMesh:
    load_stack_env(env_file)
    return FlowMesh(base_url=base_url, api_key=api_key)


def ensure_deploy_paths(base_dir: Path) -> None:
    for key, default in STACK_PATH_DEFAULTS.items():
        path = resolve_path(os.getenv(key, ""), default, base_dir)
        (ensure_file if key in _STACK_FILE_KEYS else ensure_dir)(path)
    if not os.environ.get(_PLUGIN_DATA_VOLUME_ENV):
        ensure_dir(
            resolve_path(
                os.getenv("FLOWMESH_PLUGIN_DATA_DIR", ""),
                default=_PLUGIN_DATA_DEFAULT,
                base_dir=base_dir,
            )
        )


def parse_node_role(raw: str) -> NodeRole:
    """Parse a CLI-supplied role string into a NodeRole, exiting on invalid input."""
    try:
        return NodeRole(raw.strip().lower())
    except ValueError:
        logging.error(f"Invalid role {raw!r}; expected one of {', '.join(NodeRole)}.")
        raise typer.Exit(code=1) from None


def resolve_package_version(name: str = "flowmesh-cli-stack") -> str | None:
    """Return the installed flowmesh-cli-stack version, or None if it can't be read."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None
