from pathlib import Path

import pytest
from flowmesh_cli_stack.env_schema import STACK_ENV_SCHEMA
from flowmesh_stack.env import parse_env_file
from flowmesh_stack.env_schema import render_env_example, validate_env_values

_TLS = {
    "SERVER_GRPC_TLS_CERT_FILE": "/etc/ssl/server/server.pem",
    "SERVER_GRPC_TLS_KEY_FILE": "/etc/ssl/server/server.key",
}
_PROFILE = {"COMPOSE_PROFILES": "telemetry", "TELEMETRY_OTLP_TOKEN": "otlp-token"}


def _problems(env: dict[str, str]) -> tuple[list[str], list[str]]:
    errors, warnings = validate_env_values(STACK_ENV_SCHEMA, env)
    return (
        [e for e in errors if "OTLP" in e],
        [w for w in warnings if "OTLP" in w],
    )


def test_the_telemetry_profile_requires_the_collector_token() -> None:
    errors, _ = _problems({"COMPOSE_PROFILES": "content, telemetry"})
    assert errors and "TELEMETRY_OTLP_TOKEN" in errors[0]


def test_a_token_satisfies_the_telemetry_profile() -> None:
    errors, _ = _problems(_PROFILE)
    assert errors == []


def test_the_default_endpoint_matches_the_default_collector_tls(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(render_env_example(STACK_ENV_SCHEMA))
    env = parse_env_file(env_file)

    assert all(env[key] for key in _TLS)
    assert env["SERVER_METRICS_OTLP_ENDPOINT"].startswith("https://")
    errors, _ = _problems(env | _PROFILE)
    assert errors == []


@pytest.mark.parametrize("host", ["localhost", "root.example"])
def test_a_plaintext_endpoint_to_a_tls_collector_is_an_error(host: str) -> None:
    errors, _ = _problems(
        _PROFILE | _TLS | {"SERVER_METRICS_OTLP_ENDPOINT": f"http://{host}:4317"}
    )
    assert errors and "use https://" in errors[0]


def test_an_https_endpoint_to_a_plaintext_local_collector_is_an_error() -> None:
    errors, _ = _problems(
        _PROFILE | {"SERVER_METRICS_OTLP_ENDPOINT": "https://127.0.0.1:4317"}
    )
    assert errors and "use http://" in errors[0]


def test_an_https_endpoint_to_another_collector_needs_no_local_tls() -> None:
    errors, _ = _problems(
        _PROFILE | {"SERVER_METRICS_OTLP_ENDPOINT": "https://root.example:4317"}
    )
    assert errors == []


def test_the_scheme_rule_waits_for_the_telemetry_profile() -> None:
    errors, _ = _problems(_TLS | {"SERVER_METRICS_OTLP_ENDPOINT": "http://x:4317"})
    assert errors == []


_EXPORTING = {"SERVER_METRICS_TELEMETRY_LEVEL": "coarse"}


def test_an_https_collector_without_a_ca_warns() -> None:
    _, warnings = _problems(
        {**_EXPORTING, "SERVER_METRICS_OTLP_ENDPOINT": "https://root:4317"}
    )
    assert warnings and "docs/TELEMETRY.md" in warnings[0]
    for ca_key in ("SERVER_GRPC_TLS_CA_FILE", "SERVER_METRICS_OTLP_CA_FILE"):
        _, warnings = _problems(
            {
                **_EXPORTING,
                "SERVER_METRICS_OTLP_ENDPOINT": "https://root:4317",
                ca_key: "/etc/ssl/server/root-ca.pem",
            }
        )
        assert warnings == []


@pytest.mark.parametrize(
    "telemetry",
    [
        {"SERVER_METRICS_TELEMETRY_LEVEL": "off"},
        {
            "SERVER_METRICS_TELEMETRY_LEVEL": "fine",
            "SERVER_METRICS_TRACES_ENABLED": "false",
            "SERVER_METRICS_METRICS_ENABLED": "false",
        },
    ],
)
def test_a_node_exporting_nothing_gets_no_collector_ca_warning(
    telemetry: dict[str, str],
) -> None:
    for node in ({}, {"NODE_ROLE": "worker"}):
        _, warnings = _problems(
            {
                **telemetry,
                **node,
                "SERVER_METRICS_OTLP_ENDPOINT": "https://10.0.0.5:4317",
            }
        )
        assert warnings == []


def test_a_worker_node_verifying_the_root_s_collector_with_its_own_ca_warns() -> None:
    worker = {
        **_EXPORTING,
        "NODE_ROLE": "worker",
        "SERVER_GRPC_TLS_CA_FILE": "/etc/ssl/server/server-ca.pem",
    }
    _, warnings = _problems(
        {**worker, "SERVER_METRICS_OTLP_ENDPOINT": "https://10.0.0.5:4317"}
    )
    assert len(warnings) == 1 and "root's server CA" in warnings[0]

    for settled in (
        {"SERVER_METRICS_OTLP_CA_FILE": "/etc/ssl/server/root-ca.pem"},
        {"SERVER_METRICS_OTLP_ENDPOINT": "https://localhost:4317"},
        {"NODE_ROLE": "root"},
    ):
        _, warnings = _problems(
            {
                **worker,
                "SERVER_METRICS_OTLP_ENDPOINT": "https://10.0.0.5:4317",
                **settled,
            }
        )
        assert warnings == []
