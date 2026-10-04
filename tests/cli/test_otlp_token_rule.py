from flowmesh_cli_stack.env_schema import STACK_ENV_SCHEMA
from flowmesh_stack.env_schema import validate_env_values


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
    errors, _ = _problems(
        {"COMPOSE_PROFILES": "telemetry", "TELEMETRY_OTLP_TOKEN": "otlp-token"}
    )
    assert errors == []


def test_an_https_collector_without_the_server_ca_warns() -> None:
    _, warnings = _problems({"SERVER_METRICS_OTLP_ENDPOINT": "https://root:4317"})
    assert warnings and "generate_server_tls_certs.sh" in warnings[0]
    _, warnings = _problems(
        {
            "SERVER_METRICS_OTLP_ENDPOINT": "https://root:4317",
            "SERVER_GRPC_TLS_CA_FILE": "/etc/ssl/server/server-ca.pem",
        }
    )
    assert warnings == []
