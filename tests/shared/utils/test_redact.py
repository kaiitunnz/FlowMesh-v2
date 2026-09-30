import json

import pytest

from shared.tasks.specs.misc import _looks_credential
from shared.utils.redact import (
    REDACTED,
    credential_scrubber,
    is_credential_key,
    is_credential_url,
    redact_credential_fields,
    redact_url,
)


@pytest.mark.parametrize(
    "name",
    [
        "Authorization",
        "Proxy-Authorization",
        "Cookie",
        "Set-Cookie",
        "X-API-Key",
        "x-api-key",
        "apiKey",
        "api_key",
        "APIKey",
        "apikey",
        "access_key",
        "secret_key",
        "private_key",
        "client_secret",
        "password",
        "db_password",
        "passwd",
        "credentials",
        "token",
        "access_token",
        "id_token",
        "x-auth-token",
        "authToken",
        "auth",
        "bearer",
        "connection_string",
        "cert_data",
        "certificate",
        "passphrase",
        "ssh_key",
        "subscription_key",
        "authorized_keys",
        "authorizedKeys",
        "hf_token",
        "lumid_data_token",
        "aws_secret_access_key",
        "aws_access_key_id",
        "pwd",
        "db_pwd",
        "dsn",
        "SENTRY_DSN",
        "jwt",
        "X-JWT-Assertion",
        "database_url",
        "DATABASE_URL",
        "db_url",
        "mongodb_url",
    ],
)
def test_credential_keys_match(name: str) -> None:
    assert is_credential_key(name)


@pytest.mark.parametrize(
    "name",
    [
        "max_tokens",
        "num_tokens",
        "cache_key",
        "sort_key",
        "primary_key",
        "author",
        "authority",
        "tokenizer",
        "Content-Type",
        "model",
        "monkey",
        "session_id",
        "sessionid",
        "signature",
        "key",
        "url",
        "base_url",
        "dns",
    ],
)
def test_ordinary_keys_do_not_match(name: str) -> None:
    assert not is_credential_key(name)


def test_redaction_masks_credential_values_at_any_depth() -> None:
    value = {
        "headers": {"Authorization": "Bearer s", "Accept": "json"},
        "items": [{"password": "p", "name": "n"}],
        "authorizedKeys": ["ssh-ed25519 AAA"],
        "api_key": None,
        "max_tokens": 4,
    }
    assert redact_credential_fields(value) == {
        "headers": {"Authorization": REDACTED, "Accept": "json"},
        "items": [{"password": REDACTED, "name": "n"}],
        "authorizedKeys": [REDACTED],
        "api_key": None,
        "max_tokens": 4,
    }


def test_redaction_masks_the_value_of_a_name_value_pair_naming_a_credential() -> None:
    params = [
        {"name": "Authorization", "value": "Bearer s"},
        {"name": "limit", "value": "10"},
    ]
    assert redact_credential_fields(params) == [
        {"name": "Authorization", "value": REDACTED},
        {"name": "limit", "value": "10"},
    ]


def test_redaction_masks_ordered_pairs_and_returns_lists() -> None:
    pairs = [("api_key", "s"), ("model", "m")]
    assert redact_credential_fields(pairs) == [["api_key", REDACTED], ["model", "m"]]


@pytest.mark.parametrize(
    "name",
    [
        "api_key",
        "OPENAI_API_KEY",
        "apiKey",
        "secretive",
        "passwordless",
        "credentialed",
        "authorization",
        "aws_access_key_id",
        "auth_token",
        "bearer_token",
        "session_token",
        "refresh_token",
        "auth",
        "auth_mode",
        "x auth",
    ],
)
def test_redaction_covers_every_rejected_harness_param(name: str) -> None:
    assert _looks_credential(name)
    assert is_credential_key(name)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://user:pw@api.example/v1?api_key=k&limit=3",
            "https://api.example/v1?api_key=[REDACTED]&limit=3",
        ),
        ("https://tok@git.example/r.git", "https://git.example/r.git"),
        (
            "https://api.example/v1?limit=3&q=a%20b",
            "https://api.example/v1?limit=3&q=a%20b",
        ),
        (
            "https://api.example/v1?q=a%20b&flag&token=t",
            "https://api.example/v1?q=a%20b&flag&token=[REDACTED]",
        ),
        ("postgres://u:pw@db.example/app", "postgres://db.example/app"),
        ("redis://:pw@cache.example:6379/0", "redis://cache.example:6379/0"),
        ("postgres://u@db.example/app", "postgres://u@db.example/app"),
        (
            "https://maps.example/api?key=k&sig=s&signature=g",
            "https://maps.example/api?key=[REDACTED]&sig=[REDACTED]"
            "&signature=[REDACTED]",
        ),
        (
            "https://app.example/cb#access_token=t&token_type=bearer",
            "https://app.example/cb#access_token=[REDACTED]&token_type=bearer",
        ),
        (
            "https://app.example/shop;jsessionid=s?page=2",
            "https://app.example/shop;jsessionid=[REDACTED]?page=2",
        ),
        ("https://docs.example/page#key", "https://docs.example/page#key"),
        (
            "https://b.s3.example/o?X-Amz-Date=d&X-Amz-Signature=s",
            "https://b.s3.example/o?X-Amz-Date=d&X-Amz-Signature=[REDACTED]",
        ),
        ("https://api.example/v1?sort_key=a", "https://api.example/v1?sort_key=a"),
        (
            "git+https://ghp_TOKEN@github.example/o/r.git",
            "git+https://github.example/o/r.git",
        ),
        ("wss://tok@stream.example/ws", "wss://stream.example/ws"),
        (
            "https://api.example/v1?a=1;api_key=k;b=2",
            "https://api.example/v1?a=1;api_key=[REDACTED];b=2",
        ),
        (
            "https://ftp.example/get?user=u&pass=p",
            "https://ftp.example/get?user=u&pass=[REDACTED]",
        ),
    ],
)
def test_redact_url_drops_userinfo_and_masks_credential_query_values(url, expected):
    assert redact_url(url) == expected
    assert is_credential_url(url) == (url != expected)


@pytest.mark.parametrize("name", ["key", "pass", "sig", "signature", "jsessionid"])
def test_url_only_credential_names_are_not_field_credentials(name: str) -> None:
    assert not is_credential_key(name)


def test_scrubber_masks_escaped_forms_and_skips_short_words() -> None:
    pem = "-----BEGIN KEY-----\nabcdefgh\n-----END KEY-----"
    scrub = credential_scrubber([pem, {"type": "none", "token": "tok-123456"}, True])

    assert "abcdefgh" not in scrub(f"bad key {pem!r}")
    assert "abcdefgh" not in scrub(f'{{"key": {json.dumps(pem)}}}')
    assert "tok-123456" not in scrub("auth failed for tok-123456")
    assert scrub("type none, flag true") == "type none, flag true"


def test_scrubber_masks_the_credential_parts_of_a_vaulted_url() -> None:
    url = "https://u:pass-SECRET@api.example/v1?limit=3&api_key=key-SECRET"
    scrub = credential_scrubber([url])

    assert scrub("403 for /v1?limit=3&api_key=key-SECRET") == (
        "403 for /v1?limit=3&api_key=[REDACTED]"
    )
    assert "pass-SECRET" not in scrub("login u:pass-SECRET refused")


def test_scrubber_masks_a_decoded_userinfo_password() -> None:
    scrub = credential_scrubber(["https://u:p%40ss-SECRET@api.example/v1"])

    assert "ss-SECRET" not in scrub("login u:p@ss-SECRET refused")


def test_scrubber_masks_a_token_echoed_without_its_scheme() -> None:
    scrub = credential_scrubber([{"Authorization": "Bearer sk-abc123"}])

    assert scrub("401: invalid key sk-abc123") == "401: invalid key [REDACTED]"


def test_scrubber_masks_short_named_members_and_the_repr_of_a_structure() -> None:
    credentials = {"user": "bob", "password": "hunter2"}
    scrub = credential_scrubber([credentials, {"type": "none"}])

    assert "hunter2" not in scrub("login bob/hunter2 refused")
    assert "hunter2" not in scrub(f"422: {credentials!r}")
    assert scrub(f"422: {credentials!r}") == "422: [REDACTED]"
    assert scrub("type none") == "type none"
