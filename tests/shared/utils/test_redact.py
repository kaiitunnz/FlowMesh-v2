import pytest

from shared.utils.redact import (
    REDACTED,
    find_credential_key,
    is_credential_key,
    redact_credential_fields,
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
    ],
)
def test_ordinary_keys_do_not_match(name: str) -> None:
    assert not is_credential_key(name)


def test_redaction_masks_credential_values_at_any_depth() -> None:
    value = {
        "headers": {"Authorization": "Bearer s", "Accept": "json"},
        "items": [{"password": "p", "name": "n"}],
        "api_key": None,
        "max_tokens": 4,
    }
    assert redact_credential_fields(value) == {
        "headers": {"Authorization": REDACTED, "Accept": "json"},
        "items": [{"password": REDACTED, "name": "n"}],
        "api_key": None,
        "max_tokens": 4,
    }


def test_find_credential_key_reports_the_first_nested_one() -> None:
    assert find_credential_key({"a": [{"b": {"x-api-key": 1}}]}) == "x-api-key"
    assert find_credential_key({"max_tokens": 1, "cache_key": 2}) is None
