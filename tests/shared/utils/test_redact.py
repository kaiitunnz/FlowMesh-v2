import pytest

from shared.utils.redact import is_credential_key


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
