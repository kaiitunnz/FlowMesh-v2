"""Credential-key detection and key-based redaction of nested values."""

import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"

# Matched against the name split at separators and at camelCase boundaries, so
# ``apiKey``, ``API_KEY``, ``x-api-key`` and ``apikey`` all hit ``api_key``/``apikey``.
# A bare ``_key`` suffix is not a credential shape: ``cache_key`` and ``sort_key`` are
# not.
_CREDENTIAL_SUBSTRINGS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "passphrase",
    "credential",
    "authorization",
    "cookie",
    "bearer",
    "access_key",
    "private_key",
    "signing_key",
    "ssh_key",
    "subscription_key",
    "authorized_keys",
    "authorizedkeys",
    "connection_string",
    "cert_data",
    "certificate",
    "auth_token",
    "access_token",
    "session_token",
    "refresh_token",
    "database_url",
    "db_url",
)
_CREDENTIAL_SEGMENTS = frozenset({"auth", "pwd", "dsn", "jwt"})
_CREDENTIAL_LAST_SEGMENTS = frozenset({"token"})

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[_\-\s.]+")


def _segments(name: str) -> list[str]:
    return [s for s in _SEPARATORS.split(name) if s]


def is_credential_key(name: str) -> bool:
    """Whether a header, parameter, or field name carries a credential value."""
    lowered = name.lower()
    camel = _CAMEL_BOUNDARY.sub("_", name).lower()
    for segments in (_segments(lowered), _segments(camel)):
        joined = "_".join(segments)
        if any(sub in joined for sub in _CREDENTIAL_SUBSTRINGS):
            return True
        if _CREDENTIAL_SEGMENTS.intersection(segments):
            return True
        if segments and segments[-1] in _CREDENTIAL_LAST_SEGMENTS:
            return True
    return False


def is_credential_url(value: str) -> bool:
    """Whether an ``http(s)`` URL carries a credential in its userinfo or query."""
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return False
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return False
    if "@" in parts.netloc:
        return True
    query = parse_qsl(parts.query, keep_blank_values=True)
    return any(is_credential_key(name) for name, _ in query)


def redact_url(url: str) -> str:
    """``url`` without its userinfo and with each credential query value masked."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return REDACTED
    query = parse_qsl(parts.query, keep_blank_values=True)
    if "@" not in parts.netloc and not any(is_credential_key(k) for k, _ in query):
        return url
    return urlunsplit(
        parts._replace(
            netloc=parts.netloc.rpartition("@")[2],
            query=urlencode(
                [(k, REDACTED if is_credential_key(k) else v) for k, v in query],
                safe="[]",
            ),
        )
    )


def masked(value: Any) -> Any:
    """The marker a credential value is replaced by."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [REDACTED]
    return REDACTED


def _names_credential(value: dict[Any, Any]) -> bool:
    """Whether a mapping is a ``{name, value}`` pair naming a credential."""
    name = value.get("name")
    return "value" in value and isinstance(name, str) and is_credential_key(name)


def redact_credential_fields(value: Any) -> Any:
    """A copy of ``value`` with every credential value replaced, at any depth.

    A credential value is one under a credential-looking key, the ``value`` of a
    ``{name, value}`` pair whose ``name`` is one, or an ``http(s)`` URL carrying a
    credential. A ``None`` value is kept, so an omitted credential reads as omitted;
    tuples come back as lists.
    """
    if isinstance(value, str) and is_credential_url(value):
        return REDACTED
    if isinstance(value, dict):
        redacted = {
            key: (
                masked(val)
                if is_credential_key(str(key))
                else redact_credential_fields(val)
            )
            for key, val in value.items()
        }
        if _names_credential(value):
            redacted["value"] = masked(value["value"])
        return redacted
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and is_credential_key(str(value[0]))
    ):
        return [value[0], masked(value[1])]
    if isinstance(value, (list, tuple)):
        return [redact_credential_fields(item) for item in value]
    return value


type CredentialPath = tuple[str | int, ...]


def find_credential_values(
    value: Any, path: CredentialPath = ()
) -> Iterator[tuple[CredentialPath, Any]]:
    """Each credential value in a JSON value with its path, by the rules
    ``redact_credential_fields`` masks them."""
    if isinstance(value, str):
        if is_credential_url(value):
            yield path, value
    elif isinstance(value, dict):
        for key, val in value.items():
            if val is None:
                continue
            if is_credential_key(str(key)):
                yield (*path, key), val
            elif not (key == "value" and _names_credential(value)):
                yield from find_credential_values(val, (*path, key))
        if _names_credential(value) and value["value"] is not None:
            yield (*path, "value"), value["value"]
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from find_credential_values(item, (*path, index))


__all__ = [
    "REDACTED",
    "CredentialPath",
    "find_credential_values",
    "is_credential_key",
    "is_credential_url",
    "masked",
    "redact_credential_fields",
    "redact_url",
]
