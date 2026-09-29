"""Credential-key detection and key-based redaction of nested values."""

import re
from typing import Any

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


def _masked(value: Any) -> Any:
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

    A credential value is one under a credential-looking key, or the ``value`` of a
    ``{name, value}`` pair whose ``name`` is one. A ``None`` value is kept, so an
    omitted credential reads as omitted; tuples come back as lists.
    """
    if isinstance(value, dict):
        redacted = {
            key: (
                _masked(val)
                if is_credential_key(str(key))
                else redact_credential_fields(val)
            )
            for key, val in value.items()
        }
        if _names_credential(value):
            redacted["value"] = _masked(value["value"])
        return redacted
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and is_credential_key(str(value[0]))
    ):
        return [value[0], _masked(value[1])]
    if isinstance(value, (list, tuple)):
        return [redact_credential_fields(item) for item in value]
    return value


__all__ = ["REDACTED", "is_credential_key", "redact_credential_fields"]
