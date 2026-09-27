"""Credential-key detection and key-based redaction of nested values.

A key names a credential when its name looks like one (see ``is_credential_key``);
redaction replaces the value under every such key with a fixed marker, at any depth.
"""

import re
from typing import Any

REDACTED = "[REDACTED]"

# Matched against both the lowercased name and its segmented form, so ``apiKey``,
# ``API_KEY``, ``x-api-key`` and ``apikey`` all hit ``api_key``/``apikey``. A bare
# ``_key`` suffix is not a credential shape: ``cache_key`` and ``sort_key`` are not.
_CREDENTIAL_SUBSTRINGS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "credential",
    "authorization",
    "cookie",
    "bearer",
    "access_key",
    "private_key",
    "secret_key",
    "signing_key",
    "auth_token",
    "access_token",
    "session_token",
    "refresh_token",
)
_CREDENTIAL_SEGMENTS = frozenset({"auth"})
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
    return any(sub in lowered for sub in _CREDENTIAL_SUBSTRINGS)


def find_credential_key(value: Any) -> str | None:
    """The first credential-looking key anywhere in a nested structure."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if is_credential_key(str(key)):
                return str(key)
            if (found := find_credential_key(nested)) is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            if (found := find_credential_key(item)) is not None:
                return found
    return None


def redact_credential_fields(value: Any) -> Any:
    """A copy of ``value`` with every credential-keyed value replaced, at any depth.

    A ``None`` value stays ``None`` so an omitted credential still reads as omitted.
    """
    if isinstance(value, dict):
        return {
            key: (
                (None if val is None else REDACTED)
                if is_credential_key(str(key))
                else redact_credential_fields(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_credential_fields(item) for item in value]
    return value


__all__ = [
    "REDACTED",
    "find_credential_key",
    "is_credential_key",
    "redact_credential_fields",
]
