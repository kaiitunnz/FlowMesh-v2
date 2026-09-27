"""Credential-key detection by name."""

import re

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
    return [s for s in _SEPARATORS.split(_CAMEL_BOUNDARY.sub("_", name).lower()) if s]


def is_credential_key(name: str) -> bool:
    """Whether a header, parameter, or field name carries a credential value."""
    segments = _segments(name)
    forms = (name.lower(), "_".join(segments))
    if any(sub in form for form in forms for sub in _CREDENTIAL_SUBSTRINGS):
        return True
    if _CREDENTIAL_SEGMENTS.intersection(segments):
        return True
    return bool(segments) and segments[-1] in _CREDENTIAL_LAST_SEGMENTS


__all__ = ["is_credential_key"]
