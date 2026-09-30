"""Credential-key detection and key-based redaction of nested values."""

import json
import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib.parse import SplitResult, unquote_plus, urlsplit, urlunsplit

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


# Names that carry a credential only as a URL parameter: a Google ``?key=``, an Azure
# SAS ``sig``, a servlet session id in a ``;jsessionid=`` path parameter.
_URL_CREDENTIAL_NAMES = frozenset({"key", "sig", "signature", "jsessionid"})
_PATH_PARAMETER = re.compile(r";([^;/=]*)=([^;/]*)")


def _is_url_credential_name(name: str) -> bool:
    name = unquote_plus(name)
    return name.lower() in _URL_CREDENTIAL_NAMES or is_credential_key(name)


def _split_url(value: str) -> SplitResult | None:
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return None
    return parts if parts.scheme and parts.netloc else None


def _userinfo_is_credential(parts: SplitResult) -> bool:
    userinfo, at, _ = parts.netloc.rpartition("@")
    if not at:
        return False
    return parts.scheme.lower() in ("http", "https") or ":" in userinfo


def _parameter_names(text: str) -> Iterator[str]:
    for pair in text.split("&"):
        name, equals, _ = pair.partition("=")
        if equals:
            yield name


def _has_credential_parameter(parts: SplitResult) -> bool:
    return any(
        _is_url_credential_name(name)
        for text in (parts.query, parts.fragment)
        for name in _parameter_names(text)
    ) or any(
        _is_url_credential_name(match.group(1))
        for match in _PATH_PARAMETER.finditer(parts.path)
    )


def is_credential_url(value: str) -> bool:
    """Whether a URL carries a credential: a userinfo password (any userinfo on
    ``http(s)``), or a credential in its query, fragment, or path parameters."""
    parts = _split_url(value)
    if parts is None:
        return False
    return _userinfo_is_credential(parts) or _has_credential_parameter(parts)


def _redact_parameter(pair: str) -> str:
    name, equals, _ = pair.partition("=")
    return f"{name}={REDACTED}" if equals and _is_url_credential_name(name) else pair


def _redact_parameters(text: str) -> str:
    return "&".join(_redact_parameter(pair) for pair in text.split("&"))


def redact_url(url: str) -> str:
    """``url`` without a credential userinfo and with each credential parameter's value
    masked; every other part keeps its original encoding."""
    parts = _split_url(url)
    if parts is None:
        return url
    if not (_userinfo_is_credential(parts) or _has_credential_parameter(parts)):
        return url
    return urlunsplit(
        parts._replace(
            netloc=(
                parts.netloc.rpartition("@")[2]
                if _userinfo_is_credential(parts)
                else parts.netloc
            ),
            path=_PATH_PARAMETER.sub(
                lambda match: (
                    f";{match.group(1)}={REDACTED}"
                    if _is_url_credential_name(match.group(1))
                    else match.group(0)
                ),
                parts.path,
            ),
            query=_redact_parameters(parts.query),
            fragment=_redact_parameters(parts.fragment),
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
    ``{name, value}`` pair whose ``name`` is one, or a URL carrying a credential. A
    ``None`` value is kept, so an omitted credential reads as omitted; tuples come back
    as lists.
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


# A whole-string credential shorter than the first bound is too common a word to scrub,
# and so is a string inside a structured credential (``type: none``) shorter than the
# second.
_MIN_SCRUBBED_LENGTH = 4
_MIN_SCRUBBED_MEMBER_LENGTH = 8


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def credential_scrubber(values: Iterable[Any]) -> Callable[[str], str]:
    """A function masking every occurrence of ``values`` in a text.

    Each string is matched as written and in its JSON- and ``repr``-escaped forms, so a
    multi-line key quoted in an error is masked too.
    """
    needles: set[str] = set()
    for value in values:
        minimum = (
            _MIN_SCRUBBED_LENGTH
            if isinstance(value, str)
            else _MIN_SCRUBBED_MEMBER_LENGTH
        )
        for text in _strings(value):
            if len(text) >= minimum:
                needles.update((text, json.dumps(text)[1:-1], repr(text)[1:-1]))
        if isinstance(value, (dict, list)):
            needles.add(json.dumps(value))
    ordered = sorted(needles, key=len, reverse=True)

    def scrub(text: str) -> str:
        for needle in ordered:
            text = text.replace(needle, REDACTED)
        return text

    return scrub


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
    "credential_scrubber",
    "find_credential_values",
    "is_credential_key",
    "is_credential_url",
    "masked",
    "redact_credential_fields",
    "redact_url",
]
