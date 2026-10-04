"""Credential-key detection and key-based redaction of nested values."""

import json
import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib.parse import (
    SplitResult,
    quote,
    quote_plus,
    unquote,
    unquote_plus,
    urlsplit,
    urlunsplit,
)

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


# Names that carry a credential only as a URL parameter: a Google ``?key=``, a servlet
# session id in a ``;jsessionid=`` path parameter, and a signature however it is
# prefixed (an Azure SAS ``sig``, an S3 ``X-Amz-Signature``).
_URL_CREDENTIAL_NAMES = frozenset({"key", "pass", "jsessionid"})
_URL_CREDENTIAL_LAST_SEGMENTS = frozenset({"sig", "signature"})
_PATH_PARAMETER = re.compile(r";([^;/=]*)=([^;/]*)")
_PARAMETER_SEPARATOR = re.compile(r"([&;])")
# Schemes whose userinfo is a credential even without a password: a token-only
# ``https://TOKEN@host``.
_TOKEN_USERINFO_SCHEMES = frozenset(
    {"http", "https", "git+http", "git+https", "ws", "wss"}
)


def _is_url_credential_name(name: str) -> bool:
    name = unquote_plus(name)
    segments = _segments(name.lower())
    return (
        name.lower() in _URL_CREDENTIAL_NAMES
        or bool(segments and segments[-1] in _URL_CREDENTIAL_LAST_SEGMENTS)
        or is_credential_key(name)
    )


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
    return parts.scheme.lower() in _TOKEN_USERINFO_SCHEMES or ":" in userinfo


def _parameters(text: str) -> list[str]:
    return _PARAMETER_SEPARATOR.split(text)[::2]


def _parameter_names(text: str) -> Iterator[str]:
    for pair in _parameters(text):
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
    ``http(s)``, ``git+http(s)`` and ``ws(s)``), or a credential in its query,
    fragment, or path parameters."""
    parts = _split_url(value)
    if parts is None:
        return False
    return _userinfo_is_credential(parts) or _has_credential_parameter(parts)


def _url_credentials(url: str) -> list[str]:
    """The credential parts of a URL: its userinfo and each credential parameter's
    value, raw and decoded."""
    parts = _split_url(url)
    if parts is None:
        return []
    found: list[str] = []
    if _userinfo_is_credential(parts):
        userinfo = parts.netloc.rpartition("@")[0]
        for part in (userinfo, *userinfo.split(":", 1)):
            found.extend((part, unquote(part)))
    pairs = [
        pair.partition("=")
        for text in (parts.query, parts.fragment)
        for pair in _parameters(text)
    ]
    pairs.extend(
        (m.group(1), "=", m.group(2)) for m in _PATH_PARAMETER.finditer(parts.path)
    )
    for name, equals, value in pairs:
        if equals and value and _is_url_credential_name(name):
            found.extend((value, unquote_plus(value)))
    return found


def _redact_parameter(pair: str) -> str:
    name, equals, _ = pair.partition("=")
    return f"{name}={REDACTED}" if equals and _is_url_credential_name(name) else pair


def _redact_parameters(text: str) -> str:
    return "".join(
        _redact_parameter(piece) if index % 2 == 0 else piece
        for index, piece in enumerate(_PARAMETER_SEPARATOR.split(text))
    )


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


# A credential string shorter than the first bound is too common a word to scrub, and
# so is a string inside a structured credential (``type: none``) shorter than the
# second, unless a credential-named key holds it.
_MIN_SCRUBBED_LENGTH = 4
_MIN_SCRUBBED_MEMBER_LENGTH = 8
# An ``Authorization``-style ``<scheme> <token>`` value; its token may be echoed alone.
_SCHEME_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9._+-]*\s+(\S+)")


def _members(value: Any, named: bool) -> Iterator[tuple[str, bool]]:
    """Each string in a structured value, with whether a credential names it."""
    if isinstance(value, str):
        yield value, named
    elif isinstance(value, dict):
        for key, item in value.items():
            credential = (
                named
                or is_credential_key(str(key))
                or (key == "value" and _names_credential(value))
            )
            yield from _members(item, credential)
    elif isinstance(value, list):
        for item in value:
            yield from _members(item, named)


def _credential_parts(text: str) -> Iterator[str]:
    yield text
    yield from _url_credentials(text)
    if token := _SCHEME_TOKEN.fullmatch(text.strip()):
        yield token.group(1)


def credential_scrubber(values: Iterable[Any]) -> Callable[[str], str]:
    """A function masking every occurrence of ``values`` in a text.

    Each string is matched as written and in its JSON- and ``repr``-escaped and
    percent-encoded forms, so a multi-line key quoted in an error, or a key an HTTP
    client logs inside a request URL, is masked too; a URL's credential parts and an
    ``Authorization`` value's token are matched on their own, so a text quoting only
    part of one is masked too. A structured value is also matched whole, as JSON and as
    a Python ``repr``.
    """
    needles: set[str] = set()
    for value in values:
        structured = isinstance(value, (dict, list))
        for text, named in _members(value, not structured):
            minimum = _MIN_SCRUBBED_LENGTH if named else _MIN_SCRUBBED_MEMBER_LENGTH
            for part in _credential_parts(text):
                if len(part) >= minimum:
                    needles.update(
                        (
                            part,
                            json.dumps(part)[1:-1],
                            repr(part)[1:-1],
                            quote(part, safe=""),
                            quote(part),
                            quote_plus(part),
                        )
                    )
        if structured:
            needles.update((json.dumps(value), repr(value)))
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
