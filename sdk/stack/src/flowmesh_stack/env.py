"""Environment file helpers shared by FlowMesh tooling."""

import os
import re
from collections import ChainMap
from collections.abc import Iterator, Mapping
from pathlib import Path
from urllib.parse import urlparse


class EnvFileError(ValueError):
    """An env file Docker Compose would refuse to read."""


def parse_env_file(env_file: Path) -> dict[str, str]:
    """Parse a .env file into key/value pairs, following Docker Compose's dotenv
    quoting, escape and interpolation rules."""
    if not env_file.exists():
        return {}
    return _parse_file(env_file, os.environ)


def _parse_file(env_file: Path, environ: Mapping[str, str]) -> dict[str, str]:
    try:
        return parse_env_text(env_file.read_text(), environ)
    except EnvFileError as exc:
        raise EnvFileError(f"{env_file}: {exc}") from exc


def parse_bool(value: str) -> bool | None:
    """Parse a string into a boolean value."""
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def parse_int(value: str) -> int | None:
    """Parse a string into an integer value."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    """Parse a string into a float value."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def is_url(value: str, schemes: set[str] | None = None) -> bool:
    """Check if a string is a valid URL with optional scheme restrictions."""
    parsed = urlparse(value.strip())
    if not (parsed.scheme and parsed.netloc):
        return False
    if schemes and parsed.scheme not in schemes:
        return False
    return True


def validate_env_file(
    env_file: Path,
    example: Path | None = None,
    expected_keys: set[str] | None = None,
) -> tuple[dict[str, str] | None, list[str]]:
    """Validate an env file against an example template or key set."""
    errors: list[str] = []
    if not env_file.exists():
        return None, [f"env file not found: {env_file}"]
    if expected_keys is None:
        if example is None or not example.exists():
            try:
                return parse_env_file(env_file), errors
            except EnvFileError as exc:
                return None, [str(exc)]
        expected_keys = _parse_env_keys(example)

    try:
        actual_keys = _parse_env_keys(env_file)
        values = parse_env_file(env_file)
    except EnvFileError as exc:
        return None, [str(exc)]
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"Missing required env vars in {env_file}: {', '.join(missing)}")
    if unexpected:
        errors.append(f"Unexpected env vars in {env_file}: {', '.join(unexpected)}")
    return values, errors


def ensure_env_file(env_file: Path, example: Path) -> bool:
    """Create an env file from an example if it does not exist."""
    if env_file.exists() or not example.exists():
        return False
    env_file.write_text(example.read_text())
    return True


def load_env(
    env_file: Path,
    base_dir: Path | None = None,
    path_keys: set[str] | None = None,
) -> None:
    """Load env vars from a file into ``os.environ``, following Docker Compose's
    dotenv quoting, escape and interpolation rules.

    Raises ``EnvFileError`` for a file Docker Compose would refuse to read.
    """
    env_key = (env_file, base_dir, path_keys)
    if getattr(load_env, "_loaded", None) == env_key:
        return
    if not env_file.exists():
        return
    for key, value in _parse_file(env_file, os.environ).items():
        if path_keys and key in path_keys and value:
            expanded = Path(value).expanduser()
            if expanded.is_absolute():
                os.environ[key] = str(expanded)
            elif base_dir is not None:
                os.environ[key] = str((base_dir / expanded).resolve())
            else:
                os.environ[key] = value
        else:
            os.environ[key] = value
    load_env._loaded = env_key  # type: ignore[attr-defined]


def _parse_env_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return {key for key, _, _ in _entries(path.read_text())}
    except EnvFileError as exc:
        raise EnvFileError(f"{path}: {exc}") from exc


_DOUBLE_QUOTED_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BRACED = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-+?])(.*))?\}", re.DOTALL)
_EXPORT = re.compile(r"^export\s+")


def parse_env_text(text: str, environ: Mapping[str, str]) -> dict[str, str]:
    """Parse .env text following Docker Compose's dotenv quoting, escape and
    interpolation rules.

    A double-quoted value takes ``\\n``, ``\\r``, ``\\t``, ``\\\\``, ``\\"`` and
    ``\\$`` escapes and may span lines; a single-quoted value is literal but for
    ``\\'``; an unquoted value ends at a `` #`` comment. Unquoted and double-quoted
    values interpolate ``$NAME`` and ``${NAME}`` with the ``:-``, ``-``, ``:+``, ``+``,
    ``:?`` and ``?`` modifiers, reading the file's earlier keys, then ``environ``, then
    its later keys; ``$$`` is a literal ``$``. Raises ``EnvFileError`` where Compose
    fails.
    """
    entries = list(_entries(text))
    ahead = _interpolate(entries, environ, {}, strict=False)
    return _interpolate(entries, environ, ahead, strict=True)


def _interpolate(
    entries: list[tuple[str, str, str]],
    environ: Mapping[str, str],
    ahead: Mapping[str, str],
    strict: bool,
) -> dict[str, str]:
    values: dict[str, str] = {}
    later = dict(ahead)
    lookup = ChainMap(values, dict(environ), later)
    for key, body, quote in entries:
        later.pop(key, None)
        if quote == "'":
            values[key] = body
        elif quote == '"':
            values[key] = _expand(_unescape(body), lookup, key, strict)
        else:
            values[key] = _expand(body, lookup, key, strict)
    return values


def _entries(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield each assignment's key, its value before interpolation, and its quote."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = _EXPORT.sub("", line, count=1).split("=", 1)
        key = key.strip()
        raw = raw.lstrip()
        quote = raw[:1]
        if quote not in ("'", '"'):
            yield key, _strip_comment(raw), ""
            continue
        body = raw[1:]
        while (end := _closing_quote(body, quote)) is None:
            if index >= len(lines):
                raise EnvFileError(f"{key}: unterminated quoted value")
            body += "\n" + lines[index]
            index += 1
        body = body[:end]
        yield key, body.replace("\\'", "'") if quote == "'" else body, quote


def _closing_quote(body: str, quote: str) -> int | None:
    escaped = False
    for position, char in enumerate(body):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == quote and not escaped:
            return position
        escaped = False
    return None


def _unescape(body: str) -> str:
    out: list[str] = []
    chars = iter(body)
    for char in chars:
        if char != "\\":
            out.append(char)
            continue
        following = next(chars, "")
        if following == "$":
            out.append("\0")
        else:
            out.append(_DOUBLE_QUOTED_ESCAPES.get(following, "\\" + following))
    return "".join(out)


def _strip_comment(raw: str) -> str:
    comment = raw.find(" #")
    return (raw if comment < 0 else raw[:comment]).rstrip()


def _expand(value: str, lookup: Mapping[str, str], key: str, strict: bool) -> str:
    out: list[str] = []
    position = 0
    while position < len(value):
        char = value[position]
        if char == "\0":
            out.append("$")
            position += 1
            continue
        if char != "$":
            out.append(char)
            position += 1
            continue
        following = value[position + 1 : position + 2]
        if following == "$":
            out.append("$")
            position += 2
        elif following == "{" and (end := _closing_brace(value, position + 1)):
            out.append(_substitute(value[position + 1 : end + 1], lookup, key, strict))
            position = end + 1
        elif name := _NAME.match(value, position + 1):
            out.append(lookup.get(name.group(), ""))
            position = name.end()
        else:
            out.append("$")
            position += 1
    return "".join(out)


def _closing_brace(value: str, start: int) -> int | None:
    depth = 0
    for position in range(start, len(value)):
        if value[position] == "{":
            depth += 1
        elif value[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None


def _substitute(braced: str, lookup: Mapping[str, str], key: str, strict: bool) -> str:
    match = _BRACED.fullmatch(braced)
    if match is None:
        return "$" + braced
    name, modifier, word = match.groups()
    current = lookup.get(name)
    if modifier is None:
        return current or ""
    if current is not None and not current and modifier.startswith(":"):
        current = None
    if modifier.endswith("-"):
        return current if current is not None else _expand(word, lookup, key, strict)
    if modifier.endswith("+"):
        return _expand(word, lookup, key, strict) if current is not None else ""
    if current is None and strict:
        message = _expand(word, lookup, key, strict) or "is required"
        raise EnvFileError(f"{key}: {name} {message}")
    return current or ""
