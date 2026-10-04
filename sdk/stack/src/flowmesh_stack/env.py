"""Environment file helpers shared by FlowMesh tooling."""

import os
import re
from collections import ChainMap
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse


def parse_env_file(env_file: Path) -> dict[str, str]:
    """Parse a .env file into key/value pairs, as Docker Compose reads it."""
    if not env_file.exists():
        return {}
    return parse_env_text(env_file.read_text(), os.environ)


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
            return parse_env_file(env_file), errors
        expected_keys = _parse_env_keys(example)

    actual_keys = _parse_env_keys(env_file)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"Missing required env vars in {env_file}: {', '.join(missing)}")
    if unexpected:
        errors.append(f"Unexpected env vars in {env_file}: {', '.join(unexpected)}")
    return parse_env_file(env_file), errors


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
    """Load env vars from a file into ``os.environ``, as Docker Compose reads it."""
    env_key = (env_file, base_dir, path_keys)
    if getattr(load_env, "_loaded", None) == env_key:
        return
    if not env_file.exists():
        return
    for key, value in parse_env_text(env_file.read_text(), os.environ).items():
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
    return set(parse_env_text(path.read_text(), {}))


_DOUBLE_QUOTED_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BRACED = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-+?])(.*))?\}", re.DOTALL)


def parse_env_text(text: str, environ: Mapping[str, str]) -> dict[str, str]:
    """Parse .env text the way Docker Compose's ``env_file`` and ``--env-file`` do.

    A double-quoted value takes ``\\n``, ``\\r``, ``\\t``, ``\\\\``, ``\\"`` and
    ``\\$`` escapes and may span lines; a single-quoted value is literal; an unquoted
    value ends at a `` #`` comment. Unquoted and double-quoted values interpolate
    ``$NAME`` and ``${NAME}`` with the ``:-``, ``-``, ``:+``, ``+``, ``:?`` and ``?``
    modifiers, reading the file's earlier keys and then ``environ``; ``$$`` is a literal
    ``$``.
    """
    values: dict[str, str] = {}
    lookup = ChainMap(values, dict(environ))
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.removeprefix("export ").split("=", 1)
        key = key.strip()
        raw = raw.lstrip()
        quote = raw[:1]
        if quote in ("'", '"'):
            body = raw[1:]
            while (end := _closing_quote(body, quote)) is None:
                if index >= len(lines):
                    raise ValueError(f"{key}: unterminated quoted value")
                body += "\n" + lines[index]
                index += 1
            body = body[:end]
            values[key] = (
                body if quote == "'" else _expand(_unescape(body), lookup, key)
            )
        else:
            values[key] = _expand(_strip_comment(raw), lookup, key)
    return values


def _closing_quote(body: str, quote: str) -> int | None:
    escaped = False
    for position, char in enumerate(body):
        if quote == '"' and char == "\\" and not escaped:
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
    match = re.search(r"\s#", raw)
    return (raw[: match.start()] if match else raw).rstrip()


def _expand(value: str, lookup: Mapping[str, str], key: str) -> str:
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
            out.append(_substitute(value[position + 1 : end + 1], lookup, key))
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


def _substitute(braced: str, lookup: Mapping[str, str], key: str) -> str:
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
        return current if current is not None else _expand(word, lookup, key)
    if modifier.endswith("+"):
        return _expand(word, lookup, key) if current is not None else ""
    if current is None:
        raise ValueError(f"{key}: {name} {_expand(word, lookup, key) or 'is required'}")
    return current
