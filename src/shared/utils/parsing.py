import os
import re
from collections.abc import Mapping
from typing import Any, overload

from pydantic import SecretStr


@overload
def to_int(value: Any, default: int) -> int: ...


@overload
def to_int(value: Any, default: int | None = None) -> int | None: ...


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@overload
def safe_int(
    value: Any,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int: ...


@overload
def safe_int(
    value: Any,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None: ...


def safe_int(
    value: Any,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    result = to_int(value, default=default)
    if result is None:
        return None
    if minimum is not None and result < minimum:
        return minimum
    if maximum is not None and result > maximum:
        return maximum
    return result


@overload
def to_float(value: Any, default: float) -> float: ...


@overload
def to_float(value: Any, default: float | None = None) -> float | None: ...


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@overload
def safe_float(
    value: Any,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float: ...


@overload
def safe_float(
    value: Any,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None: ...


def safe_float(
    value: Any,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    result = to_float(value, default=default)
    if result is None:
        return None
    if minimum is not None and result < minimum:
        return minimum
    if maximum is not None and result > maximum:
        return maximum
    return result


@overload
def to_bool(value: Any, default: bool) -> bool: ...


@overload
def to_bool(value: Any, default: bool | None = None) -> bool | None: ...


def to_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def to_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, list):
        parsed: list[int] = []
        for item in value:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        return parsed or list(default)
    if isinstance(value, str):
        parsed = []
        for part in value.split(","):
            part = part.strip()
            if part:
                try:
                    parsed.append(int(part))
                except (TypeError, ValueError):
                    continue
        return parsed or list(default)
    return list(default)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


@overload
def parse_int_env(name: str, default: int) -> int: ...


@overload
def parse_int_env(name: str, default: int | None = None) -> int | None: ...


def parse_int_env(name: str, default: int | None = None) -> int | None:
    return to_int(os.getenv(name), default=default)


@overload
def parse_float_env(name: str, default: float) -> float: ...


@overload
def parse_float_env(name: str, default: float | None = None) -> float | None: ...


def parse_float_env(name: str, default: float | None = None) -> float | None:
    return to_float(os.getenv(name), default=default)


@overload
def parse_bool_env(name: str, default: bool) -> bool: ...


@overload
def parse_bool_env(name: str, default: bool | None = None) -> bool | None: ...


def parse_bool_env(name: str, default: bool | None = None) -> bool | None:
    return to_bool(os.getenv(name), default=default)


def content_hydration_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether workers keep the content cache and hydrate content between them: as
    ``CONTENT_HYDRATION_ENABLED`` says, and when it is unset, whenever the network
    plane that carries the transfers is on.

    Raises ``ValueError`` when it is turned on with the network plane off.
    """
    env = os.environ if environ is None else environ
    plane = to_bool(env.get("NETWORK_PLANE_ENABLED"), default=True)
    explicit = to_bool(env.get("CONTENT_HYDRATION_ENABLED"))
    if explicit and not plane:
        raise ValueError(
            "CONTENT_HYDRATION_ENABLED requires NETWORK_PLANE_ENABLED: a content "
            "transfer is carried by the network plane's relay"
        )
    return plane if explicit is None else explicit


def parse_secret_env(name: str) -> SecretStr | None:
    value = (os.getenv(name) or "").strip()
    return SecretStr(value) if value else None


def parse_mem_to_bytes(mem: str) -> int | None:
    if not isinstance(mem, str):
        return None
    match = re.match(r"^\s*([0-9]+)\s*([KkMmGgTt][Ii]?[Bb]?)?\s*$", mem)
    if not match:
        return None
    qty = int(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit in {"k", "kb", "ki", "kib"}:
        return qty * 1024
    if unit in {"m", "mb", "mi", "mib"}:
        return qty * 1024**2
    if unit in {"g", "gb", "gi", "gib"}:
        return qty * 1024**3
    if unit in {"t", "tb", "ti", "tib"}:
        return qty * 1024**4
    if unit == "":
        return qty
    return None
