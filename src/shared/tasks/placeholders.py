import re
from typing import Annotated, Any

from pydantic import AfterValidator

PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def is_placeholder(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(PLACEHOLDER_PATTERN.fullmatch(value.strip()))


def contains_placeholder(value: Any) -> bool:
    """Whether a value renders from an upstream stage at dispatch."""
    if isinstance(value, str):
        return bool(PLACEHOLDER_PATTERN.search(value))
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    return False


def _validate_placeholder_string(value: Any) -> str:
    if not isinstance(value, str) or not is_placeholder(value):
        raise ValueError("Expected a placeholder string like ${...}")
    return value


type PlaceholderString = Annotated[str, AfterValidator(_validate_placeholder_string)]

type TemplateBool = bool | PlaceholderString
type TemplateInt = int | PlaceholderString
type TemplateFloat = float | PlaceholderString
