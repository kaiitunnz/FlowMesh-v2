"""The server's OpenAPI document, without the schemas of server-internal models."""

from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI

INTERNAL_SCHEMA: dict[str, Any] = {"x-internal": True}
"""The ``json_schema_extra`` that marks a model as server-internal."""

_SCHEMA_REF = "#/components/schemas/"


class FlowMeshAPI(FastAPI):
    """A FastAPI app whose OpenAPI document leaves out server-internal models."""

    def openapi(self) -> dict[str, Any]:
        return drop_internal_schemas(super().openapi())


def drop_internal_schemas(document: dict[str, Any]) -> dict[str, Any]:
    """Remove, in place, every component schema marked server-internal and every
    one that only those reach, and return the document."""
    schemas = document.get("components", {}).get("schemas", {})
    dropped = {name for name, schema in schemas.items() if schema.get("x-internal")}
    while dropped:
        reached = {ref for name in dropped for ref in _refs(schemas.pop(name))}
        dropped = (reached & schemas.keys()) - set(_refs(document))
    return document


def _refs(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)
    elif isinstance(value, str) and value.startswith(_SCHEMA_REF):
        yield value.removeprefix(_SCHEMA_REF)
