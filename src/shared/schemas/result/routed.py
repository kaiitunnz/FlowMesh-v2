"""A value an input carries that is not one whole task result.

Every input a task reads is a value: a whole upstream result, a part of one, an
aggregate's members, a literal, or an explicit empty. Upstream values travel to a worker
in one typed map of results, so a value that is not a whole result rides in it as a
``RoutedValue``, and every reader sees through it to the value it carries.
"""

from typing import Any

from pydantic import ConfigDict, SerializerFunctionWrapHandler, model_serializer

from ._base import BaseExecutorResult


class RoutedValue(BaseExecutorResult):
    """An input value carried where task results travel; readers see through it to
    ``routed_value``."""

    model_config = ConfigDict(extra="forbid")

    routed_value: Any

    @model_serializer(mode="wrap")
    def _drop_none_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # An explicit empty is a value, so it survives a dump that drops nulls.
        dumped = BaseExecutorResult._drop_none_fields(self, handler)
        dumped.setdefault("routed_value", None)
        return dumped


def routed_root(value: Any) -> Any:
    """The value a reader sees: what a ``RoutedValue`` carries, else ``value``."""
    return value.routed_value if isinstance(value, RoutedValue) else value


__all__ = ["RoutedValue", "routed_root"]
