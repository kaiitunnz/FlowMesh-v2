"""Query filters and cursor pages for the list routes."""

import inspect
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Annotated

from fastapi import Query, Request, status

from ...utils.cursors import InvalidCursor
from ...utils.query import InvalidQuery, QueryFilter
from ._errors import api_error

PAGE_LIMIT_DEFAULT = 100

PageLimit = Annotated[
    int, Query(ge=1, le=1000, description="Maximum entries to return.")
]
PageAfter = Annotated[
    str | None, Query(description="Return entries strictly after this cursor.")
]
PageBefore = Annotated[
    str | None, Query(description="Return entries strictly before this cursor.")
]
PAGE_PARAMS = frozenset({"limit", "after", "before"})


@dataclass(frozen=True, slots=True)
class ListFilter:
    """A list route's filter over its declared fields, parsed when the route reads
    it, so a route can authorize the caller first."""

    request: Request
    fields: Collection[str]
    params: Collection[str]

    def parse(self) -> QueryFilter:
        """Parse the request's filter past the route's own ``params``; reject any
        other key with a 400."""
        try:
            return QueryFilter.parse(
                self.request.query_params, self.fields, self.params
            )
        except InvalidQuery as exc:
            raise api_error(
                status.HTTP_400_BAD_REQUEST, "invalid_request", str(exc)
            ) from exc


def filter_params(
    fields: Collection[str], params: Collection[str] = ()
) -> Callable[..., ListFilter]:
    """A dependency declaring one optional query parameter per filter field, so the
    route's schema lists its filters, and yielding the route's ``ListFilter``."""

    def dependency(request: Request, **_: list[str] | None) -> ListFilter:
        return ListFilter(request, fields, params)

    declared = [
        inspect.Parameter(
            f"filter_{index}",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=Annotated[
                list[str] | None,
                Query(
                    alias=field,
                    description=(
                        f"Match `{field}`; a repeated key matches any of its values."
                    ),
                ),
            ],
        )
        for index, field in enumerate(sorted(fields))
    ]
    setattr(
        dependency,
        "__signature__",
        inspect.Signature(
            [
                inspect.Parameter(
                    "request",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Request,
                ),
                *declared,
            ]
        ),
    )
    return dependency


def page_bounds[K](
    after: str | None, before: str | None, decode: Callable[[str], K]
) -> tuple[K | None, K | None]:
    """Decode the ``after``/``before`` cursors; reject both set, or one malformed,
    with a 400."""
    if after and before:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "only one of before/after may be set",
        )
    try:
        return (
            decode(after) if after else None,
            decode(before) if before else None,
        )
    except InvalidCursor as exc:
        raise api_error(
            status.HTTP_400_BAD_REQUEST, "invalid_cursor", str(exc)
        ) from exc


__all__ = [
    "ListFilter",
    "PAGE_LIMIT_DEFAULT",
    "PAGE_PARAMS",
    "PageAfter",
    "PageBefore",
    "PageLimit",
    "filter_params",
    "page_bounds",
]
