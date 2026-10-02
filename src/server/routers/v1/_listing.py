"""Query filters and cursor pages for the list routes."""

from collections.abc import Callable, Collection
from typing import Annotated

from fastapi import HTTPException, Query, Request, status

from ...utils.cursors import InvalidCursor
from ...utils.query import InvalidQuery, QueryFilter

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


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code, detail={"code": code, "message": message})


def query_filter(request: Request, fields: Collection[str]) -> QueryFilter:
    """Parse the request's filter over ``fields``; reject any other key with a
    400."""
    try:
        return QueryFilter.parse(request.query_params, fields)
    except InvalidQuery as exc:
        raise api_error(
            status.HTTP_400_BAD_REQUEST, "invalid_request", str(exc)
        ) from exc


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
