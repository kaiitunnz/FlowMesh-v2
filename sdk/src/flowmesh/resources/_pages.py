"""Collect every entry of a cursor-paged list endpoint."""

from typing import TYPE_CHECKING, Any

from ..exceptions import FlowMeshError
from ..models.common import CursorPage

if TYPE_CHECKING:
    from .._base_client import BaseAsyncClient, BaseClient

PAGE_PARAMS = frozenset({"limit", "before", "after"})
PAGE_LIMIT = 1000


def _check_filters(params: list[tuple[str, str]]) -> None:
    if paging := sorted({key for key, _ in params} & PAGE_PARAMS):
        raise FlowMeshError(
            f"list() takes filters only and pages by itself; remove {', '.join(paging)}"
        )


def _page_params(
    params: list[tuple[str, str]], before: str | None
) -> list[tuple[str, str]]:
    page = [*params, ("limit", str(PAGE_LIMIT))]
    return [*page, ("before", before)] if before else page


def _read_page[P: CursorPage](page_model: type[P], path: str, data: Any) -> P:
    if isinstance(data, list):
        raise FlowMeshError(
            f"the server does not page {path}; upgrade it to list with this SDK"
        )
    return page_model.model_validate(data)


def _older(page: CursorPage, before: str | None) -> str | None:
    """The cursor of the next older page, or None at the oldest."""
    if len(page.entries) < PAGE_LIMIT or not page.prev_cursor:
        return None
    if page.prev_cursor == before:
        raise FlowMeshError("the server returned the same page twice")
    return page.prev_cursor


def list_all[P: CursorPage](
    client: "BaseClient",
    path: str,
    params: list[tuple[str, str]],
    page_model: type[P],
) -> list[P]:
    """Return every page the endpoint lists, oldest first, walking back from the
    newest page until a short one."""
    _check_filters(params)
    pages: list[P] = []
    before: str | None = None
    while True:
        data = client._request("GET", path, params=_page_params(params, before))
        pages.append(page := _read_page(page_model, path, data))
        if (before := _older(page, before)) is None:
            return pages[::-1]


async def list_all_async[P: CursorPage](
    client: "BaseAsyncClient",
    path: str,
    params: list[tuple[str, str]],
    page_model: type[P],
) -> list[P]:
    """Return every page the endpoint lists, oldest first, walking back from the
    newest page until a short one."""
    _check_filters(params)
    pages: list[P] = []
    before: str | None = None
    while True:
        data = await client._request("GET", path, params=_page_params(params, before))
        pages.append(page := _read_page(page_model, path, data))
        if (before := _older(page, before)) is None:
            return pages[::-1]
