"""Collect every entry of a cursor-paged list endpoint."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._base_client import BaseAsyncClient, BaseClient

PAGE_PARAMS = frozenset({"limit", "before", "after"})
PAGE_LIMIT = 1000


def _filter_params(params: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if paging := sorted({key for key, _ in params} & PAGE_PARAMS):
        raise ValueError(f"list() pages by itself; remove {', '.join(paging)}")
    return params


def _page_params(
    params: list[tuple[str, str]], before: str | None
) -> list[tuple[str, str]]:
    page = [*params, ("limit", str(PAGE_LIMIT))]
    return [*page, ("before", before)] if before else page


def list_all(
    client: "BaseClient", path: str, params: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Return every entry the endpoint lists, oldest first, walking back from the
    newest page until an empty one."""
    params = _filter_params(params)
    pages: list[list[dict[str, Any]]] = []
    before: str | None = None
    while True:
        data = client._request("GET", path, params=_page_params(params, before))
        if not data["entries"]:
            break
        pages.append(data["entries"])
        if not (before := data.get("prev_cursor")):
            break
    return [entry for page in reversed(pages) for entry in page]


async def list_all_async(
    client: "BaseAsyncClient", path: str, params: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Return every entry the endpoint lists, oldest first, walking back from the
    newest page until an empty one."""
    params = _filter_params(params)
    pages: list[list[dict[str, Any]]] = []
    before: str | None = None
    while True:
        data = await client._request("GET", path, params=_page_params(params, before))
        if not data["entries"]:
            break
        pages.append(data["entries"])
        if not (before := data.get("prev_cursor")):
            break
    return [entry for page in reversed(pages) for entry in page]
