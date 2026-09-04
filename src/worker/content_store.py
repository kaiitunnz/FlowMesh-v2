"""Build the worker's content-store client from its resolved server endpoint."""

from shared.outcome import FabricContentStore
from shared.outcome.http_store import HttpFabricContentStore


def build_content_store(base_url: str | None) -> FabricContentStore | None:
    """The HTTP content store for a resolved server base URL, or None when absent.

    A worker with no reachable server endpoint materializes nothing and inlines its
    outcomes as the compatibility fallback.
    """
    if not base_url:
        return None
    return HttpFabricContentStore(base_url)
