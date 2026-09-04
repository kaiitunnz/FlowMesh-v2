"""Server import path for the shared web-search backends.

The provider backends live in ``shared.tools.providers`` so a worker executor can run
them; the transitional in-server relay path and the control plane reach them here.
"""

from shared.tools.providers import (
    DuckDuckGoProvider,
    LazySearchProvider,
    ProviderConfig,
    SearchProvider,
    SearchQuotaExceeded,
    SearchResult,
    SearchTimeout,
    SearchUnavailable,
    SerperProvider,
    build_search_provider,
)

__all__ = [
    "DuckDuckGoProvider",
    "LazySearchProvider",
    "ProviderConfig",
    "SearchProvider",
    "SearchQuotaExceeded",
    "SearchResult",
    "SearchTimeout",
    "SearchUnavailable",
    "SerperProvider",
    "build_search_provider",
]
