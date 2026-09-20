"""Build the worker's outcome content store from its resolved configuration."""

import logging

from shared.outcome import (
    FabricContentStore,
    FinalizationIndexClient,
    FinalizingContentStore,
)

from .config import WorkerConfig
from .content import build_shared_store


def build_content_store(
    cfg: WorkerConfig, logger: logging.Logger
) -> FabricContentStore | None:
    """The store outcomes materialize into, or None when this worker reaches none.

    It needs both halves — the shared store for the content and the server for the
    finalization binding — so a worker missing either materializes nothing and inlines
    its outcomes as the compatibility fallback.
    """
    shared = build_shared_store(cfg.object_store, logger)
    if shared is None or not cfg.server_base_url:
        return None
    return FinalizingContentStore(shared, FinalizationIndexClient(cfg.server_base_url))
