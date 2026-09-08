"""The gated task-ID resident serve surface.

Every public user-declared serve task is a resident-gated standing allocation addressed
only by its task ID. This package holds the durable serve-task residency binding, the
durable external status-terminal facts, the transport-only edge-to-sidecar relay
executor, and the gated edge that authenticates, admits, and streams one request.
"""

from .binding import (
    ServeBindingSnapshot,
    ServeBindingStore,
    ServeSnapshot,
    ServeTaskResidencyBinding,
    serve_family_key,
)
from .relay import SERVE_EDGE_STREAM_ID, ServeRelayExecutor
from .service import (
    BindingNotFound,
    GatedServe,
    MethodNotAllowed,
    PathNotAllowed,
    ServeResult,
)
from .state import (
    ServeStatusTerminal,
    ServeTerminalSnapshot,
    ServeTerminalStatus,
    ServeTerminalStore,
)

__all__ = [
    "SERVE_EDGE_STREAM_ID",
    "BindingNotFound",
    "GatedServe",
    "MethodNotAllowed",
    "PathNotAllowed",
    "ServeBindingSnapshot",
    "ServeBindingStore",
    "ServeRelayExecutor",
    "ServeResult",
    "ServeSnapshot",
    "ServeStatusTerminal",
    "ServeTaskResidencyBinding",
    "ServeTerminalSnapshot",
    "ServeTerminalStore",
    "ServeTerminalStatus",
    "serve_family_key",
]
