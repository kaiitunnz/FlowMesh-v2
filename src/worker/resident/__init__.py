"""Worker-owned resident inference protocol.

The origin worker drives a resident invocation (bootstrap, ack, authorized stream, chunk
assembly, materialize) and the replica worker serves its co-located engine behind the
claim gate. Both own the windowed relay session end to end; the servers relay its frames
opaquely and never read its cursor or window.
"""

from .capture import capture_resident_request
from .engine import EngineOpen, EngineResponse, HttpEngineDelivery
from .lane_host import ResidentLaneHost
from .origin_driver import ResidentOriginDriver, ResidentOriginRequest
from .replica_sidecar import ResidentReplicaSidecar
from .request_store import ResidentRequestStore
from .session import ResidentRelaySession, ResidentSessionRole
from .transport import ResidentFrameSink

__all__ = [
    "EngineOpen",
    "EngineResponse",
    "HttpEngineDelivery",
    "capture_resident_request",
    "ResidentFrameSink",
    "ResidentLaneHost",
    "ResidentOriginDriver",
    "ResidentOriginRequest",
    "ResidentRelaySession",
    "ResidentReplicaSidecar",
    "ResidentRequestStore",
    "ResidentSessionRole",
]
