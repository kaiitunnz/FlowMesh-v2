"""Resident-inference protocol contracts shared by control, relay, and workers.

The claim-bound admission fences (`AdmissionHandoff`, `RouteAuthorization`), the engine
endpoint, the target-side claim gate, the framed request/response wire, and the engine
request builder. Central control mints the fences and owns the `ServiceClaim` FSM; the
origin and replica workers execute the protocol against these contracts; the servers
relay the frames opaquely.
"""

from .contracts import AdmissionHandoff, ReplicaEndpoint, RouteAuthorization
from .engine_request import chat_body
from .gate import (
    GateDecision,
    GateRejection,
    LoadEvidence,
    SidecarClaimGate,
    SidecarSession,
    TrafficClass,
)
from .wire import (
    KIND_ACK,
    KIND_BOOTSTRAP,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_REJECT,
    KIND_STREAM,
    decode_msg,
    encode_msg,
)

__all__ = [
    "KIND_ACK",
    "KIND_BOOTSTRAP",
    "KIND_CHUNK",
    "KIND_DONE",
    "KIND_FAILED",
    "KIND_REJECT",
    "KIND_STREAM",
    "AdmissionHandoff",
    "GateDecision",
    "GateRejection",
    "LoadEvidence",
    "ReplicaEndpoint",
    "RouteAuthorization",
    "SidecarClaimGate",
    "SidecarSession",
    "TrafficClass",
    "chat_body",
    "decode_msg",
    "encode_msg",
]
