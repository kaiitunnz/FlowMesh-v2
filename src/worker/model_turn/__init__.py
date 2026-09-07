"""The worker-local Responses facade and the held model turn it runs.

The facade rebinds Codex's provider to a loopback surface, translates each turn between
the Responses wire and Chat Completions, captures the agent's fabric-facade calls, and
runs the held egress through the ``egress`` lane: it arms a rendezvous, proposes the
request digest, and egresses synchronously under the returned one-use permit.
"""

from .facade import FacadeTurnError, ResponsesFacade
from .held_egress import HeldModelEgress
from .rendezvous import ModelTurnRendezvous, PermitDenied

__all__ = [
    "FacadeTurnError",
    "HeldModelEgress",
    "ModelTurnRendezvous",
    "PermitDenied",
    "ResponsesFacade",
]
