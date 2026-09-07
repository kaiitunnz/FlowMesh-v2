"""Controlled external inference ingress.

An authenticated external principal reuses a published, tenant-authorized resident
service-family alias through the same admission gate and worker-executed resident path
as a workflow consumer. The ingress is an authentication and control edge: it
authenticates
and quota-limits the principal, resolves only a published alias, records a durable
invocation subject, and asks normal admission to raise a claim. It selects no replica,
owns no credit, and never touches the engine — the designated worker constructs the
engine request, uses engine credentials, parses the response, and materializes any
result reference.
"""

from .aliases import AliasCatalog, PublishedAlias
from .quota import PrincipalQuota, QuotaExceeded
from .service import InferenceIngress, IngressResult
from .state import IngressTerminal, IngressTerminalStore

__all__ = [
    "AliasCatalog",
    "InferenceIngress",
    "IngressResult",
    "IngressTerminal",
    "IngressTerminalStore",
    "PrincipalQuota",
    "PublishedAlias",
    "QuotaExceeded",
]
