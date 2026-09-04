"""Shared fabric external-tool contract and worker-side execution.

The control plane (root) authorizes and records external-tool operations; a worker
executor validates the operation fence and performs the provider egress. Both planes
import these schemas, the provider backends, the egress surface, and the frame codec
from here, so the execution pieces live outside ``src/server`` where a worker can run
them.
"""
