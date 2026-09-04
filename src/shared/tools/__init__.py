"""Shared fabric external-tool contract and worker-side execution.

The generic operation contract (``contract``) and frame codec (``wire``) are
tool-agnostic; a per-tool package (today ``search``) supplies the request schema,
provider backends, and egress surface. These live outside ``src/server`` so a worker
executor can run the provider egress.
"""
