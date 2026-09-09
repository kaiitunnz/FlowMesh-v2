"""Assemble the root forward serve listener from the server configuration.

Builds the ``RootForwardIngress`` bound to the deployment's port-forward interface and
public host and attaches it to the gated serve edge, or returns ``None`` when forward
serve is disabled so every forward-pinned request fails closed.
"""

import logging

from ..config import PortForwardConfig
from .forward_listener import RootForwardIngress
from .service import GatedServe


def build_forward_serve_ingress(
    config: PortForwardConfig,
    gated_serve: GatedServe,
    logger: logging.Logger,
) -> RootForwardIngress | None:
    """Build the root forward serve listener and attach it to the gated edge.

    Returns ``None`` when forward serve is disabled: the caller wires no listener and
    every forward-pinned request fails closed. The listener binds on the same interface
    and advertises the same public host SSH port forwarding uses.
    """
    if not config.serve_forward_enabled:
        return None
    listener = RootForwardIngress(
        bind_host=config.bind_host,
        public_host=config.public_host,
        admit=gated_serve.admit_forward_request,
        on_bound=gated_serve.commit_forward,
        body_budget_bytes=config.serve_forward_body_budget_bytes,
        logger=logger,
    )
    gated_serve.set_forward_listener(listener)
    return listener
