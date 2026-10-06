"""A node's own peer TLS identity, and the root's capability to dial a peer transport.

Every server-side party that takes part in a peer transport reads the same operator
files: a node's supervisor to serve its peer listener and probe its peers, and the root
to dial a target as the origin of its own gated serve requests.
"""

import logging
import ssl
from dataclasses import dataclass

from shared.network.mtls import (
    MutualTlsMaterial,
    MutualTlsMaterialError,
    client_context,
)

from ..config import NetworkPlaneConfig, TrustedPeerConfig


def load_peer_material(
    peer: TrustedPeerConfig, logger: logging.Logger
) -> MutualTlsMaterial | None:
    """Read this node's peer TLS material from the operator's configured files.

    Returns ``None`` under the operator's attested no-mTLS posture, and raises
    ``MutualTlsMaterialError`` for material that is missing or unreadable rather than
    falling back to plaintext.
    """
    if peer.disable_mtls:
        logger.warning(
            "running peer connections without mutual TLS: the deployment is "
            "configured for a trusted network, so no peer proves an identity"
        )
        return None
    return MutualTlsMaterial.from_files(
        ca_file=peer.tls_ca_file,
        cert_file=peer.tls_cert_file,
        key_file=peer.tls_key_file,
    )


@dataclass(frozen=True)
class PeerDialer:
    """What an origin dials a peer transport with.

    ``ssl_context`` is ``None`` only under the operator's explicit no-mTLS posture.
    """

    ssl_context: ssl.SSLContext | None
    connect_budget_sec: float


def root_peer_dialer(
    network: NetworkPlaneConfig, logger: logging.Logger
) -> PeerDialer | None:
    """Return how the root dials a peer as its serve ingress's origin, or ``None``.

    The root dials with its own node's identity. Without the peer plane, or with
    material it cannot build a client context from, the root dials nothing and its
    serve requests ride ``control_relay``: an unusable identity never becomes a
    plaintext dial.
    """
    if not (network.enabled and network.peer.enabled):
        return None
    try:
        material = load_peer_material(network.peer, logger)
        context = client_context(material) if material is not None else None
    except (MutualTlsMaterialError, ssl.SSLError) as exc:
        logger.error(
            "root-originated gated serve rides control_relay: the node's peer TLS "
            "material is unusable (%s)",
            exc,
        )
        return None
    return PeerDialer(
        ssl_context=context, connect_budget_sec=network.connect_budget_sec
    )


__all__ = ["PeerDialer", "load_peer_material", "root_peer_dialer"]
