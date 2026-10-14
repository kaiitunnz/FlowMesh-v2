"""Where the shared durable content store is, and how a process opens it.

Both the workers that read and write content and the control plane that cuts their
access describe the same external store, so they describe it the same way; each process
builds this at its own configuration edge, as it does whether workers cache and hydrate
content.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from shared.utils.parsing import parse_bool_env, parse_int_env, to_bool

BACKEND_S3 = "s3"
BACKEND_FILESYSTEM = "filesystem"

# The port the store a default deployment co-locates publishes on its own host.
_DEFAULT_STORE_PORT = "9800"
# S3 refuses a part of a multipart upload smaller than this, the last one aside, and
# any part larger than the maximum.
MIN_PART_BYTES = 5 << 20
MAX_PART_BYTES = 5 << 30
DEFAULT_PART_BYTES = 16 << 20
DEFAULT_UPLOAD_CONCURRENCY = 4
# How much of a content transfer may be in flight, and the object size from which a
# worker reads the shared store before the root relay.
DEFAULT_TRANSFER_WINDOW_BYTES = 1 << 20
DEFAULT_RELAY_TRANSFER_MAX_BYTES = 64 << 20


def _colocated_endpoint() -> str:
    """The store beside the root, as the host it runs on reaches it."""
    port = os.getenv("CONTENT_STORE_PORT", "").strip() or _DEFAULT_STORE_PORT
    return f"http://127.0.0.1:{port}"


@dataclass(frozen=True)
class ObjectStoreConfig:
    """The shared store's backend and address.

    ``backend`` selects how the fleet's store is reached: ``s3`` for anything speaking
    the S3 API — the co-located Silo store a default deployment runs, cloud S3, another
    S3-compatible service — or ``filesystem`` for one durable filesystem mounted on
    every node.

    With nothing configured this describes the store a default deployment brings up
    beside the root, so a fresh deployment stores content without being told where. A
    deployment whose store is elsewhere — cloud object storage, an external
    S3-compatible store, a node not the root's — names it, and every node other than the
    store's own host has to, since the co-located address is only local to that host.

    ``part_bytes`` is the size of one part of a large object's upload and
    ``upload_concurrency`` how many parts one upload sends at once, so one write holds
    at most their product in memory.
    """

    backend: str = BACKEND_S3
    endpoint_url: str = ""
    bucket: str = "flowmesh-content"
    prefix: str = ""
    region: str = "us-east-1"
    access_key: str = ""
    secret_key: str = ""
    filesystem_root: Path = Path("./shared-content")
    scoped_credentials: bool = True
    part_bytes: int = DEFAULT_PART_BYTES
    upload_concurrency: int = DEFAULT_UPLOAD_CONCURRENCY

    @staticmethod
    def from_env(default_root: Path) -> "ObjectStoreConfig":
        return ObjectStoreConfig(
            backend=os.getenv("CONTENT_STORE_BACKEND", BACKEND_S3).strip()
            or BACKEND_S3,
            endpoint_url=(
                os.getenv("CONTENT_STORE_ENDPOINT_URL", "").strip()
                or _colocated_endpoint()
            ),
            # A variable relayed to a worker arrives set-but-empty when the node that
            # relayed it had none, so an empty value means "unset", not "no bucket".
            bucket=os.getenv("CONTENT_STORE_BUCKET", "").strip() or "flowmesh-content",
            prefix=os.getenv("CONTENT_STORE_PREFIX", "").strip(),
            region=os.getenv("CONTENT_STORE_REGION", "").strip() or "us-east-1",
            access_key=os.getenv("CONTENT_STORE_ACCESS_KEY", "").strip(),
            secret_key=os.getenv("CONTENT_STORE_SECRET_KEY", "").strip(),
            filesystem_root=Path(
                os.getenv("CONTENT_STORE_FILESYSTEM_ROOT", "").strip()
                or (default_root / "shared-content")
            ).absolute(),
            scoped_credentials=parse_bool_env("CONTENT_STORE_SCOPED_CREDENTIALS", True),
            part_bytes=min(
                MAX_PART_BYTES,
                max(
                    MIN_PART_BYTES,
                    parse_int_env("CONTENT_STORE_PART_BYTES", DEFAULT_PART_BYTES),
                ),
            ),
            upload_concurrency=max(
                1,
                parse_int_env(
                    "CONTENT_STORE_UPLOAD_CONCURRENCY", DEFAULT_UPLOAD_CONCURRENCY
                ),
            ),
        )


def content_hydration_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether workers keep the content cache and hydrate content between them:
    as ``CONTENT_HYDRATION_ENABLED`` says, and when it is unset, whenever the network
    plane that carries the transfers is on.

    Raises ``ValueError`` when it is turned on with the network plane off.
    """
    env = os.environ if environ is None else environ
    plane = to_bool(env.get("NETWORK_PLANE_ENABLED"), default=True)
    explicit = to_bool(env.get("CONTENT_HYDRATION_ENABLED"))
    if explicit and not plane:
        raise ValueError(
            "CONTENT_HYDRATION_ENABLED requires NETWORK_PLANE_ENABLED: a content "
            "transfer is carried by the network plane's relay"
        )
    return plane if explicit is None else explicit


__all__ = [
    "BACKEND_FILESYSTEM",
    "BACKEND_S3",
    "ObjectStoreConfig",
    "content_hydration_enabled",
]
