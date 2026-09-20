"""Where the shared durable content store is, and how a process opens it.

Both the workers that read and write content and the control plane that cuts their
access describe the same external store, so they describe it the same way; each process
builds this at its own configuration edge.
"""

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_S3 = "s3"
BACKEND_FILESYSTEM = "filesystem"

# The port the store a default deployment co-locates publishes on its own host.
_DEFAULT_STORE_PORT = "9000"


def _colocated_endpoint() -> str:
    """The store beside the root, as the host it runs on reaches it."""
    port = os.getenv("CONTENT_STORE_PORT", "").strip() or _DEFAULT_STORE_PORT
    return f"http://127.0.0.1:{port}"


@dataclass(frozen=True)
class ObjectStoreConfig:
    """The shared store's backend and address.

    ``backend`` selects how the fleet's store is reached: ``s3`` for anything speaking
    the S3 API — the co-located MinIO a default deployment runs, cloud S3, an external
    MinIO — or ``filesystem`` for one durable filesystem mounted on every node.

    With nothing configured this describes the store a default deployment brings up
    beside the root, so a fresh deployment stores content without being told where. A
    deployment whose store is elsewhere — cloud object storage, an external MinIO, a
    node not the root's — names it, and every node other than the store's own host has
    to, since the co-located address is only local to that host.
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
            scoped_credentials=(
                os.getenv("CONTENT_STORE_SCOPED_CREDENTIALS", "true").strip().lower()
                not in {"0", "false", "no"}
            ),
        )


__all__ = ["BACKEND_FILESYSTEM", "BACKEND_S3", "ObjectStoreConfig"]
