"""A filesystem-backed content store for reference-backed outcomes.

The store holds worker-materialized outcome content as immutable, content-addressed
objects under a per-tenant partition, plus an idempotency index so a re-drive under the
same fabric ``idempotency_key`` resolves the first materialization rather than writing a
second object. It stores opaque bytes and metadata only; it never assembles content into
orchestration state. Writes come from the worker over the content router; the server
never originates a materialization.
"""

import os
import tempfile
from pathlib import Path, PurePosixPath

from shared.outcome import (
    ContentStoreError,
    FabricContentStore,
    OutcomeAccessBinding,
    OutcomeHydrationError,
    OutcomeManifest,
    OutcomeSpool,
    content_digest,
)

_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _segment(value: str | None) -> str:
    """A single path segment safe from traversal, or ``_`` for an empty tenant."""
    if not value:
        return "_"
    if value in {".", ".."} or any(c not in _SAFE for c in value):
        raise ContentStoreError(f"unsafe content-store segment {value!r}")
    return value


class ServerContentStore(FabricContentStore):
    """A per-tenant, content-addressed immutable store rooted at a local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _tenant_dir(self, tenant: str | None) -> Path:
        return self._root / _segment(tenant)

    def _object_path(self, tenant: str | None, digest: str) -> Path:
        return (
            self._tenant_dir(tenant)
            / "objects"
            / _segment(digest[:2])
            / _segment(digest)
        )

    def _idem_path(self, tenant: str | None, idempotency_key: str) -> Path:
        name = _segment(idempotency_key)
        return self._tenant_dir(tenant) / "idem" / f"{name}.json"

    def find(self, tenant: str | None, idempotency_key: str) -> OutcomeManifest | None:
        path = self._idem_path(tenant, idempotency_key)
        if not path.exists():
            return None
        return OutcomeManifest.model_validate_json(path.read_text())

    def open_spool(self, tenant: str | None, idempotency_key: str) -> OutcomeSpool:
        return _FileSpool(self, idempotency_key)

    def read(self, tenant: str | None, digest: str) -> bytes:
        path = self._object_path(tenant, _segment(digest))
        if not path.exists():
            raise OutcomeHydrationError(
                f"no content for {digest} under tenant {tenant}"
            )
        return path.read_bytes()

    def _commit(self, tenant: str | None, data: bytes) -> str:
        digest = content_digest(data)
        self._atomic_write(self._object_path(tenant, digest), data)
        return digest

    def _record_idem(
        self, tenant: str | None, idempotency_key: str, manifest: OutcomeManifest
    ) -> None:
        self._atomic_write(
            self._idem_path(tenant, idempotency_key),
            manifest.model_dump_json().encode(),
        )

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write ``data`` at ``path`` if absent, leaving an existing object untouched.

        Content is immutable, so a concurrent or re-driven write of the same content is
        a no-op rather than a rewrite.
        """
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class _FileSpool(OutcomeSpool):
    def __init__(self, store: ServerContentStore, idempotency_key: str) -> None:
        self._store = store
        self._idm = idempotency_key
        self._buf = bytearray()
        self._cursor = 0

    def append(self, data: bytes) -> int:
        self._buf.extend(data)
        self._cursor += len(data)
        return self._cursor

    def finalize(
        self,
        *,
        media_type: str,
        provenance: str | None,
        access: OutcomeAccessBinding,
    ) -> OutcomeManifest:
        data = bytes(self._buf)
        digest = self._store._commit(access.tenant, data)
        manifest = OutcomeManifest(
            content_digest=digest,
            size_bytes=len(data),
            media_type=media_type,
            provenance=provenance,
            idempotency_key=self._idm,
            access=access,
        )
        self._store._record_idem(access.tenant, self._idm, manifest)
        return manifest


def default_content_root(base: str) -> Path:
    """The content-store root under a server data directory, as a native path."""
    return Path(PurePosixPath(base) / "content")
