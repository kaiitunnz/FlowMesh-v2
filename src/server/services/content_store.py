"""A filesystem-backed content store for reference-backed outcomes.

The store holds worker-materialized outcome content as immutable, content-addressed
objects under a per-principal partition, plus an idempotency index so a re-drive under
the same fabric ``idempotency_key`` resolves the first materialization rather than
writing a second object. It stores opaque bytes and metadata only; it never assembles
content into orchestration state. Writes come from the worker over the content router;
the server never originates a materialization. Isolation is the partition: a read is
scoped to the requesting principal, so it never reaches another principal's content.
"""

import os
import tempfile
from pathlib import Path

from shared.outcome import (
    ContentStoreError,
    OutcomeHydrationError,
    OutcomeManifest,
    content_digest,
)

_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _segment(value: str | None) -> str:
    """A single path segment safe from traversal, or ``_`` for an empty principal."""
    if not value:
        return "_"
    if value in {".", ".."} or any(c not in _SAFE for c in value):
        raise ContentStoreError(f"unsafe content-store segment {value!r}")
    return value


class ServerContentStore:
    """A per-principal, content-addressed immutable store under a local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _object_path(self, principal: str | None, digest: str) -> Path:
        digest = _segment(digest)
        return (
            self._root / _segment(principal) / "objects" / _segment(digest[:2]) / digest
        )

    def _idem_path(self, principal: str | None, idempotency_key: str) -> Path:
        return (
            self._root
            / _segment(principal)
            / "idem"
            / f"{_segment(idempotency_key)}.json"
        )

    def find(
        self, principal: str | None, idempotency_key: str
    ) -> OutcomeManifest | None:
        path = self._idem_path(principal, idempotency_key)
        if not path.exists():
            return None
        return OutcomeManifest.model_validate_json(path.read_text())

    def materialize(
        self,
        principal: str | None,
        idempotency_key: str,
        data: bytes,
        *,
        media_type: str,
        provenance: str | None = None,
    ) -> OutcomeManifest:
        if (found := self.find(principal, idempotency_key)) is not None:
            return found
        digest = content_digest(data)
        self._atomic_write(self._object_path(principal, digest), data)
        manifest = OutcomeManifest(
            content_digest=digest,
            size_bytes=len(data),
            media_type=media_type,
            provenance=provenance,
            idempotency_key=idempotency_key,
            tenant=principal,
        )
        self._atomic_write(
            self._idem_path(principal, idempotency_key),
            manifest.model_dump_json().encode(),
        )
        return manifest

    def read(self, principal: str | None, digest: str) -> bytes:
        path = self._object_path(principal, digest)
        if not path.exists():
            raise OutcomeHydrationError(
                f"no content for {digest} under principal {principal}"
            )
        return path.read_bytes()

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
