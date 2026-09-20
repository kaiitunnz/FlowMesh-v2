"""The shared durable store the fabric's content actually lives in.

An S3-compatible object store — MinIO, cloud S3, anything speaking the same API — holds
every content object as the source of truth. A worker writes an object here before it
reports the reference naming it, so a reference that reaches any binding names bytes
that already survive the worker that produced them; a worker that needs an object and
does not have it reads from here. Workers hold copies to save that read, but a copy is
only ever a copy: this is where the object is.

The key is the reference's own identity — scope first, so a scope is a prefix an access
policy can be written against, then the algorithm and digest naming the bytes. Content
addressing makes a write idempotent: the same bytes under the same key are the same
object, so a re-drive rewrites rather than conflicts.
"""

from typing import Any, Protocol

from .reference import OCTET_STREAM, ContentReference
from .store import (
    ContentHydrationError,
    ContentStoreError,
    FabricObjectStore,
    reference_for,
)


class S3Client(Protocol):
    """The two calls a content object needs from an S3-compatible client."""

    def put_object(self, **kwargs: Any) -> Any: ...

    def get_object(self, **kwargs: Any) -> Any: ...


class S3ObjectStore(FabricObjectStore):
    """Content objects in one bucket of an S3-compatible store."""

    def __init__(self, client: S3Client, bucket: str, *, prefix: str = "") -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def _key(self, reference: ContentReference) -> str:
        scope, algorithm, digest = reference.identity
        key = f"{scope}/{algorithm}/{digest}"
        return f"{self._prefix}/{key}" if self._prefix else key

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = reference_for(scope, data, media_type=media_type)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._key(reference),
                Body=data,
                ContentType=media_type,
            )
        except Exception as exc:  # noqa: BLE001 - every backend failure is one to us
            raise ContentStoreError(f"content write failed: {exc}") from exc
        return reference

    def fetch(self, reference: ContentReference) -> bytes:
        try:
            response: dict[str, Any] = self._client.get_object(
                Bucket=self._bucket, Key=self._key(reference)
            )
        except Exception as exc:  # noqa: BLE001 - a miss and a refusal read the same
            raise ContentHydrationError(
                f"no content for {reference.content_digest} in scope "
                f"{reference.authorization_scope}: {exc}"
            ) from exc
        body = response["Body"]
        try:
            return bytes(body.read())
        finally:
            body.close()


__all__ = ["S3Client", "S3ObjectStore"]
