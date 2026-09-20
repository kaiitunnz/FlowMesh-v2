"""Outcome finalization over the shared content store and the control-plane index.

Two halves, each where it belongs. The content goes to the shared durable store, so a
completed outcome outlives the worker that produced it. The binding from the
finalization's fabric idempotency key to that content goes to the control plane, so a
re-drive can tell that the outcome already exists and re-report it rather than re-run a
sampled producer. Neither half holds the other's: the server never sees the bytes, and
the store never learns what an outcome is.
"""

import requests

from shared.content import (
    OCTET_STREAM,
    ContentReference,
    ContentStoreError,
    ScopedObjectStore,
)
from shared.telemetry.propagation import inject_ambient_traceparent
from shared.utils.http import auth_headers

from .content_store import FabricContentStore
from .manifest import OutcomeManifest


class FinalizationIndexClient:
    """The control plane's record of what each finalization materialized."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self) -> str:
        return f"{self._base}/api/v1/content/finalizations"

    @staticmethod
    def _headers() -> dict[str, str]:
        """The auth headers plus the ambient trace context's ``traceparent``.

        These calls run inside a task's span, so the ambient context is the right one to
        forward; with no active span the inject is a no-op.
        """
        return inject_ambient_traceparent(auth_headers())

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        resp = requests.get(
            self._url(),
            params={"scope": scope, "idem": idempotency_key},
            headers=self._headers(),
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise ContentStoreError(f"finalization lookup failed: {resp.status_code}")
        return OutcomeManifest.model_validate_json(resp.content)

    def record(
        self, scope: str, idempotency_key: str, content: ContentReference
    ) -> OutcomeManifest:
        resp = requests.put(
            self._url(),
            params={"scope": scope, "idem": idempotency_key},
            json=content.model_dump(mode="json"),
            headers=self._headers(),
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise ContentStoreError(f"finalization failed: {resp.status_code}")
        return OutcomeManifest.model_validate_json(resp.content)


class FinalizingContentStore(FabricContentStore):
    """Outcome finalization: content in the shared store, its binding in control."""

    def __init__(
        self, objects: ScopedObjectStore, index: FinalizationIndexClient
    ) -> None:
        self._objects = objects
        self._index = index

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._objects.for_scope(scope).write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        return self._objects.for_scope(reference.authorization_scope).fetch(reference)

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        return self._index.find(scope, idempotency_key)

    def materialize(
        self, scope: str, idempotency_key: str, data: bytes, *, media_type: str
    ) -> OutcomeManifest:
        """Write the content, then bind it — in that order, and never the reverse.

        A binding that named content the store did not have yet would be a reference no
        reader could resolve, so the content is durable before anything can learn of it.
        A re-drive that reaches a binding already recorded takes it and discards what it
        just wrote, which costs a write of identical bytes and settles as the first.
        """
        if (found := self._index.find(scope, idempotency_key)) is not None:
            return found
        content = self.write(scope, data, media_type=media_type)
        return self._index.record(scope, idempotency_key, content)
