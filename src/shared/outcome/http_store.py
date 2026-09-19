"""The worker-side client for the server-hosted outcome content store.

A worker materializes an outcome by uploading its bytes to the content router and
hydrates one by fetching content-addressed bytes; the server authenticates the worker,
partitions content by its principal, and is authoritative for the manifest identity.
"""

import requests

from shared.content import ContentStoreError, ObjectWriteAck
from shared.telemetry.propagation import inject_ambient_traceparent
from shared.utils.http import auth_headers

from .content_store import FabricContentStore, OutcomeHydrationError
from .manifest import OutcomeManifest


def _headers() -> dict[str, str]:
    """The auth headers plus the ambient trace context's ``traceparent``.

    The worker's content-store calls run inside a task's span, so the ambient context
    is the right one to forward; with no active span the inject is a no-op.
    """
    return inject_ambient_traceparent(auth_headers())


class HttpFabricContentStore(FabricContentStore):
    """A ``FabricContentStore`` backed by the server content router over HTTP."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base}/api/v1/content{path}"

    def put_object(self, data: bytes) -> str:
        resp = requests.put(
            self._url("/objects"),
            data=data,
            headers={**_headers(), "Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise ContentStoreError(f"object write failed: {resp.status_code}")
        return ObjectWriteAck.model_validate_json(resp.content).content_digest

    def find(self, idempotency_key: str) -> OutcomeManifest | None:
        resp = requests.get(
            self._url(""),
            params={"idem": idempotency_key},
            headers=_headers(),
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise ContentStoreError(f"content find failed: {resp.status_code}")
        return OutcomeManifest.model_validate_json(resp.content)

    def materialize(
        self, idempotency_key: str, data: bytes, *, media_type: str
    ) -> OutcomeManifest:
        if (found := self.find(idempotency_key)) is not None:
            return found
        resp = requests.put(
            self._url(""),
            params={"idem": idempotency_key},
            data=data,
            headers={**_headers(), "Content-Type": media_type},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise ContentStoreError(f"content materialize failed: {resp.status_code}")
        return OutcomeManifest.model_validate_json(resp.content)

    def read(self, digest: str) -> bytes:
        resp = requests.get(
            self._url(f"/{digest}"), headers=_headers(), timeout=self._timeout
        )
        if resp.status_code == 404:
            raise OutcomeHydrationError(f"no content for {digest}")
        if resp.status_code >= 400:
            raise ContentStoreError(f"content read failed: {resp.status_code}")
        return resp.content
