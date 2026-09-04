"""The worker-side client for the server-hosted outcome content store.

A worker materializes an outcome by uploading its bytes to the content router and
hydrates one by fetching content-addressed bytes; the server authenticates the worker,
partitions content by its principal, and is authoritative for the manifest identity.
"""

import requests

from shared.utils.http import auth_headers

from .content_store import ContentStoreError, FabricContentStore, OutcomeHydrationError
from .manifest import OutcomeManifest


class HttpFabricContentStore(FabricContentStore):
    """A ``FabricContentStore`` backed by the server content router over HTTP."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base}/api/v1/content{path}"

    def find(self, idempotency_key: str) -> OutcomeManifest | None:
        resp = requests.get(
            self._url(f"/by-idem/{idempotency_key}"),
            headers=auth_headers(),
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
            headers={**auth_headers(), "Content-Type": media_type},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise ContentStoreError(f"content materialize failed: {resp.status_code}")
        return OutcomeManifest.model_validate_json(resp.content)

    def read(self, digest: str) -> bytes:
        resp = requests.get(
            self._url(f"/{digest}"), headers=auth_headers(), timeout=self._timeout
        )
        if resp.status_code == 404:
            raise OutcomeHydrationError(f"no content for {digest}")
        if resp.status_code >= 400:
            raise ContentStoreError(f"content read failed: {resp.status_code}")
        return resp.content
