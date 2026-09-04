"""The worker-side client for the server-hosted outcome content store.

A worker materializes an outcome by uploading its bytes to the content router and
hydrates one by fetching content-addressed bytes; the server is authoritative for the
tenant scope and manifest identity. This is the one concrete backend selected now;
worker-to-worker data-direct hydration can replace it behind the ``FabricContentStore``
seam without changing the outcome contract.
"""

import requests

from shared.utils.http import auth_headers

from .content_store import (
    ContentStoreError,
    FabricContentStore,
    OutcomeAccessBinding,
    OutcomeHydrationError,
    OutcomeSpool,
)
from .manifest import OutcomeManifest


class HttpFabricContentStore(FabricContentStore):
    """A ``FabricContentStore`` backed by the server content router over HTTP."""

    def __init__(
        self, base_url: str, *, token: str | None = None, timeout: float = 30.0
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base}/api/v1/content{path}"

    def find(self, tenant: str | None, idempotency_key: str) -> OutcomeManifest | None:
        resp = requests.get(
            self._url(f"/by-idem/{idempotency_key}"),
            headers=auth_headers(self._token),
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise ContentStoreError(f"content find failed: {resp.status_code}")
        return OutcomeManifest.model_validate_json(resp.content)

    def open_spool(self, tenant: str | None, idempotency_key: str) -> OutcomeSpool:
        return _HttpSpool(self, idempotency_key)

    def read(self, tenant: str | None, digest: str) -> bytes:
        resp = requests.get(
            self._url(f"/{digest}"),
            headers=auth_headers(self._token),
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            raise OutcomeHydrationError(f"no content for {digest}")
        if resp.status_code >= 400:
            raise ContentStoreError(f"content read failed: {resp.status_code}")
        return resp.content

    def materialize(
        self,
        tenant: str | None,
        idempotency_key: str,
        data: bytes,
        *,
        media_type: str,
        provenance: str | None = None,
        owner_subject: str | None = None,
    ) -> OutcomeManifest:
        if (found := self.find(tenant, idempotency_key)) is not None:
            return found
        return self._put(idempotency_key, data, media_type)

    def _put(
        self, idempotency_key: str, data: bytes, media_type: str
    ) -> OutcomeManifest:
        resp = requests.put(
            self._url(""),
            params={"idem": idempotency_key},
            data=data,
            headers={**auth_headers(self._token), "Content-Type": media_type},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise ContentStoreError(f"content materialize failed: {resp.status_code}")
        return OutcomeManifest.model_validate_json(resp.content)


class _HttpSpool(OutcomeSpool):
    def __init__(self, store: HttpFabricContentStore, idempotency_key: str) -> None:
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
        return self._store._put(self._idm, bytes(self._buf), media_type)
