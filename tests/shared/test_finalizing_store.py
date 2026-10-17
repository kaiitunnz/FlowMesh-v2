"""Outcome finalization split across the shared store and the control-plane index."""

import threading
from typing import Any

import pytest
import requests

from shared.content import (
    OCTET_STREAM,
    ContentHydrationError,
    ContentReference,
    ContentStoreError,
    FabricObjectStore,
    StoreRefusal,
    WriteCancelled,
    reference_for,
)
from shared.outcome import FinalizingContentStore, OutcomeManifest
from shared.outcome.finalizing_store import (
    FinalizationIndexClient,
    FinalizationRefusedError,
    FinalizationUnavailableError,
)


class _SharedStore(FabricObjectStore):
    """The durable store, holding bytes for every scope."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.writes = 0

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = reference_for(scope, data, media_type=media_type)
        self.objects[(scope, reference.content_digest)] = data
        self.writes += 1
        return reference

    def fetch(self, reference: ContentReference) -> bytes:
        key = (reference.authorization_scope, reference.content_digest)
        if key not in self.objects:
            raise ContentHydrationError(f"no content for {reference.content_digest}")
        return self.objects[key]

    def for_scope(self, scope: str) -> FabricObjectStore:
        return self


class _Index:
    """The control plane's finalization bindings."""

    def __init__(self) -> None:
        self.bindings: dict[tuple[str, str], OutcomeManifest] = {}

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        return self.bindings.get((scope, idempotency_key))

    def record(
        self, scope: str, idempotency_key: str, content: ContentReference
    ) -> OutcomeManifest:
        key = (scope, idempotency_key)
        if key not in self.bindings:
            self.bindings[key] = OutcomeManifest(
                content=content, idempotency_key=idempotency_key
            )
        return self.bindings[key]


def _store() -> tuple[FinalizingContentStore, _SharedStore, _Index]:
    shared, index = _SharedStore(), _Index()
    return FinalizingContentStore(shared, index), shared, index  # type: ignore[arg-type]


def test_a_materialized_outcome_is_in_the_shared_store_before_it_is_bound() -> None:
    store, shared, index = _store()
    manifest = store.materialize(
        "local", "idm-1", b"result", media_type="application/json"
    )
    assert shared.objects[("local", manifest.content.content_digest)] == b"result"
    assert index.find("local", "idm-1") == manifest


def test_a_redrive_settles_as_the_first_materialization() -> None:
    store, shared, _ = _store()
    first = store.materialize("local", "idm-2", b"sampled", media_type="text/plain")
    second = store.materialize("local", "idm-2", b"resampled", media_type="text/plain")
    assert first == second
    assert shared.writes == 1  # the second drive never reached the store


def test_an_outcome_hydrates_from_the_shared_store() -> None:
    store, _, _ = _store()
    manifest = store.materialize(
        "local", "idm-3", b"result", media_type="application/json"
    )
    assert store.hydrate(manifest.content) == b"result"


def test_a_scope_never_resolves_another_scopes_key() -> None:
    store, _, _ = _store()
    store.materialize("local", "idm-4", b"result", media_type="application/json")
    assert store.find("other", "idm-4") is None


def test_content_the_shared_store_does_not_have_fails_to_hydrate() -> None:
    store, _, _ = _store()
    with pytest.raises(ContentStoreError):
        store.hydrate(reference_for("local", b"never written"))


class _RefusingIndex(_Index):
    """An index that refuses each recording with the dispositions given, in order."""

    def __init__(self, *refusals: str) -> None:
        super().__init__()
        self.refusals = list(refusals)
        self.records = 0

    def record(
        self, scope: str, idempotency_key: str, content: ContentReference
    ) -> OutcomeManifest:
        self.records += 1
        if self.refusals:
            raise FinalizationRefusedError(StoreRefusal(self.refusals.pop(0)))
        return super().record(scope, idempotency_key, content)


class _Pauses:
    """Records each pause and never waits; reports a stop once ``stop_after`` pauses
    have been taken."""

    def __init__(self, stop_after: int | None = None) -> None:
        self.delays: list[float] = []
        self.stop_after = stop_after

    def __call__(self, stop: threading.Event, delay: float) -> bool:
        self.delays.append(delay)
        if self.stop_after is not None and len(self.delays) >= self.stop_after:
            stop.set()
        return stop.is_set()


def test_a_refused_recording_records_the_same_bytes_again() -> None:
    shared = _SharedStore()
    index = _RefusingIndex("retry_retirement", "rematerialize", "retry")
    store = FinalizingContentStore(
        shared, index, pause=_Pauses()  # type: ignore[arg-type]
    )

    manifest = store.materialize("local", "idm-9", b"once", media_type="text/plain")

    assert manifest.content == reference_for("local", b"once", media_type="text/plain")
    assert index.records == 4
    # Written once, and again only for the refusal that found its copy gone.
    assert shared.writes == 2


class _OutIndex(_Index):
    """An index control cannot answer for the first ``outages`` calls of each kind."""

    def __init__(self, finds: int = 0, records: int = 0) -> None:
        super().__init__()
        self.find_outages = finds
        self.record_outages = records
        self.finds = 0
        self.records = 0

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        self.finds += 1
        if self.find_outages:
            self.find_outages -= 1
            raise FinalizationUnavailableError("control restarting")
        return super().find(scope, idempotency_key)

    def record(
        self, scope: str, idempotency_key: str, content: ContentReference
    ) -> OutcomeManifest:
        self.records += 1
        if self.record_outages:
            self.record_outages -= 1
            raise FinalizationUnavailableError("control restarting")
        return super().record(scope, idempotency_key, content)


def test_an_unavailable_control_is_waited_out_in_place() -> None:
    shared = _SharedStore()
    index = _OutIndex(finds=2, records=7)
    pauses = _Pauses()
    store = FinalizingContentStore(
        shared, index, pause=pauses  # type: ignore[arg-type]
    )

    manifest = store.materialize("local", "idm-5", b"once", media_type="text/plain")

    assert manifest.content == reference_for("local", b"once", media_type="text/plain")
    assert (index.finds, index.records) == (3, 8)
    assert shared.writes == 1
    assert pauses.delays == [0.5, 1.0, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 15.0]


def test_a_stop_ends_a_materialization_waiting_on_control() -> None:
    shared = _SharedStore()
    index = _OutIndex(records=1_000)
    pauses = _Pauses(stop_after=3)
    store = FinalizingContentStore(
        shared, index, pause=pauses  # type: ignore[arg-type]
    )

    with pytest.raises(WriteCancelled):
        store.materialize("local", "idm-6", b"once", media_type="text/plain")

    assert index.records == 3
    assert index.bindings == {}


def test_a_stopped_store_does_not_wait() -> None:
    stop = threading.Event()
    stop.set()
    pauses = _Pauses()
    store = FinalizingContentStore(
        _SharedStore(), _OutIndex(finds=1), stop=stop, pause=pauses  # type: ignore[arg-type]
    )

    with pytest.raises(WriteCancelled):
        store.materialize("local", "idm-7", b"once", media_type="text/plain")

    assert pauses.delays == []


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.content = b"{}"

    def json(self) -> Any:
        return {}


@pytest.mark.parametrize(
    "outcome",
    [
        _Response(503),
        requests.ConnectionError("refused"),
        requests.Timeout("timed out"),
    ],
    ids=["503", "transport", "timeout"],
)
@pytest.mark.parametrize("method", ["get", "put"])
def test_the_index_client_reports_an_unreachable_control_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, outcome: Any, method: str
) -> None:
    def answer(*args: Any, **kwargs: Any) -> Any:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(f"shared.outcome.finalizing_store.requests.{method}", answer)
    client = FinalizationIndexClient("http://control")

    with pytest.raises(FinalizationUnavailableError):
        if method == "get":
            client.find("local", "idm-8")
        else:
            client.record("local", "idm-8", reference_for("local", b"x"))
