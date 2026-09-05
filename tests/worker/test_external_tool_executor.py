"""Unit coverage for the worker-hosted external-tool executor.

The executor validates the operation fence and one-use delivery nonce, runs the provider
egress in this worker process, and returns one opaque reply frame — or none at all when
a crash leaves the boundary ambiguous. A fence failure returns a reject frame with no
provider call, and a delivery for a stale worker incarnation is rejected before egress.
"""

import threading
import time
from typing import Any

import pytest

from shared.outcome import content_digest
from shared.tools.contract import RemoteToolOperationEnvelope, ToolOutcome
from shared.tools.search.providers import SearchResult
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    ToolRequest,
    tool_request_digest,
)
from shared.tools.wire import (
    FRAME_CANCEL,
    FRAME_OPERATION,
    FRAME_REAP,
    FRAME_REPLY,
    KIND_MANIFEST,
    KIND_OPERATION,
    KIND_REJECT,
    KIND_RESULT,
    decode_msg,
    encode_msg,
)
from worker.external_tool_executor import WorkerExternalToolExecutor

from ..shared.outcome_helpers import InMemoryContentStore

WORKER = "wrk-1"
GEN = 7


class _Sink:
    def __init__(self) -> None:
        self.frames: list[tuple[str, str, bytes]] = []
        self._ev = threading.Event()

    def __call__(self, session_id: str, kind: str, frame: bytes) -> None:
        self.frames.append((session_id, kind, frame))
        self._ev.set()

    def wait(self, timeout: float = 2.0) -> bool:
        got = self._ev.wait(timeout)
        self._ev.clear()
        return got


def _fence(**over: Any) -> RemoteToolOperationEnvelope:
    base: dict[str, Any] = dict(
        interface=SEARCH_INTERFACE,
        provider="fake",
        idempotency_key="idm-1",
        request_digest=tool_request_digest(SEARCH_INTERFACE, "q", 3),
        target_id=WORKER,
        target_generation=GEN,
        delivery_nonce="xdn-1",
        policy_class="default",
        deadline_epoch=time.time() + 30,
        max_results=3,
        timeout_sec=5.0,
        result_char_cap=6000,
    )
    base.update(over)
    return RemoteToolOperationEnvelope(**base)


def _op(
    envelope: RemoteToolOperationEnvelope,
    query: str = "q",
    req_interface: str = SEARCH_INTERFACE,
) -> bytes:
    return encode_msg(
        KIND_OPERATION,
        envelope=envelope.model_dump(mode="json"),
        request=ToolRequest(
            interface=req_interface, query=query, max_results=3
        ).model_dump(mode="json"),
    )


def _executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation: int = GEN,
    raises: Exception | None = None,
    block: threading.Event | None = None,
    api_key: str | None = None,
    max_workers: int = 4,
    content_store: InMemoryContentStore | None = None,
) -> tuple[WorkerExternalToolExecutor, _Sink, list[str]]:
    calls: list[str] = []

    class _FakeProvider:
        def search(
            self, query: str, *, max_results: int, timeout_sec: float
        ) -> list[SearchResult]:
            calls.append(query)
            if block is not None:
                block.wait(2.0)
            if raises is not None:
                raise raises
            return [SearchResult(title="T", url="https://x", snippet="S")]

    monkeypatch.setattr(
        "shared.tools.search.providers.build_search_provider",
        lambda cfg: _FakeProvider(),
    )
    sink = _Sink()
    ex = WorkerExternalToolExecutor(
        worker_id=WORKER,
        generation=generation,
        provider="fake",
        api_key=api_key,
        result_sink=sink,
        content_store=content_store,
        max_workers=max_workers,
    )
    return ex, sink, calls


def test_success_without_a_store_is_unavailable_not_inlined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The provider egresses in this worker, but with no content store the result cannot
    # be referenced; it returns a bounded unavailable datum rather than an unbounded
    # inline body.
    ex, sink, calls = _executor(monkeypatch)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    assert sink.wait()
    session_id, kind, frame = sink.frames[0]
    assert (session_id, kind) == ("xtr-1", FRAME_REPLY)
    body = decode_msg(frame)
    assert body["kind"] == KIND_RESULT and body["outcome"]["status"] == "unavailable"
    assert calls == ["q"]


def test_success_materializes_by_reference_with_a_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryContentStore()
    ex, sink, calls = _executor(monkeypatch, content_store=store)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    assert sink.wait()
    _session, kind, frame = sink.frames[0]
    body = decode_msg(frame)
    # The provider result leaves the worker as a manifest, never an inline result body.
    assert kind == FRAME_REPLY and body["kind"] == KIND_MANIFEST
    manifest = body["manifest"]
    hydrated = store.read(manifest["content_digest"])
    assert manifest["content_digest"] == content_digest(hydrated)
    assert ToolOutcome.model_validate_json(hydrated).status.value == "success"
    assert calls == ["q"]


def test_success_reference_is_idempotent_under_idem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryContentStore()
    ex, sink, calls = _executor(monkeypatch, content_store=store)
    op = _op(_fence())
    ex.submit("xtr-1", FRAME_OPERATION, op)
    assert sink.wait()
    ex.submit("xtr-2", FRAME_OPERATION, _op(_fence(delivery_nonce="xdn-2")))
    assert sink.wait()
    # A same-idm re-drive finds the first materialization; the store holds one object,
    # and the provider is not sampled twice.
    assert store.write_count == 1
    assert calls == ["q"]


def test_control_status_stays_inline_with_a_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_cfg: Any) -> None:
        raise ValueError("the serper web-search provider needs WEB_SEARCH_API_KEY")

    monkeypatch.setattr("shared.tools.search.providers.build_search_provider", _raise)
    store = InMemoryContentStore()
    sink = _Sink()
    ex = WorkerExternalToolExecutor(
        worker_id=WORKER,
        generation=GEN,
        provider="fake",
        api_key=None,
        result_sink=sink,
        content_store=store,
    )
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    assert sink.wait()
    body = decode_msg(sink.frames[0][2])
    # A typed control status is a bounded inline datum, never materialized by reference.
    assert body["kind"] == KIND_RESULT
    assert body["outcome"]["status"] == "unavailable"
    assert store.write_count == 0


def test_store_failure_leaves_it_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryContentStore()
    store.fail_finalize = True
    ex, sink, _ = _executor(monkeypatch, content_store=store)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    # A store-write failure raises out of egress: no reply frame, so the origin holds
    # the boundary pending and re-drives under the same idempotency key.
    assert sink.wait(0.5) is False


def test_replayed_nonce_is_rejected_before_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex, sink, calls = _executor(monkeypatch)
    op = _op(_fence())
    ex.submit("xtr-1", FRAME_OPERATION, op)
    assert sink.wait()
    ex.submit("xtr-2", FRAME_OPERATION, op)  # same one-use nonce
    assert sink.wait()
    assert decode_msg(sink.frames[1][2]) == {"kind": KIND_REJECT, "reason": "replay"}
    # The replay never reached the provider.
    assert calls == ["q"]


def test_stale_generation_is_rejected_before_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker came up at GEN + 1; a delivery stamped for GEN is fence-rejected.
    ex, sink, calls = _executor(monkeypatch, generation=GEN + 1)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence(target_generation=GEN)))
    assert sink.wait()
    assert decode_msg(sink.frames[0][2]) == {
        "kind": KIND_REJECT,
        "reason": "generation",
    }
    assert calls == []


def test_wrong_audience_and_digest_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex, sink, calls = _executor(monkeypatch)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence(target_id="wrk-other")))
    assert sink.wait()
    assert decode_msg(sink.frames[0][2])["reason"] == "audience"
    ex.submit("xtr-2", FRAME_OPERATION, _op(_fence(request_digest="deadbeef")))
    assert sink.wait()
    assert decode_msg(sink.frames[1][2])["reason"] == "digest"
    assert calls == []


@pytest.mark.parametrize(
    ("fence_over", "req_interface", "reason"),
    [
        ({"provider": "other"}, SEARCH_INTERFACE, "provider"),
        ({"policy_class": "strict"}, SEARCH_INTERFACE, "policy"),
        ({"deadline_epoch": time.time() - 1}, SEARCH_INTERFACE, "expired"),
        ({"interface": "other/v1"}, SEARCH_INTERFACE, "interface"),
        ({}, "other/v1", "interface_mismatch"),
        ({"max_results": 1}, SEARCH_INTERFACE, "budget"),
    ],
)
def test_fence_branches_reject_before_egress(
    monkeypatch: pytest.MonkeyPatch,
    fence_over: dict[str, Any],
    req_interface: str,
    reason: str,
) -> None:
    ex, sink, calls = _executor(monkeypatch)
    ex.submit(
        "xtr-1", FRAME_OPERATION, _op(_fence(**fence_over), req_interface=req_interface)
    )
    assert sink.wait()
    assert decode_msg(sink.frames[0][2]) == {"kind": KIND_REJECT, "reason": reason}
    # Every fence branch rejects before any provider egress.
    assert calls == []


def test_the_keyed_secret_never_enters_the_reply_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker reads its keyed credential from its own environment; it must never ride
    # the outgoing attachment frame back to the supervisor and origin.
    probe = "KEYED-CREDENTIAL-ABSENT-PROBE"  # nosec B105 - test probe, not a credential
    ex, sink, _ = _executor(monkeypatch, api_key=probe)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    assert sink.wait()
    _session_id, kind, frame = sink.frames[0]
    assert kind == FRAME_REPLY
    assert probe.encode() not in frame
    assert probe not in frame.decode("latin-1")


def test_crashed_egress_sends_no_reply_leaving_it_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex, sink, _ = _executor(monkeypatch, raises=RuntimeError("boom"))
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    # No reply frame: the origin's carriage times out into an ambiguous delivery.
    assert sink.wait(0.5) is False


def test_a_misprovisioned_provider_is_terminal_not_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The keyed provider has no key, so building it raises: a deterministic config
    # fault. The executor returns a terminal unavailable outcome rather than crashing
    # into a no-reply ambiguous loss that the origin would re-drive to exhaustion.
    def _raise(_cfg: Any) -> None:
        raise ValueError("the serper web-search provider needs WEB_SEARCH_API_KEY")

    monkeypatch.setattr("shared.tools.search.providers.build_search_provider", _raise)
    sink = _Sink()
    ex = WorkerExternalToolExecutor(
        worker_id=WORKER,
        generation=GEN,
        provider="fake",
        api_key=None,
        result_sink=sink,
    )
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    assert sink.wait()
    session_id, kind, frame = sink.frames[0]
    # A terminal reply frame, not a no-reply ambiguous loss.
    assert (session_id, kind) == ("xtr-1", FRAME_REPLY)
    body = decode_msg(frame)
    assert body["kind"] == KIND_RESULT
    assert body["outcome"]["status"] == "unavailable"


def test_cancel_reaps_and_fences_a_late_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = threading.Event()
    ex, sink, _ = _executor(monkeypatch, block=block)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    ex.submit("xtr-1", FRAME_CANCEL, b"")
    assert sink.wait()
    assert sink.frames[0][1] == FRAME_REAP
    # Release the blocked egress: its late reply is fenced by the cancelled set.
    block.set()
    assert sink.wait(0.5) is False
    # The cancelled egress leaves neither bookkeeping set populated.
    assert ex._cancelled == set()
    assert ex._inflight == {}


def test_cancel_before_start_drops_the_inflight_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One worker thread, held by a blocked op, so the second op stays queued and its
    # cancel wins before it starts. Its future is cancelled, so _run never fires — the
    # cancel must drop the in-flight entry itself and record nothing to suppress.
    block = threading.Event()
    ex, sink, calls = _executor(monkeypatch, block=block, max_workers=1)
    ex.submit("xtr-busy", FRAME_OPERATION, _op(_fence()))
    # Wait until the single worker thread is occupied by the busy op, so the second op
    # is queued behind it and its cancel wins before it starts.
    for _ in range(200):
        if calls:
            break
        time.sleep(0.01)
    assert calls == ["q"]
    ex.submit("xtr-queued", FRAME_OPERATION, _op(_fence(delivery_nonce="xdn-2")))
    ex.submit("xtr-queued", FRAME_CANCEL, b"")
    assert sink.wait()
    assert sink.frames[0] == ("xtr-queued", FRAME_REAP, b"")
    assert "xtr-queued" not in ex._inflight
    assert ex._cancelled == set()
    # The queued op never reached the provider.
    block.set()
    assert calls == ["q"]
