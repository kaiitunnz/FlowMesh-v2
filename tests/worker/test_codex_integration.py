"""A live Codex app-server proves the bounded held-facade recovery outside simulation.

A real ``codex app-server`` runs against a local Responses-compatible backend; the
model defers one mediated facade call, the fabric settles it, and the outcome injects
back over ``thread/inject_items``. Across a ``kill -9`` of the app-server the same
persisted rollout resumes by thread id under a stable ``CODEX_HOME``, and the settled
facade effect runs exactly once. This exercises the shipped adapter contract against a
live process rather than a fake transport; it is not a CPU Docker end-to-end test,
whose environment lacks the Codex binary.

The proof is bounded to a single-forward, single-facade recovery. The facade
origination is signalled by the model on its turn output; the resolution path, rollout
persistence, resume, and exactly-once effect are real.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "openai_codex", reason="needs the openai-codex worker harness dependency"
)

from shared.harness import (  # noqa: E402
    DeliveredOutcome,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)
from worker.executors.harness.codex import CodexAppServerHarnessAdapter  # noqa: E402
from worker.executors.harness.codex_transport import (  # noqa: E402
    CodexTransportConfig,
    RealCodexAppServerTransport,
)

_FACADE_ENVELOPE = json.dumps({"facade": {"tool": "spawn_agent", "region": "reviewer"}})
_FINAL_TEXT = "final"


def _codex_available() -> bool:
    from openai_codex.client import CodexConfig, _resolve_codex_bin

    try:
        return _resolve_codex_bin(CodexConfig()).exists()
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.codex_integration,
    pytest.mark.skipif(not _codex_available(), reason="codex binary unavailable"),
]


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


def _message_output(text: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "id": "msg_0",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }
    ]


class _ResponsesStub:
    """A local Responses backend: it defers a facade until its output is injected."""

    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        def _facade_resolved(body: dict[str, Any]) -> bool:
            return any(
                item.get("type") == "function_call_output"
                and str(item.get("call_id", "")).startswith("fab-")
                for item in body.get("input", [])
            )

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode()) if n else {}
                text = _FINAL_TEXT if _facade_resolved(body) else _FACADE_ENVELOPE
                output = _message_output(text)
                events: list[dict[str, Any]] = [
                    {"type": "response.created", "response": {"id": "r"}}
                ]
                events.append(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": output[0],
                    }
                )
                events.append(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "r",
                            "status": "completed",
                            "output": output,
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    }
                )
                data = _sse(events)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    @property
    def base_url(self) -> str:
        port = int(self._server.server_address[1])
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> "_ResponsesStub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class _Fabric:
    """The fabric's role: settle a boundary once, keyed by a stable idempotency key."""

    def __init__(self) -> None:
        self._settled: dict[str, DeliveredOutcome] = {}
        self.effect_runs: dict[str, int] = {}

    def settle(
        self, result: HarnessResult, value: str = _FINAL_TEXT
    ) -> DeliveredOutcome:
        assert result.request is not None
        corr = result.request.call_correlation
        assert corr is not None
        if corr not in self._settled:
            key = f"idm-{corr}"
            self.effect_runs[key] = self.effect_runs.get(key, 0) + 1
            self._settled[corr] = DeliveredOutcome(
                call_correlation=corr,
                idempotency_key=key,
                kind=OutcomeKind.RESULT,
                value=value,
            )
        return self._settled[corr]

    @property
    def total_effects(self) -> int:
        return sum(self.effect_runs.values())


def _transport(base_url: str, home: Path) -> RealCodexAppServerTransport:
    return RealCodexAppServerTransport(
        CodexTransportConfig(base_url=base_url, model="gpt-5-codex", codex_home=home)
    )


def test_kill9_before_injection_runs_the_facade_effect_once(tmp_path: Path) -> None:
    home = tmp_path / "codex_home"
    fabric = _Fabric()
    with _ResponsesStub() as stub:
        issue = _transport(stub.base_url, home)
        adapter = CodexAppServerHarnessAdapter(issue, "v1")
        first = adapter.start("a", capsule=None, outcomes=[])
        assert first.kind is HarnessResultKind.BOUNDARY
        assert (
            first.request is not None and first.request.child_region_ref == "reviewer"
        )

        # The app-server dies after the boundary issues, before its outcome injects.
        os.kill(issue.pid, 9)

        recover = _transport(stub.base_url, home)
        adapter2 = CodexAppServerHarnessAdapter(recover, "v1")
        done = adapter2.start(
            "a", capsule=first.capsule, outcomes=[fabric.settle(first)]
        )
        recover.close()

    assert done.kind is HarnessResultKind.COMPLETION and done.value == _FINAL_TEXT
    assert fabric.total_effects == 1


def test_kill9_after_injection_before_terminal_does_not_reexecute(
    tmp_path: Path,
) -> None:
    home = tmp_path / "codex_home"
    fabric = _Fabric()
    with _ResponsesStub() as stub:
        run = _transport(stub.base_url, home)
        adapter = CodexAppServerHarnessAdapter(run, "v1")
        first = adapter.start("a", capsule=None, outcomes=[])
        assert first.kind is HarnessResultKind.BOUNDARY
        outcome = fabric.settle(first)

        # Inject the outcome and complete the turn, then lose the app-server before
        # the completion is durably recorded.
        completed = adapter.start("a", capsule=first.capsule, outcomes=[outcome])
        assert completed.kind is HarnessResultKind.COMPLETION
        os.kill(run.pid, 9)

        # Recovery re-delivers the same keyed outcome on the same rollout; the effect
        # is deduped, not re-run, and the episode still completes.
        recover = _transport(stub.base_url, home)
        adapter2 = CodexAppServerHarnessAdapter(recover, "v1")
        again = adapter2.start(
            "a", capsule=first.capsule, outcomes=[fabric.settle(first)]
        )
        recover.close()

    assert again.kind is HarnessResultKind.COMPLETION
    assert fabric.total_effects == 1
