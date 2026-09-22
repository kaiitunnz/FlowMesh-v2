"""The agent-episode executor hydrates a reference-backed outcome before injection."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shared.content import reference_for
from shared.harness import (
    REQUIRED_MEDIATED_FACADES,
    DeliveredOutcome,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessResult,
    HarnessResultKind,
    MediatedFacade,
)
from shared.outcome import OutcomeManifest
from shared.tasks.task_type import TaskType
from tests.shared.outcome_helpers import InMemoryContentStore
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.base_executor import ExecutionError
from worker.executors.harness import register_adapter


class _RecordingAdapter(HarnessAdapter):
    def __init__(self) -> None:
        self.injected: list[DeliveredOutcome] = []

    def backend_key(self) -> HarnessBackendKey:
        return HarnessBackendKey(backend="fake", version="v1")

    def start(self, activation_id, *, capsule, outcomes) -> HarnessResult:
        self.injected = list(outcomes)
        return HarnessResult(kind=HarnessResultKind.COMPLETION, value="done")

    def cancel(self, activation_id: str) -> None:
        pass

    def mediated_facades(self) -> frozenset[MediatedFacade]:
        return REQUIRED_MEDIATED_FACADES


def _dispatch_with_ref(manifest: OutcomeManifest):
    return make_worker_task_message(
        {"taskType": "agent"},
        task_type=TaskType.AGENT,
        agent_episode={
            "backend": {"backend": "fake", "version": "v1"},
            "capsule_blob": "after:c0",
            "delivered_outcomes": [
                DeliveredOutcome(
                    call_correlation="c0", outcome_ref=manifest
                ).model_dump(mode="json")
            ],
        },
    )


class _Plane:
    """A content plane whose every task reads one in-memory store."""

    def __init__(self, store: InMemoryContentStore) -> None:
        self._store = store

    def for_task(self, task_id: str) -> InMemoryContentStore:
        return self._store


def _lifecycle(store: InMemoryContentStore) -> Any:
    return SimpleNamespace(content_plane=_Plane(store), responses_facade=None)


def test_reference_outcome_is_hydrated_before_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryContentStore()
    manifest = store.materialize(
        "local", "idm-1", b"the-result", media_type="application/json"
    )
    adapter = _RecordingAdapter()
    register_adapter(
        "fake", lambda backend, task, config, facade, state, sandbox: adapter
    )

    ex = AgentEpisodeExecutor(make_worker_config(), lifecycle=_lifecycle(store))
    ex.run(_dispatch_with_ref(manifest), tmp_path)

    assert len(adapter.injected) == 1
    injected = adapter.injected[0]
    assert injected.value == "the-result" and injected.outcome_ref is None


def test_hydration_failure_fails_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryContentStore()
    absent = OutcomeManifest(
        content=reference_for("local", b"absent", media_type="application/json")
    )
    register_adapter(
        "fake",
        lambda backend, task, config, facade, state, sandbox: _RecordingAdapter(),
    )

    ex = AgentEpisodeExecutor(make_worker_config(), lifecycle=_lifecycle(store))
    with pytest.raises(ExecutionError, match="hydrat"):
        ex.run(_dispatch_with_ref(absent), tmp_path)
