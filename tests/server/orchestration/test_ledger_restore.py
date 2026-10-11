"""A stored ledger restores every fact it captured."""

from server.orchestration import LedgerSnapshot, OrchestrationEngine
from tests.server.orchestration.helpers import chain_bundle, engine, rehydrated


def _failed_chain() -> OrchestrationEngine:
    eng = engine(chain_bundle())
    eng.on_dispatched("A", "w1")
    eng.on_failed("A", "boom", retryable=False)
    return eng


def test_a_snapshot_captures_every_ledger_field() -> None:
    snapshot = _failed_chain().to_snapshot()

    assert snapshot.model_fields_set == set(LedgerSnapshot.model_fields)


def test_a_restored_ledger_captures_the_same_snapshot() -> None:
    live = _failed_chain()

    restored = rehydrated(live)

    assert (
        restored.to_snapshot().model_dump_json() == live.to_snapshot().model_dump_json()
    )
