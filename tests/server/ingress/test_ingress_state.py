"""Ingress-terminal facts are durable and settle once.

The terminal fact is the fenced record the Admission controller consumes by
``invocation_id``. The first record wins so a duplicate or late report never releases a
credit twice, and the facts round-trip through the snapshot so a restart reconciles.
"""

from server.ingress import IngressTerminal, IngressTerminalStore
from server.ingress.state import IngressTerminalStatus


def test_first_terminal_wins_and_duplicates_are_no_ops():
    store = IngressTerminalStore()
    assert store.record(
        IngressTerminal(invocation_id="inv-1", status=IngressTerminalStatus.COMPLETED)
    )
    # A late failure report cannot overwrite the recorded completion.
    assert not store.record(
        IngressTerminal(invocation_id="inv-1", status=IngressTerminalStatus.FAILED)
    )
    recorded = store.get("inv-1")
    assert recorded is not None and recorded.status is IngressTerminalStatus.COMPLETED


def test_snapshot_round_trips_the_facts():
    store = IngressTerminalStore()
    store.record(
        IngressTerminal(invocation_id="inv-1", status=IngressTerminalStatus.COMPLETED)
    )
    store.record(
        IngressTerminal(invocation_id="inv-2", status=IngressTerminalStatus.FAILED)
    )
    restored = IngressTerminalStore()
    restored.load_snapshot(store.to_snapshot())
    assert restored.get("inv-1") is not None
    assert restored.get("inv-2") is not None
    # A rehydrated fact still refuses a duplicate.
    assert not restored.record(
        IngressTerminal(invocation_id="inv-1", status=IngressTerminalStatus.CANCELLED)
    )
