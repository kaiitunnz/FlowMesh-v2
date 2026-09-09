"""The external status-terminal store is first-wins and durable.

A serve invocation's terminal is a durable fact keyed by invocation_id; the first
fence-matching terminal is authoritative and a duplicate never overwrites it, so a
settled credit is never released twice.
"""

from server.serve import (
    ServeStatusTerminal,
    ServeTerminalSnapshot,
    ServeTerminalStatus,
    ServeTerminalStore,
)


def test_record_is_first_wins() -> None:
    store = ServeTerminalStore()
    assert store.record(
        ServeStatusTerminal(invocation_id="inv-1", status=ServeTerminalStatus.COMPLETED)
    )
    # A second terminal for the same invocation does not win and does not overwrite.
    assert not store.record(
        ServeStatusTerminal(invocation_id="inv-1", status=ServeTerminalStatus.FAILED)
    )
    got = store.get("inv-1")
    assert got is not None and got.status is ServeTerminalStatus.COMPLETED


def test_all_lists_recorded_terminals() -> None:
    store = ServeTerminalStore()
    store.record(
        ServeStatusTerminal(invocation_id="inv-1", status=ServeTerminalStatus.COMPLETED)
    )
    store.record(
        ServeStatusTerminal(invocation_id="inv-2", status=ServeTerminalStatus.FAILED)
    )
    assert {t.invocation_id for t in store.all()} == {"inv-1", "inv-2"}


def test_snapshot_round_trips() -> None:
    store = ServeTerminalStore()
    store.record(
        ServeStatusTerminal(
            invocation_id="inv-1", status=ServeTerminalStatus.CANCELLED, detail="reap"
        )
    )
    snapshot = store.to_snapshot()
    assert isinstance(snapshot, ServeTerminalSnapshot)

    restored = ServeTerminalStore()
    restored.load_snapshot(snapshot)
    got = restored.get("inv-1")
    assert got is not None and got.status is ServeTerminalStatus.CANCELLED
    assert got.detail == "reap"
