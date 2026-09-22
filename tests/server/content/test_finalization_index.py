"""Which content an outcome finalization is bound to, and which drive decides it."""

from types import SimpleNamespace
from typing import Any, cast

from server.content import FinalizationIndex
from shared.content import reference_for

_SCOPE = "tenant-a"


class _FakeRedis:
    """A redis whose set-if-absent is the only way a first writer is decided."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set_value(self, key: str, value: str) -> None:
        self.values[key] = value

    def set_value_if_absent(self, key: str, value: str) -> bool:
        if key in self.values:
            return False
        self.values[key] = value
        return True

    def expire(self, key: str, ttl: int) -> None:
        self.expirations[key] = ttl


def _index(fake: _FakeRedis, ttl: float = 0.0) -> FinalizationIndex:
    return FinalizationIndex(cast(Any, SimpleNamespace(sync=fake)), ttl_sec=ttl)


def test_a_finalization_binds_the_content_it_materialized() -> None:
    fake = _FakeRedis()
    reference = reference_for(_SCOPE, b"the outcome", media_type="application/json")

    manifest = _index(fake).record(_SCOPE, "idm-1", reference)

    assert manifest.content == reference
    assert manifest.idempotency_key == "idm-1"


def test_a_re_drive_resolves_the_first_materialization() -> None:
    fake = _FakeRedis()
    index = _index(fake)
    first = reference_for(_SCOPE, b"sampled once", media_type="application/json")
    index.record(_SCOPE, "idm-1", first)

    again = index.record(
        _SCOPE,
        "idm-1",
        reference_for(_SCOPE, b"sampled twice", media_type="application/json"),
    )

    assert again.content == first
    settled = index.find(_SCOPE, "idm-1")
    assert settled is not None and settled.content == first


class _RacingRedis(_FakeRedis):
    """A redis where another drive wins the key before this one writes."""

    def __init__(self, winner: str) -> None:
        super().__init__()
        self._winner = winner

    def set_value_if_absent(self, key: str, value: str) -> bool:
        if key not in self.values:
            self.values[key] = self._winner
            return False
        return super().set_value_if_absent(key, value)


def test_a_drive_that_loses_the_write_takes_the_winners_content() -> None:
    """Two drives race the same key and the refused one adopts what the winner bound.

    Both drives see no binding when they look, so the decision has to be the write
    itself. Reading first and then writing would let both believe they were first: the
    later one's content would end up bound and each caller would be told its own stood,
    which is exactly the outcome this index exists to prevent.
    """
    winner = reference_for(_SCOPE, b"sampled once", media_type="application/json")
    index_for_winner = _index(_FakeRedis())
    winning_manifest = index_for_winner.record(_SCOPE, "idm-1", winner)

    fake = _RacingRedis(winning_manifest.model_dump_json())
    loser = _index(fake).record(
        _SCOPE, "idm-1", reference_for(_SCOPE, b"sampled twice")
    )

    assert loser.content == winner
    settled = _index(fake).find(_SCOPE, "idm-1")
    assert settled is not None and settled.content == winner


def test_a_scope_does_not_see_another_scopes_finalization() -> None:
    fake = _FakeRedis()
    index = _index(fake)
    index.record(_SCOPE, "idm-1", reference_for(_SCOPE, b"mine"))

    assert index.find("tenant-b", "idm-1") is None
