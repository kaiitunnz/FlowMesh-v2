"""The Redis layout of the resident facts: one field per live entity, retirement and
archiving that refuse before writing and publish before they delete, and a load that
fails closed on anything it cannot vouch for."""

import asyncio
import json
from collections.abc import Sequence

import fakeredis
import pytest

from server.clients.redis import (
    resident_cs_key,
    resident_replica_history_key,
    resident_retired_key,
)
from server.registries.resident import ResidentRegistry
from server.resident import (
    AdmissionController,
    ClaimState,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ResidentStores,
)
from server.resident.durability import ResidentStateError
from server.resident.state import ReplicaState, RetiredInvocation, StoredReplica
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.resident._helpers import PROFILE, warm_stores

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")


def _registry() -> ResidentRegistry:
    return ResidentRegistry(fake_redis_client(fakeredis.FakeServer()))


def _consumed(registry: ResidentRegistry) -> tuple[ResidentStores, AdmissionController]:
    """Stores holding one invocation whose terminal was consumed, all of it durable."""
    stores = warm_stores(durability=registry)
    admission = AdmissionController(stores)
    claim = admission.raise_claim(
        invocation_id="inv-1", subject=_SUBJECT, family="fam", profile=PROFILE
    )
    admission.admit(claim, PROFILE, idempotency_key=None)
    admission.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    return stores, admission


def _live(registry: ResidentRegistry) -> dict[str, str]:
    return raw_control(registry).hgetall(resident_cs_key())


def _retired(registry: ResidentRegistry) -> dict[str, str]:
    return raw_control(registry).hgetall(resident_retired_key())


def test_the_live_facts_round_trip_one_field_per_entity() -> None:
    registry = _registry()
    stores, _admission = _consumed(registry)

    fields = set(_live(registry))
    assert fields == {
        "family:fam",
        "replica:rpl-1",
        "request:inv-1",
        f"claim:{stores.claims.by_invocation('inv-1')[0].claim_id}",
    }
    loaded = asyncio.run(registry.load_snapshot_async())
    assert loaded is not None
    assert [c.state for c in loaded.claims] == [ClaimState.TERMINAL]
    assert loaded.invocations[0].consumed is ClaimTerminalReason.COMPLETED


def test_retirement_publishes_its_record_and_deletes_the_live_fields() -> None:
    registry = _registry()
    stores, admission = _consumed(registry)

    assert admission.retire("inv-1")

    assert set(_live(registry)) == {"family:fam", "replica:rpl-1"}
    record = RetiredInvocation.model_validate_json(_retired(registry)["inv-1"])
    assert record.epoch_high == 0 and record.subject == _SUBJECT
    assert not stores.live("inv-1")


def test_a_retired_key_of_the_wrong_type_refuses_before_any_write() -> None:
    registry = _registry()
    stores, admission = _consumed(registry)
    raw_control(registry).set(resident_retired_key(), "not a hash")
    before = _live(registry)

    with pytest.raises(ResidentStateError, match="WRONGTYPE"):
        admission.retire("inv-1")

    assert _live(registry) == before
    assert stores.live("inv-1")


def test_a_stored_claim_still_open_refuses_retirement() -> None:
    registry = _registry()
    stores, admission = _consumed(registry)
    [claim] = stores.claims.by_invocation("inv-1")
    field = f"claim:{claim.claim_id}"
    raw_control(registry).hset(
        resident_cs_key(),
        field,
        json.dumps({**json.loads(_live(registry)[field]), "state": "accepted"}),
    )
    before = _live(registry)

    with pytest.raises(ResidentStateError, match="OPENCLAIM"):
        admission.retire("inv-1")

    assert _live(registry) == before and _retired(registry) == {}


class _ReplyLost(ResidentRegistry):
    """Applies each retirement, then loses its reply once."""

    lose = True

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        super().retire_invocation(record, fields)
        if self.lose:
            self.lose = False
            raise ConnectionError("reply lost")


def test_a_retirement_applied_but_unanswered_retries_to_the_same_result() -> None:
    registry = _ReplyLost(fake_redis_client(fakeredis.FakeServer()))
    stores, admission = _consumed(registry)
    registry.lose = True

    with pytest.raises(ConnectionError):
        admission.retire("inv-1")
    assert stores.live("inv-1")  # unconfirmed, so still owned here
    assert "request:inv-1" not in _live(registry)

    assert admission.retire("inv-1")
    assert not stores.live("inv-1")
    assert RetiredInvocation.model_validate_json(_retired(registry)["inv-1"])


def test_a_stored_record_with_a_higher_epoch_refuses_a_lower_one() -> None:
    registry = _registry()
    _stores, admission = _consumed(registry)
    raw_control(registry).hset(
        resident_retired_key(),
        "inv-1",
        RetiredInvocation(
            invocation_id="inv-1",
            subject=_SUBJECT,
            family="fam",
            epoch_high=5,
            terminal_reason=ClaimTerminalReason.COMPLETED,
        ).model_dump_json(),
    )

    with pytest.raises(ResidentStateError, match="EPOCH"):
        admission.retire("inv-1")
    assert "request:inv-1" in _live(registry)


def test_a_replica_the_store_holds_live_is_never_archived() -> None:
    registry = _registry()
    stores = warm_stores(durability=registry)
    stores.flush()
    replica = stores.directory.get("rpl-1")
    assert replica is not None
    stopped = StoredReplica(
        replica=replica.model_copy(update={"state": ReplicaState.STOPPED})
    )

    with pytest.raises(ResidentStateError, match="NOTINERT"):
        registry.archive_replica(stopped, ["replica:rpl-1"])

    assert "replica:rpl-1" in _live(registry)
    assert raw_control(registry).hgetall(resident_replica_history_key()) == {}


def test_a_history_key_of_the_wrong_type_refuses_an_archive_before_any_write() -> None:
    registry = _registry()
    stores = warm_stores(durability=registry)
    stores.flush()
    raw_control(registry).set(resident_replica_history_key(), "not a hash")
    before = _live(registry)
    replica = stores.directory.get("rpl-1")
    assert replica is not None
    stopped = StoredReplica(
        replica=replica.model_copy(update={"state": ReplicaState.STOPPED})
    )

    with pytest.raises(ResidentStateError, match="WRONGTYPE"):
        registry.archive_replica(stopped, ["replica:rpl-1"])

    assert _live(registry) == before


def test_a_load_refuses_the_single_value_state_naming_its_key() -> None:
    registry = _registry()
    raw_control(registry).set(resident_cs_key(), "{}")

    with pytest.raises(ResidentStateError, match="resident:cs"):
        asyncio.run(registry.load_snapshot_async())


@pytest.mark.parametrize(
    "corrupt",
    [
        {"claim:scl-x": json.dumps({"claim_id": "scl-x"})},
        {"unknown:x": "{}"},
        {"replica:rpl-1": "not json"},
    ],
)
def test_a_load_refuses_a_field_it_cannot_vouch_for(corrupt: dict[str, str]) -> None:
    registry = _registry()
    _consumed(registry)
    raw_control(registry).hset(resident_cs_key(), mapping=corrupt)

    with pytest.raises(ResidentStateError):
        asyncio.run(registry.load_snapshot_async())


def test_a_load_refuses_a_claim_whose_request_is_gone() -> None:
    registry = _registry()
    _consumed(registry)
    raw_control(registry).hdel(resident_cs_key(), "request:inv-1")

    with pytest.raises(ResidentStateError, match="names no stored request"):
        asyncio.run(registry.load_snapshot_async())
