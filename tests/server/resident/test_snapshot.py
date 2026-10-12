"""The live resident facts round-trip through their stored layout.

What the stores flush loads back through the decoder a restart uses: the DemandLedger is
rebuilt from pending claims while an admitted claim is not re-enqueued, the credit
ledger recomputes the outstanding credit from the loaded claims, and an endpoint
credential is never stored.
"""

import asyncio
import json

from server.resident import (
    AdmissionController,
    ClaimState,
    InvocationSubject,
    InvocationSubjectKind,
    ReplicaEndpoint,
    ReplicaIncarnation,
    ReplicaState,
    ResidentStores,
)
from server.resident.durability import FAMILY, MemoryResidentDurability, live_field
from tests.server.resident._helpers import PROFILE, stored_snapshot, warm_stores

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")


def _seed() -> tuple[ResidentStores, str, str]:
    stores = warm_stores(slots=2, durability=MemoryResidentDurability())
    ctl = AdmissionController(stores)
    reserved = ctl.raise_claim(
        invocation_id="inv-1", subject=_SUBJECT, family="fam", profile=PROFILE
    )
    ctl.admit(reserved, PROFILE, idempotency_key="idm-x")
    ctl.accept_and_authorize(reserved, idempotency_key="idm-x", origin_id="rog-1")
    ctl.on_stream_started(reserved)
    pending = ctl.raise_claim(
        invocation_id="inv-2", subject=_SUBJECT, family="fam", profile=PROFILE
    )
    return stores, reserved.claim_id, pending.claim_id


def test_the_live_facts_round_trip() -> None:
    stores, reserved, pending = _seed()

    restored = ResidentStores()
    restored.load_snapshot(stored_snapshot(stores))

    assert restored.families.get("fam") is not None
    replica = restored.directory.get("rpl-1")
    assert replica is not None and replica.state is ReplicaState.WARM
    assert restored.leases.all() == stores.leases.all()
    assert {c.claim_id for c in restored.claims.all()} == {reserved, pending}
    assert restored.invocations.get("inv-1") is not None


def test_a_replica_endpoint_credential_is_never_stored() -> None:
    stores = warm_stores(durability=MemoryResidentDurability())
    stores.directory.add(
        ReplicaIncarnation(
            replica_id="rpl-keyed",
            family="fam",
            incarnation=1,
            state=ReplicaState.WARM,
            endpoint=ReplicaEndpoint(
                base_url="http://replica/v1", model="m", api_key="super-secret"
            ),
        )
    )
    restored = ResidentStores()
    restored.load_snapshot(stored_snapshot(stores))

    assert isinstance(stores.durability, MemoryResidentDurability)
    assert not any("super-secret" in v for v in stores.durability.live.values())
    replica = restored.directory.get("rpl-keyed")
    assert replica is not None and replica.endpoint is not None
    assert replica.endpoint.api_key is None


def test_derived_credit_and_demand_rebuild_on_load() -> None:
    stores, reserved, pending = _seed()
    restored = ResidentStores()
    restored.load_snapshot(stored_snapshot(stores))

    assert restored.credit_ledger.held("rpl-1") == 1
    assert restored.demand.get(pending) is not None
    assert restored.demand.get(reserved) is None
    claim = restored.claims.get(reserved)
    assert claim is not None and claim.state is ClaimState.STREAMING


def test_an_unreadable_family_is_dropped_without_stranding_the_claims() -> None:
    """A family definition that fails validation costs a cold start, since the next
    eligible demand re-derives the same family; the claims it served still load."""
    stores, reserved, _pending = _seed()
    stores.flush()
    durability = stores.durability
    assert isinstance(durability, MemoryResidentDurability)
    durability.live[live_field(FAMILY, "legacy")] = json.dumps(
        {"family": "legacy", "engine_batch_key": "k"}
    )

    loaded = asyncio.run(durability.load_snapshot_async())

    assert loaded is not None
    assert "legacy" not in {family.family for family in loaded.families}
    assert reserved in {claim.claim_id for claim in loaded.claims}
