"""Durable persistence of the authoritative resident-capacity control facts.

The snapshot round-trips the authoritative stores through JSON, the DemandLedger is
rebuilt from pending claims on load while an admitted claim is not re-enqueued, and the
derived credit ledger recomputes the outstanding credit from the rehydrated claims.
"""

from server.resident import (
    AdmissionController,
    ClaimState,
    ResidentSnapshot,
    ResidentStores,
)
from tests.server.resident._helpers import PROFILE, warm_stores


def _seed():
    stores = warm_stores(slots=2)
    ctl = AdmissionController(stores)
    reserved = ctl.raise_claim(
        invocation_id="inv-1", workflow_id="wfl-1", family="fam", profile=PROFILE
    )
    ctl.admit(reserved, PROFILE)
    ctl.on_enqueue_ack(reserved)
    pending = ctl.raise_claim(
        invocation_id="inv-2", workflow_id="wfl-1", family="fam", profile=PROFILE
    )
    return stores, reserved, pending


def test_snapshot_round_trips_authoritative_facts():
    stores, reserved, pending = _seed()

    blob = stores.to_snapshot().model_dump_json()
    restored = ResidentStores()
    restored.load_snapshot(ResidentSnapshot.model_validate_json(blob))

    assert restored.families.get("fam") is not None
    assert restored.directory.get("rpl-1").state.value == "warm"
    assert restored.leases.all() == stores.leases.all()
    assert {c.claim_id for c in restored.claims.all()} == {
        reserved.claim_id,
        pending.claim_id,
    }
    assert restored.invocations.get("inv-1") is not None


def test_replica_endpoint_credential_is_not_persisted():
    from server.resident import ReplicaEndpoint, ReplicaIncarnation, ReplicaState

    stores = warm_stores()
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
    blob = stores.to_snapshot().model_dump_json()
    assert "super-secret" not in blob
    restored = ResidentStores()
    restored.load_snapshot(ResidentSnapshot.model_validate_json(blob))
    assert restored.directory.get("rpl-keyed").endpoint.api_key is None


def test_derived_credit_and_demand_rebuild_on_load():
    stores, reserved, pending = _seed()
    restored = ResidentStores()
    restored.load_snapshot(stores.to_snapshot())

    # The still-streaming claim's credit is recomputed from the rehydrated facts.
    assert restored.credit_ledger.held("rpl-1") == 1
    # The pending claim returns to the demand queue; the admitted one does not.
    assert restored.demand.get(pending.claim_id) is not None
    assert restored.demand.get(reserved.claim_id) is None
    assert restored.claims.get(reserved.claim_id).state is ClaimState.STREAMING
