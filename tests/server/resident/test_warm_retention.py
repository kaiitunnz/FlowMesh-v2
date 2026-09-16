"""Per-family idle retention, and the warmth preference a family definition carries."""

from server.resident.lifecycle import LifecycleScaleManager
from server.resident.policy import ResidentPolicyLimits
from server.resident.state import (
    ClaimCredit,
    ReplicaIncarnation,
    ReplicaState,
    ServiceFamily,
)
from server.resident.stores import ResidentStores
from server.task.v2.representations.plan import ResidencyWarmth

from ._helpers import warm_stores
from .test_claim_fsm import new_claim, reserve

# The base idle window every test derives its sweep instants from.
_B = 30.0
_EPSILON = 1.0
# The instant the replica went idle, as an epoch second and its ISO form.
_IDLE_AT = 1_000_000_000.0
_IDLE_AT_ISO = "2001-09-09T01:46:40+00:00"


def _stores(warmth: ResidencyWarmth | None = None) -> ResidentStores:
    stores = warm_stores()
    stores.families.register(
        ServiceFamily(
            family="fam", engine_batch_key="fam", model_ref="m", warmth=warmth
        )
    )
    return stores


def _manager(
    stores: ResidentStores, *, retain: float = _B, stop_fn=None
) -> LifecycleScaleManager:
    return LifecycleScaleManager(
        stores,
        limits=ResidentPolicyLimits(max_replicas_per_family=1),
        admission_slots=2,
        idle_retain_sec=retain,
        stop_fn=stop_fn,
    )


def _replica(stores: ResidentStores) -> ReplicaIncarnation:
    replica = stores.directory.get("rpl-1")
    assert replica is not None
    return replica


def _family(stores: ResidentStores) -> ServiceFamily:
    definition = stores.families.get("fam")
    assert definition is not None
    return definition


def _sweep(stores: ResidentStores, at: float, **kw) -> ReplicaState:
    _replica(stores).last_active_at = _IDLE_AT_ISO
    _manager(stores, **kw).sweep_idle(now_ts=_IDLE_AT + at)
    return _replica(stores).state


def test_an_unstyled_family_drains_at_the_base_window() -> None:
    assert _sweep(_stores(), _B + _EPSILON) is ReplicaState.DRAINING


def test_a_warm_family_survives_the_base_window() -> None:
    assert _sweep(_stores(ResidencyWarmth.WARM), _B + _EPSILON) is ReplicaState.WARM


def test_a_warm_family_drains_at_twice_the_base_window() -> None:
    assert (
        _sweep(_stores(ResidencyWarmth.WARM), 2 * _B + _EPSILON)
        is ReplicaState.DRAINING
    )


def test_a_persisted_warmth_the_fabric_does_not_express_reads_as_unset() -> None:
    # A definition stored under another vocabulary still loads; it simply carries no
    # preference, so its family keeps the base window.
    definition = ServiceFamily.model_validate(
        {
            "family": "fam",
            "engine_batch_key": "fam",
            "model_ref": "m",
            "warmth": "scalding",
        }
    )
    assert definition.warmth is None

    stores = warm_stores()
    stores.families.register(definition)
    assert _sweep(stores, _B + _EPSILON) is ReplicaState.DRAINING


def test_a_persisted_warm_definition_loads_as_the_fabric_s_own_value() -> None:
    definition = ServiceFamily.model_validate_json(
        '{"family": "fam", "engine_batch_key": "fam", "model_ref": "m",'
        ' "warmth": "warm"}'
    )
    assert definition.warmth is ResidencyWarmth.WARM


def test_a_non_positive_base_disables_every_family() -> None:
    for warmth in (None, ResidencyWarmth.WARM):
        state = _sweep(_stores(warmth), 10 * _B, retain=0.0)
        assert state is ReplicaState.WARM


def test_a_warm_family_is_held_while_its_replica_holds_credit() -> None:
    stores = _stores(ResidencyWarmth.WARM)
    claim = new_claim(invocation_id="inv-x", family="fam")
    reserve(claim, replica_id="rpl-1", incarnation=1, credit=ClaimCredit(slots=1))
    stores.claims.add(claim)
    assert _sweep(stores, 10 * _B) is ReplicaState.WARM


def test_warmth_creates_no_replica_and_no_claim() -> None:
    stores = _stores(ResidencyWarmth.WARM)
    before = len(stores.directory.all())
    _sweep(stores, _B + _EPSILON)
    assert len(stores.directory.all()) == before
    assert stores.claims.all() == []


def test_a_warm_preference_refines_an_unset_family() -> None:
    stores = _stores()
    stores.families.register(
        ServiceFamily(
            family="fam",
            engine_batch_key="fam",
            model_ref="m",
            warmth=ResidencyWarmth.WARM,
        )
    )
    assert _family(stores).warmth is ResidencyWarmth.WARM


def test_a_later_definition_never_demotes_a_warm_family() -> None:
    stores = _stores(ResidencyWarmth.WARM)
    stores.families.register(
        ServiceFamily(family="fam", engine_batch_key="fam", model_ref="m")
    )
    assert _family(stores).warmth is ResidencyWarmth.WARM


def test_an_incompatible_definition_never_rewrites_a_family() -> None:
    stores = _stores()
    stores.families.register(
        ServiceFamily(
            family="fam",
            engine_batch_key="other-key",
            model_ref="other-model",
            warmth=ResidencyWarmth.WARM,
        )
    )
    definition = _family(stores)
    assert definition.warmth is None
    assert definition.engine_batch_key == "fam"
    assert definition.model_ref == "m"
