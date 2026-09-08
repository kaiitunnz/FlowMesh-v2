"""The serve-task residency binding store versions, drains, and persists a binding.

Each public serve task maps to one live binding keyed by a per-task family; a
re-adoption supersedes the prior generation, a drain refuses new calls, and the store
round-trips via its snapshot.
"""

from server.serve import ServeBindingSnapshot, ServeBindingStore, serve_family_key
from server.serve.binding import ServeBindingStatus
from server.task.v2.representations.operators import ServiceInterface


def _adopt(store: ServeBindingStore, task_id: str = "tsk-1", model: str = "org/model"):
    return store.adopt(
        task_id,
        service_ref=model,
        interface=ServiceInterface.CHAT,
        isolation=None,
        adapter=None,
        adapter_source=None,
        engine_batch_key=f"{model}|chat",
        max_output_tokens=256,
    )


def test_adopt_registers_a_live_binding_at_generation_zero() -> None:
    store = ServeBindingStore()
    binding = _adopt(store)
    assert binding.binding_generation == 0
    assert binding.live
    assert store.live("tsk-1") is binding
    assert binding.family == serve_family_key("tsk-1") == "serve/tsk-1"


def test_readopt_supersedes_and_bumps_the_generation() -> None:
    store = ServeBindingStore()
    _adopt(store)
    second = _adopt(store)
    assert second.binding_generation == 1
    assert store.live("tsk-1") is second


def test_drain_then_remove_refuses_new_calls() -> None:
    store = ServeBindingStore()
    _adopt(store)
    drained = store.drain("tsk-1")
    assert drained is not None and drained.status is ServeBindingStatus.DRAINING
    # A drained binding is no longer live, so the edge rejects a new request.
    assert store.live("tsk-1") is None
    assert store.get("tsk-1") is drained
    store.remove("tsk-1")
    assert store.get("tsk-1") is None


def test_dependency_and_profile_carry_the_bound_identity() -> None:
    store = ServeBindingStore()
    binding = _adopt(store)
    dep = binding.dependency()
    assert dep.service_ref == "org/model"
    assert dep.interface is ServiceInterface.CHAT
    profile = binding.profile(descriptor_digest="sha")
    assert profile.serve_task_id == "tsk-1"
    assert profile.binding_generation == 0
    assert profile.descriptor_digest == "sha"
    assert profile.max_output_tokens == 256


def test_snapshot_round_trips() -> None:
    store = ServeBindingStore()
    _adopt(store, task_id="tsk-a")
    _adopt(store, task_id="tsk-b")
    snapshot = store.to_snapshot()
    assert isinstance(snapshot, ServeBindingSnapshot)

    restored = ServeBindingStore()
    restored.load_snapshot(snapshot)
    assert {b.serve_task_id for b in restored.all()} == {"tsk-a", "tsk-b"}
    assert restored.live("tsk-a") is not None
