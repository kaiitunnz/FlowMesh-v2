"""The worker-originated mediated-operation permit contract and its id."""

from shared.tools.contract import MediatedOperationPermit
from shared.tools.search.schema import SEARCH_INTERFACE, tool_request_digest
from shared.utils.ids import PREFIX_MEDIATED_PERMIT, new_mediated_permit_id


def _permit(**overrides: object) -> MediatedOperationPermit:
    fields: dict[str, object] = {
        "permit_id": new_mediated_permit_id(),
        "agent_task_id": "tsk-agent",
        "call_correlation": "m0",
        "interface": SEARCH_INTERFACE,
        "subject": SEARCH_INTERFACE,
        "invocation_id": "inv-1",
        "idempotency_key": "idm-1",
        "request_digest": tool_request_digest(SEARCH_INTERFACE, "weather", 5),
        "target_id": "wkr-1",
        "target_generation": 3,
        "policy_epoch": 1,
        "deadline_epoch": 123.0,
        "max_results": 5,
        "timeout_sec": 10.0,
        "result_char_cap": 4000,
    }
    fields.update(overrides)
    return MediatedOperationPermit(**fields)  # type: ignore[arg-type]


def test_permit_id_prefix_is_unguessable() -> None:
    pid = new_mediated_permit_id()
    assert pid.startswith(f"{PREFIX_MEDIATED_PERMIT}-")
    assert pid != new_mediated_permit_id()


def test_permit_round_trips_and_is_frozen() -> None:
    permit = _permit()
    restored = MediatedOperationPermit.model_validate_json(permit.model_dump_json())
    assert restored == permit
    assert restored.policy_class == "default"


def test_permit_binds_the_canonical_request_digest() -> None:
    # The permit's digest is the same canonical hash the worker computes from the raw
    # request, so a tampered request fails the worker fence.
    permit = _permit()
    assert permit.request_digest == tool_request_digest(SEARCH_INTERFACE, "weather", 5)
    assert permit.request_digest != tool_request_digest(SEARCH_INTERFACE, "weather", 6)
