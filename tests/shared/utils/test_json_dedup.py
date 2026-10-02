"""The deduplicated JSON form and its check."""

from shared.utils.json import dedup_json, is_deduped_json, restore_json


def test_a_deduplicated_payload_is_recognized_and_restores() -> None:
    plain = {"task_id": "tsk-1", "spec": {"name": "tsk-1"}, "n": 3}
    deduped = dedup_json(plain)

    assert is_deduped_json(deduped)
    assert restore_json(deduped) == plain


def test_a_plain_payload_is_not_deduplicated() -> None:
    assert not is_deduped_json({"task_id": "tsk-1"})
    assert not is_deduped_json({"content": {}, "data": {}, "task_id": "tsk-1"})
