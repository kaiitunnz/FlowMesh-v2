"""The shared Lua fragments, run inside every script that includes them."""

import json
from typing import cast

import fakeredis
import pytest
from redis.exceptions import ResponseError

from server.content.finalizations import _RECORD
from server.reclamation.certification import _CERTIFY
from server.reclamation.collector import _CONDEMN
from server.reclamation.lua import HOLDERS, KINDS
from server.reclamation.multipart import _REGISTER, _TRANSFER
from server.reclamation.pins import _ACQUIRE
from server.reclamation.trees import _PROMOTE

# Each script refusing a wrong-typed key: its key count and the KEYS indexes it checks.
_REFUSING = {
    "pins acquire": (_ACQUIRE, 6, range(1, 7)),
    "tree promote": (_PROMOTE, 4, range(1, 4)),
    "multipart register": (_REGISTER, 6, range(1, 7)),
    "multipart transfer": (_TRANSFER, 2, range(1, 3)),
    "collector condemn": (_CONDEMN, 8, (3, 4, 5, 7, 8)),
    "finalization record": (_RECORD, 8, range(3, 9)),
}


@pytest.fixture
def raw() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def _keys(count: int) -> list[str]:
    return [f"k{i}" for i in range(1, count + 1)]


@pytest.mark.parametrize(
    ("script", "count", "index"),
    [
        pytest.param(script, count, index, id=f"{name}-KEYS[{index}]")
        for name, (script, count, checked) in _REFUSING.items()
        for index in checked
    ],
)
def test_each_script_refuses_a_wrong_typed_key_before_writing(
    raw: fakeredis.FakeRedis, script: str, count: int, index: int
) -> None:
    keys = _keys(count)
    raw.rpush(keys[index - 1], "x")

    with pytest.raises(ResponseError, match=f"^WRONGTYPE {keys[index - 1]}$"):
        raw.eval(script, count, *keys)

    assert cast(list[str], raw.keys()) == [keys[index - 1]]


@pytest.mark.parametrize("index", [2, 3, 4])
def test_a_wrong_typed_certification_key_withdraws_the_certificate(
    raw: fakeredis.FakeRedis, index: int
) -> None:
    keys = _keys(6)
    raw.set(keys[5], "7")
    raw.rpush(keys[index - 1], "x")

    assert raw.eval(_CERTIFY, 6, *keys, "scope") == ["wrongtype"]
    assert not raw.exists(keys[5])


def test_wrong_kind_admits_absent_and_allowed_types_and_skips_unchecked_keys(
    raw: fakeredis.FakeRedis,
) -> None:
    script = KINDS + "return wrong_kind({[1] = 'set hash', [3] = 'string'}) or 'ok'"
    raw.sadd("k1", "x")
    raw.rpush("k2", "x")

    assert raw.eval(script, 3, "k1", "k2", "k3") == "ok"
    raw.delete("k1")
    raw.hset("k1", "f", "v")
    assert raw.eval(script, 3, "k1", "k2", "k3") == "ok"
    raw.sadd("k3", "x")
    assert raw.eval(script, 3, "k1", "k2", "k3") == "k3"


def test_a_holder_list_drops_only_its_own_nonces_holds(
    raw: fakeredis.FakeRedis,
) -> None:
    add = HOLDERS + "add_holder(KEYS[1], ARGV[1], ARGV[2], ARGV[3])"
    drop = HOLDERS + "return drop_holder(KEYS[1], ARGV[1], ARGV[2])"
    raw.eval(add, 1, "pins", "d", "1:1", "a")
    raw.eval(add, 1, "pins", "d", "1:2", "b")
    raw.eval(add, 1, "pins", "d", "1:1", "c")

    assert raw.eval(drop, 1, "pins", "d", "1:9") == 0
    assert raw.eval(drop, 1, "pins", "d", "1:1") == 2
    assert json.loads(str(raw.hget("pins", "d"))) == [{"n": "1:2", "l": "b"}]
    assert raw.eval(drop, 1, "pins", "d", "1:2") == 1
    assert not raw.exists("pins")
    assert raw.eval(drop, 1, "pins", "d", "1:2") == 0
