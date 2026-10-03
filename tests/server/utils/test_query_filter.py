"""A query filter matches declared fields by attribute and rejects any other key."""

import time

import pytest
from pydantic import BaseModel
from starlette.datastructures import QueryParams

from server.utils.query import InvalidQuery, QueryFilter


class _SampleModel(BaseModel):
    id: str
    status: str
    stale: bool = False
    tags: list[str] = []
    env: dict = {}
    score: float | None = None


def _make(id: str, status: str = "IDLE", **kw) -> _SampleModel:
    return _SampleModel(id=id, status=status, **kw)


# ---------- Fixtures ----------

_FIELDS = frozenset({"id", "status", "stale", "tags", "env.region", "score"})

_MODELS = [
    _make("w-1", tags=["gpu", "a100"], env={"region": "us-east"}, score=1.0),
    _make("w-2", status="BUSY", tags=["cpu"], stale=True, score=2.0),
    _make("w-3", tags=["gpu"], env={"region": "eu-west"}, score=3.0),
    _make("w-4", status="STOPPED", score=None),
]


# ---------- Tests ----------


def _filter(models: list, query: dict | QueryParams) -> list:
    params = query if isinstance(query, QueryParams) else QueryParams(query)
    return QueryFilter.parse(params, _FIELDS).filter(models)


@pytest.mark.parametrize(
    "query, expected_ids",
    [
        # Exact match
        ({"status": "IDLE"}, {"w-1", "w-3"}),
        ({"status": "BUSY"}, {"w-2"}),
        # No matches
        ({"status": "UNKNOWN"}, set()),
        # Empty query returns all
        ({}, {"w-1", "w-2", "w-3", "w-4"}),
    ],
    ids=["exact", "exact-busy", "no-match", "empty-query"],
)
def test_basic_filtering(query: dict, expected_ids: set[str]) -> None:
    result = _filter(_MODELS, query)
    assert {m.id for m in result} == expected_ids


@pytest.mark.parametrize("key", ["nonexistent", "env", "env.secret", "id.x", "limit"])
def test_undeclared_key_is_rejected(key: str) -> None:
    with pytest.raises(InvalidQuery):
        QueryFilter.parse(QueryParams({key: "val"}), _FIELDS)


def test_the_route_s_own_parameters_are_not_filters() -> None:
    params = QueryParams({"limit": "1", "after": "c", "status": "BUSY"})

    query = QueryFilter.parse(params, _FIELDS, {"limit", "after", "before"})

    assert {m.id for m in query.filter(_MODELS)} == {"w-2"}


def test_repeated_key_or_semantics() -> None:
    """?status=IDLE&status=BUSY should match either."""
    result = _filter(_MODELS, QueryParams("status=IDLE&status=BUSY"))
    assert {m.id for m in result} == {"w-1", "w-2", "w-3"}


def test_nested_models_are_read_by_attribute() -> None:
    class _Cpu(BaseModel):
        model: str

    class _Hardware(BaseModel):
        cpu: _Cpu

    class _Worker(BaseModel):
        id: str
        hardware: _Hardware

    workers = [
        _Worker(id="a", hardware=_Hardware(cpu=_Cpu(model="xeon"))),
        _Worker(id="b", hardware=_Hardware(cpu=_Cpu(model="epyc"))),
    ]
    query = QueryFilter.parse(
        QueryParams({"hardware.cpu.model": "xeon"}), {"hardware.cpu.model"}
    )
    assert [w.id for w in query.filter(workers)] == ["a"]


def test_a_path_through_a_missing_value_reads_as_none() -> None:
    class _Gpu(BaseModel):
        cuda_version: str | None = None

    class _Hardware(BaseModel):
        gpu: _Gpu

    class _Worker(BaseModel):
        id: str
        hardware: _Hardware | None = None

    workers = [
        _Worker(id="gpu", hardware=_Hardware(gpu=_Gpu(cuda_version="12.4"))),
        _Worker(id="cpu", hardware=_Hardware(gpu=_Gpu())),
        _Worker(id="unreported"),
    ]
    fields = {"hardware.gpu.cuda_version"}

    def ids(value: str) -> list[str]:
        params = QueryParams({"hardware.gpu.cuda_version": value})
        return [w.id for w in QueryFilter.parse(params, fields).filter(workers)]

    assert ids("12.4") == ["gpu"]
    assert ids("null") == ["cpu", "unreported"]


def test_many_values_match_in_time_independent_of_their_count() -> None:
    items = [{"task_id": f"tsk-{index}"} for index in range(20_000)]
    params = QueryParams([("task_id", f"tsk-x{index}") for index in range(3000)])
    query = QueryFilter.parse(params, {"task_id"})

    started = time.perf_counter()
    assert query.filter(items) == []
    # A linear scan of the values per item takes seconds here.
    assert time.perf_counter() - started < 0.5


def test_nested_dot_notation() -> None:
    """Dot-notation matches nested dict fields; an absent dict key reads as None."""
    assert {m.id for m in _filter(_MODELS, {"env.region": "us-east"})} == {"w-1"}
    assert {m.id for m in _filter(_MODELS, {"env.region": "null"})} == {"w-2", "w-4"}


def test_list_membership() -> None:
    result = _filter(_MODELS, {"tags": "gpu"})
    assert {m.id for m in result} == {"w-1", "w-3"}


@pytest.mark.parametrize(
    "query_value",
    ["true", "True", "1", "yes", "on"],
    ids=lambda v: f"truthy-{v}",
)
def test_boolean_truthy(query_value: str) -> None:
    result = _filter(_MODELS, {"stale": query_value})
    assert {m.id for m in result} == {"w-2"}


@pytest.mark.parametrize(
    "query_value",
    ["false", "False", "0", "no", "off"],
    ids=lambda v: f"falsy-{v}",
)
def test_boolean_falsy(query_value: str) -> None:
    result = _filter(_MODELS, {"stale": query_value})
    assert {m.id for m in result} == {"w-1", "w-3", "w-4"}


@pytest.mark.parametrize(
    "query_value",
    ["null", "None", ""],
    ids=lambda v: f"null-{v!r}",
)
def test_none_matching(query_value: str) -> None:
    result = _filter(_MODELS, {"score": query_value})
    assert {m.id for m in result} == {"w-4"}


def test_csv_tag_string() -> None:
    """If tags are stored as comma-separated string, membership still works."""

    class _CsvModel(BaseModel):
        id: str
        tags: str

    models = [_CsvModel(id="a", tags="gpu,a100"), _CsvModel(id="b", tags="cpu")]
    result = QueryFilter.parse(QueryParams({"tags": "gpu"}), {"tags"}).filter(models)
    assert [m.id for m in result] == ["a"]
