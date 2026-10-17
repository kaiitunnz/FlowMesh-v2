"""Certification: a scope is certified only while every coverage condition holds."""

import fakeredis
import pytest

from server.clients.redis import (
    RECLAIM_BACKFILL_FAULTS_KEY,
    RECLAIM_HOOKS_ON_KEY,
    reclaim_cert_key,
    reclaim_uncertified_key,
    reclaim_unexpanded_key,
)
from server.reclamation.backfill import FAMILIES
from server.reclamation.certification import (
    ALL_SCOPES,
    CertificationStatus,
    CoverageStore,
)
from tests.server.redis_helpers import fake_redis_client

_SCOPE = "org-a"


@pytest.fixture
def setup() -> tuple[CoverageStore, fakeredis.FakeRedis]:
    server = fakeredis.FakeServer()
    return (
        CoverageStore(fake_redis_client(server), FAMILIES),
        fakeredis.FakeRedis(server=server, decode_responses=True),
    )


def _complete(coverage: CoverageStore, generation: int) -> None:
    for family in FAMILIES:
        assert coverage.complete(generation, family)


def test_the_marker_only_rises(setup) -> None:
    coverage, raw = setup
    assert coverage.mark_hooks_on(3) == 3
    assert coverage.mark_hooks_on(2) == 3
    assert raw.get(RECLAIM_HOOKS_ON_KEY) == "3"
    assert coverage.mark_hooks_on(5) == 5


def test_a_scope_is_certified_once_every_family_completed_in_its_generation(
    setup,
) -> None:
    coverage, raw = setup
    generation = coverage.mark_hooks_on(4)
    _complete(coverage, generation)

    certification = coverage.certify(_SCOPE)

    assert certification.certified
    assert raw.get(reclaim_cert_key(_SCOPE)) == "4"


def test_no_marker_refuses(setup) -> None:
    coverage, _ = setup
    assert coverage.certify(_SCOPE).status is CertificationStatus.NO_MARKER


def test_an_incomplete_family_refuses(setup) -> None:
    coverage, _ = setup
    generation = coverage.mark_hooks_on(1)
    for family in FAMILIES[:-1]:
        coverage.complete(generation, family)

    certification = coverage.certify(_SCOPE)

    assert certification.status is CertificationStatus.INCOMPLETE
    assert certification.detail == FAMILIES[-1]


def test_completion_in_an_earlier_generation_does_not_count(setup) -> None:
    coverage, raw = setup
    _complete(coverage, coverage.mark_hooks_on(1))
    assert coverage.certify(_SCOPE).certified
    coverage.mark_hooks_on(2)

    assert coverage.certify(_SCOPE).status is CertificationStatus.INCOMPLETE
    assert not raw.exists(reclaim_cert_key(_SCOPE))


def test_a_later_root_takes_completion_from_an_earlier_one(setup) -> None:
    coverage, _ = setup
    coverage.mark_hooks_on(1)
    coverage.mark_hooks_on(2)
    assert not coverage.complete(1, FAMILIES[0])


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (_SCOPE, CertificationStatus.UNDECODABLE),
        ("org-b", CertificationStatus.CERTIFIED),
    ],
)
def test_a_fault_withholds_its_own_scope(setup, scope, expected) -> None:
    coverage, _ = setup
    generation = coverage.mark_hooks_on(1)
    _complete(coverage, generation)
    coverage.record_fault(generation, _SCOPE)

    assert coverage.certify(scope).status is expected


def test_an_unattributed_fault_withholds_every_scope(setup) -> None:
    coverage, raw = setup
    generation = coverage.mark_hooks_on(1)
    _complete(coverage, generation)
    coverage.record_fault(generation, None)

    assert raw.hget(RECLAIM_BACKFILL_FAULTS_KEY, ALL_SCOPES) == "1"
    assert coverage.certify("anything").status is CertificationStatus.UNDECODABLE


def test_a_fault_of_an_earlier_generation_does_not_count(setup) -> None:
    coverage, _ = setup
    coverage.record_fault(1, None)
    _complete(coverage, coverage.mark_hooks_on(2))

    assert coverage.certify(_SCOPE).certified


def test_unexpanded_trees_refuse_and_withdraw_a_certification(setup) -> None:
    coverage, raw = setup
    _complete(coverage, coverage.mark_hooks_on(1))
    assert coverage.certify(_SCOPE).certified
    raw.sadd(reclaim_unexpanded_key(_SCOPE), "d1", "d2")

    certification = coverage.certify(_SCOPE)

    assert certification.status is CertificationStatus.UNEXPANDED
    assert certification.detail == "2"
    assert not raw.exists(reclaim_cert_key(_SCOPE))


def test_an_owed_protection_refuses(setup) -> None:
    coverage, raw = setup
    _complete(coverage, coverage.mark_hooks_on(1))
    raw.set(reclaim_uncertified_key(_SCOPE), "1")

    assert coverage.certify(_SCOPE).status is CertificationStatus.OWED


def test_a_coverage_key_of_the_wrong_type_refuses(setup) -> None:
    coverage, raw = setup
    _complete(coverage, coverage.mark_hooks_on(1))
    raw.set(reclaim_unexpanded_key(_SCOPE), "x")

    assert coverage.certify(_SCOPE).status is CertificationStatus.WRONGTYPE
