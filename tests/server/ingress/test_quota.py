"""The per-principal quota bounds a principal's in-flight ingress requests.

The limit is a pre-admission gate: exceeding it raises before any claim, and releasing a
slot lets the next request proceed, so one principal cannot exhaust admission.
"""

import pytest

from server.ingress import PrincipalQuota, QuotaExceeded


def test_quota_bounds_concurrent_requests_per_principal():
    quota = PrincipalQuota(max_concurrent=2)
    quota.acquire("p1")
    quota.acquire("p1")
    with pytest.raises(QuotaExceeded):
        quota.acquire("p1")
    # A different principal has its own budget.
    quota.acquire("p2")


def test_release_frees_a_slot_for_the_next_request():
    quota = PrincipalQuota(max_concurrent=1)
    quota.acquire("p1")
    quota.release("p1")
    quota.acquire("p1")  # the freed slot admits the next request


def test_release_below_zero_is_safe():
    quota = PrincipalQuota(max_concurrent=1)
    quota.release("p1")  # no slot held; a no-op
    quota.acquire("p1")
