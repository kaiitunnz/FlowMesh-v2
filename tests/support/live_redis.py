"""Marks for tests that write to the live Redis ``FLOWMESH_TEST_REDIS_URL`` names."""

import pytest

# They share that Redis's keyspace, so ``--dist loadgroup`` runs them on one worker.
LIVE_REDIS = pytest.mark.xdist_group("live-redis")
