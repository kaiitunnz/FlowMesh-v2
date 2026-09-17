"""The fleet/residency sampler: on-loop gauges that join a span by service_family."""

import asyncio
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource

from server.resident import (
    ClaimCredit,
    ReplicaIncarnation,
    ReplicaState,
    ResidentStores,
    ServiceFamily,
    new_claim,
    reserve,
)
from server.services.fleet_metrics import build_fleet_sampler
from shared.telemetry.semconv import PHYSICAL_SERVICE_FAMILY, RESOURCE_NODE_ID


def _meter_with_reader() -> tuple[InMemoryMetricReader, object]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(resource=Resource.create({}), metric_readers=[reader])
    return reader, provider.get_meter("test")


def _stores_with_one_family() -> ResidentStores:
    stores = ResidentStores()
    stores.families.register(
        ServiceFamily(family="fam", engine_batch_key="k", model_ref="m")
    )
    stores.directory.add(
        ReplicaIncarnation(
            replica_id="rpl-1",
            family="fam",
            incarnation=1,
            state=ReplicaState.WARM,
            healthy=True,
        )
    )
    claim = new_claim(invocation_id="inv-1", family="fam")
    reserve(claim, replica_id="rpl-1", incarnation=1, credit=ClaimCredit(slots=2))
    stores.claims.add(claim)
    return stores


def _datapoints(reader: InMemoryMetricReader) -> dict[str, list[tuple[float, dict]]]:
    out: dict[str, list[tuple[float, dict]]] = {}
    data = reader.get_metrics_data()
    if data is None:
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out.setdefault(metric.name, [])
                for dp in metric.data.data_points:
                    out[metric.name].append((dp.value, dict(dp.attributes)))
    return out


def _fake_runtime(queue_len: int) -> object:
    runtime = MagicMock()
    runtime.ready_queue_length.return_value = queue_len
    return runtime


def test_sample_once_emits_per_family_gauges_with_service_family_and_node_id():
    reader, meter = _meter_with_reader()
    stores = _stores_with_one_family()
    sampler = build_fleet_sampler(
        meter,
        stores=stores,
        runtime=_fake_runtime(3),
        node_id=lambda: "nde-1",
        interval_sec=1.0,
        enabled=True,
    )

    sampler._sample_once()

    points = _datapoints(reader)
    replica_value, replica_attrs = points["flowmesh.resident.replica_count"][0]
    assert replica_value == 1
    assert replica_attrs[PHYSICAL_SERVICE_FAMILY] == "fam"
    assert replica_attrs[RESOURCE_NODE_ID] == "nde-1"

    slots_value, slots_attrs = points["flowmesh.resident.admission_slots_in_use"][0]
    assert slots_value == 2
    assert slots_attrs[PHYSICAL_SERVICE_FAMILY] == "fam"

    credit_value, credit_attrs = points["flowmesh.resident.claim_credit_held"][0]
    assert credit_value == 1
    assert credit_attrs[PHYSICAL_SERVICE_FAMILY] == "fam"

    queue_value, queue_attrs = points["flowmesh.resident.queue_depth"][0]
    assert queue_value == 3
    assert queue_attrs[RESOURCE_NODE_ID] == "nde-1"


def test_sample_once_with_no_stores_emits_queue_depth_only():
    reader, meter = _meter_with_reader()
    sampler = build_fleet_sampler(
        meter,
        stores=None,
        runtime=_fake_runtime(5),
        node_id=lambda: "nde-1",
        interval_sec=1.0,
        enabled=True,
    )

    sampler._sample_once()

    points = _datapoints(reader)
    assert "flowmesh.resident.replica_count" not in points
    assert points["flowmesh.resident.queue_depth"][0][0] == 5


def test_disabled_sampler_never_starts_a_task():
    _, meter = _meter_with_reader()
    sampler = build_fleet_sampler(
        meter,
        stores=None,
        runtime=_fake_runtime(0),
        node_id=lambda: None,
        interval_sec=1.0,
        enabled=False,
    )

    sampler.start()

    assert sampler.is_running is False


@pytest.mark.anyio
async def test_enabled_sampler_starts_and_stops_a_task_on_the_loop():
    _, meter = _meter_with_reader()
    sampler = build_fleet_sampler(
        meter,
        stores=None,
        runtime=_fake_runtime(0),
        node_id=lambda: None,
        interval_sec=60.0,
        enabled=True,
    )

    sampler.start(asyncio.get_event_loop())
    assert sampler.is_running is True

    sampler.shutdown()
    assert sampler.is_running is False
