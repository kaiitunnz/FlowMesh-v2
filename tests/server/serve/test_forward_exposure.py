"""The root forward ingress directory allocates per-task ports and fences commit.

The root exposes one public authority and port range; each forward binding reserves a
port, the root binds a listener and returns evidence, and only then does the exposure go
LIVE and its url resolve. A drained exposure retires and quarantines its port so a
reused number never carries a stale generation's traffic.
"""

from server.serve.forward_exposure import (
    ForwardExposureStatus,
    ForwardIngressDirectory,
)


def _dir(authority="serve.example", low=34000, high=34001) -> ForwardIngressDirectory:
    return ForwardIngressDirectory(authority, low, high)


def _reserve(d: ForwardIngressDirectory, task="tsk-1"):
    return d.reserve(
        serve_task_id=task,
        binding_generation=0,
        requested_port=None,
    )


def test_reserve_fails_closed_without_a_configured_authority() -> None:
    d = ForwardIngressDirectory("", 0, 0)
    assert not d.configured
    assert _reserve(d) is None


def test_reserve_allocates_from_the_range_and_publishes_a_port_url() -> None:
    d = _dir()
    exposure = _reserve(d)
    assert exposure is not None
    assert 34000 <= exposure.public_port <= 34001
    assert exposure.status is ForwardExposureStatus.RESERVED
    # The deployment terminates TLS ahead of the root, so the root url is plain http.
    assert exposure.public_url == f"http://serve.example:{exposure.public_port}"


def test_commit_goes_live_only_from_the_matching_reservation() -> None:
    d = _dir()
    exposure = _reserve(d)
    assert exposure is not None
    d.mark_binding("tsk-1", exposure.exposure_generation)
    # A commit for a stale reservation generation is refused.
    assert (
        d.commit(
            serve_task_id="tsk-1",
            exposure_generation=exposure.exposure_generation + 1,
            listener_generation=1,
        )
        is None
    )
    live = d.commit(
        serve_task_id="tsk-1",
        exposure_generation=exposure.exposure_generation,
        listener_generation=1,
    )
    assert live is not None and live.status is ForwardExposureStatus.LIVE
    assert live.listener_generation == 1
    assert d.live("tsk-1") is not None


def test_drain_stops_live_resolution_and_retire_quarantines_the_port() -> None:
    d = _dir()
    first = _reserve(d)
    assert first is not None
    d.commit(
        serve_task_id="tsk-1",
        exposure_generation=first.exposure_generation,
        listener_generation=1,
    )
    d.drain("tsk-1")
    assert d.live("tsk-1") is None
    retired = d.retire("tsk-1")
    assert retired is not None and retired.status is ForwardExposureStatus.RETIRED

    # The retired port is quarantined: a fresh reservation takes the other range port.
    second = _reserve(d, task="tsk-2")
    assert second is not None and second.public_port != first.public_port


def test_a_persisted_live_exposure_rebinds_its_same_port_on_restart() -> None:
    d = _dir()
    exposure = _reserve(d)
    assert exposure is not None
    d.commit(
        serve_task_id="tsk-1",
        exposure_generation=exposure.exposure_generation,
        listener_generation=1,
    )
    # A fresh directory loads the persisted snapshot, as a restart does.
    restored = _dir()
    restored.load_snapshot(d.to_snapshot())
    loaded = restored.current("tsk-1")
    assert loaded is not None and loaded.public_port == exposure.public_port

    # Until the listener rebinds, the exposure is treated as binding, not live.
    restored.mark_rebinding("tsk-1")
    assert restored.live("tsk-1") is None
    # A rebind recommits the same port under a fresh listener generation.
    live = restored.commit(
        serve_task_id="tsk-1",
        exposure_generation=loaded.exposure_generation,
        listener_generation=2,
    )
    assert live is not None and live.public_port == exposure.public_port
    assert live.listener_generation == 2


def test_the_range_can_be_exhausted() -> None:
    d = _dir()
    assert _reserve(d, task="tsk-1") is not None
    assert _reserve(d, task="tsk-2") is not None
    # Both ports in the two-wide range are in use.
    assert _reserve(d, task="tsk-3") is None
