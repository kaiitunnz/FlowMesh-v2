"""The forward ingress directory allocates per-task ports and fences two-phase commit.

A deployment registers ingress hosts with a public authority and port range; each
forward binding reserves a port, its worker binds and returns evidence, and only then
does the exposure go LIVE and its url resolve. A drained exposure retires and
quarantines its port so a reused number never carries a stale generation's traffic.
"""

from server.serve.forward_exposure import (
    ForwardExposureStatus,
    ForwardIngressDirectory,
    ForwardIngressHost,
)


def _host(**over: object) -> ForwardIngressHost:
    base: dict[str, object] = dict(
        authority="serve.example",
        worker_id="wrk-a",
        origin_id="rog-a",
        port_low=34000,
        port_high=34001,
        tls_profile_generation=0,
        generation=1,
    )
    base.update(over)
    return ForwardIngressHost(**base)  # type: ignore[arg-type]


def _reserve(d: ForwardIngressDirectory, task="tsk-1", require_tls=False):
    return d.reserve(
        serve_task_id=task,
        binding_generation=0,
        requested_port=None,
        require_tls=require_tls,
    )


def test_reserve_fails_closed_without_a_registered_host() -> None:
    d = ForwardIngressDirectory()
    assert _reserve(d) is None


def test_reserve_allocates_from_the_host_range_and_publishes_a_port_url() -> None:
    d = ForwardIngressDirectory()
    d.register_host(_host())
    exposure = _reserve(d)
    assert exposure is not None
    assert 34000 <= exposure.public_port <= 34001
    assert exposure.status is ForwardExposureStatus.RESERVED
    assert exposure.public_url == f"http://serve.example:{exposure.public_port}"


def test_a_tls_requirement_refuses_a_plaintext_host() -> None:
    d = ForwardIngressDirectory()
    d.register_host(_host(tls_profile_generation=0))
    assert _reserve(d, require_tls=True) is None
    d.register_host(_host(tls_profile_generation=5, generation=2))
    exposure = _reserve(d, require_tls=True)
    assert exposure is not None and exposure.tls
    assert exposure.public_url.startswith("https://")


def test_commit_goes_live_only_from_the_matching_reservation() -> None:
    d = ForwardIngressDirectory()
    d.register_host(_host())
    exposure = _reserve(d)
    assert exposure is not None
    d.mark_binding("tsk-1", exposure.exposure_generation)
    # A commit for a stale reservation generation is refused.
    assert (
        d.commit(
            serve_task_id="tsk-1",
            exposure_generation=exposure.exposure_generation + 1,
            listener_generation=1,
            attachment_generation=1,
        )
        is None
    )
    live = d.commit(
        serve_task_id="tsk-1",
        exposure_generation=exposure.exposure_generation,
        listener_generation=1,
        attachment_generation=1,
    )
    assert live is not None and live.status is ForwardExposureStatus.LIVE
    assert d.live("tsk-1") is not None


def test_drain_stops_live_resolution_and_retire_quarantines_the_port() -> None:
    d = ForwardIngressDirectory()
    d.register_host(_host())
    first = _reserve(d)
    assert first is not None
    d.commit(
        serve_task_id="tsk-1",
        exposure_generation=first.exposure_generation,
        listener_generation=1,
        attachment_generation=1,
    )
    d.drain("tsk-1")
    assert d.live("tsk-1") is None
    retired = d.retire("tsk-1")
    assert retired is not None and retired.status is ForwardExposureStatus.RETIRED

    # The retired port is quarantined: a fresh reservation takes the other range port.
    d.register_host(_host())
    second = _reserve(d, task="tsk-2")
    assert second is not None and second.public_port != first.public_port


def test_the_range_can_be_exhausted() -> None:
    d = ForwardIngressDirectory()
    d.register_host(_host())
    assert _reserve(d, task="tsk-1") is not None
    assert _reserve(d, task="tsk-2") is not None
    # Both ports in the two-wide range are in use.
    assert _reserve(d, task="tsk-3") is None
