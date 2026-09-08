"""A binding's pinned gated mode must resolve to a registered ingress.

The root-local proxy ingress is registered with the gated surface itself. A forward
ingress exists only where a deployment registered one, so a task pinned to ``forward``
without one fails closed rather than being served over the proxy.
"""

from server.serve.ingress import ServeAccessMode, ServeIngressRegistry


def test_the_root_local_proxy_ingress_is_registered_when_permitted() -> None:
    registry = ServeIngressRegistry("serve-edge")
    proxy = registry.live(ServeAccessMode.PROXY)
    assert proxy is not None
    assert proxy.origin_id == "serve-edge"


def test_a_deployment_that_refuses_public_serve_registers_no_proxy_ingress() -> None:
    # The operator switch works by registering nothing, so the ordinary fail-closed
    # path refuses the request rather than a second enforcement point.
    registry = ServeIngressRegistry(None)
    assert registry.live(ServeAccessMode.PROXY) is None


def test_forward_resolves_to_nothing_until_a_deployment_registers_one() -> None:
    registry = ServeIngressRegistry("serve-edge")
    assert registry.live(ServeAccessMode.FORWARD) is None
    registry.register_forward("node-a", generation=3)
    forward = registry.live(ServeAccessMode.FORWARD)
    assert forward is not None
    assert (forward.origin_id, forward.generation) == ("node-a", 3)


def test_a_superseded_forward_registration_does_not_replace_a_newer_one() -> None:
    registry = ServeIngressRegistry("serve-edge")
    registry.register_forward("node-a", generation=5)
    registry.register_forward("node-a", generation=4)
    forward = registry.live(ServeAccessMode.FORWARD)
    assert forward is not None and forward.generation == 5


def test_withdrawing_the_forward_ingress_fails_its_mode_closed_again() -> None:
    registry = ServeIngressRegistry("serve-edge")
    registry.register_forward("node-a", generation=1)
    registry.withdraw_forward()
    assert registry.live(ServeAccessMode.FORWARD) is None
    # Withdrawing forward never disturbs the root-local proxy.
    assert registry.live(ServeAccessMode.PROXY) is not None
