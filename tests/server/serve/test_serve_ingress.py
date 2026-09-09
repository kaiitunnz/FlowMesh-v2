"""The root-local proxy ingress registration.

The proxy ingress is registered with the gated surface itself; an operator that refuses
public proxy serve exposure registers none and every proxy request fails closed. Forward
exposure is not held here — each forward binding owns a port exposure in the
``ForwardIngressDirectory`` (see ``test_forward_exposure``).
"""

from server.serve.ingress import ServeAccessMode, ServeIngressRegistry


def test_the_root_local_proxy_ingress_is_registered_when_permitted() -> None:
    registry = ServeIngressRegistry("serve-edge")
    proxy = registry.live(ServeAccessMode.PROXY)
    assert proxy is not None
    assert proxy.origin_id == "serve-edge"


def test_a_deployment_that_refuses_public_serve_registers_no_proxy_ingress() -> None:
    # The operator switch works by registering nothing, so the ordinary fail-closed path
    # refuses the request rather than a second enforcement point.
    registry = ServeIngressRegistry(None)
    assert registry.live(ServeAccessMode.PROXY) is None


def test_the_registry_holds_no_forward_ingress() -> None:
    # Forward exposure lives in the directory, not this registry, so the registry never
    # resolves a forward ingress.
    registry = ServeIngressRegistry("serve-edge")
    assert registry.live(ServeAccessMode.FORWARD) is None
