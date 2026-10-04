"""The environment that keeps a local engine's collective traffic on loopback."""


def loopback_collective_env() -> dict[str, str]:
    """Return the variables that bind collective transports to the loopback interface.

    An engine or rank group a worker launches runs on one host, so its NCCL and gloo
    bootstrap and vLLM's distributed init have no peer off the host.
    """
    return {
        "NCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "VLLM_HOST_IP": "127.0.0.1",
    }
