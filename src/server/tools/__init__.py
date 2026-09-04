"""Fabric-served external-tool machinery: the broker, carriage, and egress path.

The control plane authorizes and routes a mediated tool boundary (today ``search/v1``)
here; the provider egress itself runs in a worker executor over ``shared.tools``.
"""
