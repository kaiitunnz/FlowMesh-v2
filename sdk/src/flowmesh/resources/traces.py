"""Workflow trace resource — raw rows, the analyzer, and telemetry queries."""

import json
from collections.abc import AsyncIterator, Iterator
from enum import StrEnum
from typing import Literal

import httpx

from .._base_client import (
    _make_url,
    _raise_for_stream_status,
    _raise_for_stream_status_async,
)
from ..exceptions import FlowMeshConnectionError
from ..models.traces import ProfileSummary, TraceAggregate, TraceTree
from ._base import AsyncResource, SyncResource


class TraceType(StrEnum):
    """Trace row type. Members serialize as their values."""

    SPANS = "spans"
    ASSETS = "assets"
    LINEAGE = "lineage"


AggregateStat = Literal["count", "sum", "avg", "min", "max", "p50", "p95", "p99"]
MetricKind = Literal["gauge", "histogram"]


def _aggregate_params(
    metric: str,
    group_by: str,
    stat: AggregateStat,
    kind: MetricKind,
    workflow_id: str | None,
) -> dict[str, str]:
    params = {"metric": metric, "group_by": group_by, "stat": stat, "kind": kind}
    if workflow_id is not None:
        params["workflow_id"] = workflow_id
    return params


class Traces(SyncResource):
    """Synchronous workflow trace operations."""

    def fetch(self, workflow_id: str, trace_type: TraceType) -> Iterator[dict]:
        """Yield JSONL rows for `spans`, `assets`, or `lineage`."""
        url = _make_url(
            self._client.base_url, f"/traces/workflows/{workflow_id}/{trace_type}"
        )
        try:
            with self._client._http.stream("GET", url) as response:
                _raise_for_stream_status(response, "GET")
                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    def analyze(self, workflow_id: str) -> ProfileSummary:
        """Run the trace analyzer and return a parsed `ProfileSummary`."""
        return ProfileSummary.model_validate(
            self._client._request("GET", f"/traces/workflows/analyze/{workflow_id}")
        )

    def tree(self, workflow_id: str) -> TraceTree:
        """Fetch the workflow's spans assembled into their parent/child hierarchy."""
        return TraceTree.model_validate(
            self._client._request("GET", f"/traces/workflows/{workflow_id}/spans/tree")
        )

    def aggregate(
        self,
        metric: str,
        group_by: str,
        stat: AggregateStat = "avg",
        kind: MetricKind = "gauge",
        workflow_id: str | None = None,
    ) -> TraceAggregate:
        """Aggregate one telemetry metric, grouped by one attribute key."""
        return TraceAggregate.model_validate(
            self._client._request(
                "GET",
                "/traces/aggregate",
                params=_aggregate_params(metric, group_by, stat, kind, workflow_id),
            )
        )


class AsyncTraces(AsyncResource):
    """Asynchronous workflow trace operations."""

    async def fetch(
        self, workflow_id: str, trace_type: TraceType
    ) -> AsyncIterator[dict]:
        url = _make_url(
            self._client.base_url, f"/traces/workflows/{workflow_id}/{trace_type}"
        )
        try:
            async with self._client._http.stream("GET", url) as response:
                await _raise_for_stream_status_async(response, "GET")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    async def analyze(self, workflow_id: str) -> ProfileSummary:
        return ProfileSummary.model_validate(
            await self._client._request(
                "GET", f"/traces/workflows/analyze/{workflow_id}"
            )
        )

    async def tree(self, workflow_id: str) -> TraceTree:
        return TraceTree.model_validate(
            await self._client._request(
                "GET", f"/traces/workflows/{workflow_id}/spans/tree"
            )
        )

    async def aggregate(
        self,
        metric: str,
        group_by: str,
        stat: AggregateStat = "avg",
        kind: MetricKind = "gauge",
        workflow_id: str | None = None,
    ) -> TraceAggregate:
        return TraceAggregate.model_validate(
            await self._client._request(
                "GET",
                "/traces/aggregate",
                params=_aggregate_params(metric, group_by, stat, kind, workflow_id),
            )
        )
