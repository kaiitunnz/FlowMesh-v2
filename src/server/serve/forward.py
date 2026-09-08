"""The transport for a worker-hosted forward serve ingress.

A forward-pinned serve request arrives on a worker's public ingress, which relays it to
control for admission. Control drives the same two-phase claim as the root-local proxy,
but the ingress worker — not the root — owns the origin relay to the replica sidecar and
tees the response to its own client. This transport is control's side of that: it relays
the admission decision and the route authorization down to the ingress worker over its
authenticated attachment, and tears the request's ingress rendezvous down on reap.

It never carries a response frame: the ingress tees the engine's frames to its client
data-direct, and only the sidecar's attested acknowledgement and fenced terminal ride
back up to control, which owns the credit.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.envelope import ServeRequestEnvelope

# Relays one control frame to a worker over its attachment; False if the worker is gone.
WorkerRelay = Callable[[str, str, dict[str, object]], bool]


@dataclass
class _ForwardEntry:
    """One forward request's routing: the ingress worker and the rendezvous key."""

    worker_id: str
    request_id: str
    invocation_id: str
    session_id: str | None = None


class ServeForwardTransport:
    """Relays admission, authorization, and reap to a forward ingress worker."""

    def __init__(
        self, relay: WorkerRelay, logger: logging.Logger | None = None
    ) -> None:
        self._relay = relay
        self._logger = logger or logging.getLogger("serve-forward-transport")
        self._by_invocation: dict[str, _ForwardEntry] = {}
        self._by_session: dict[str, _ForwardEntry] = {}

    def track(self, *, invocation_id: str, worker_id: str, request_id: str) -> None:
        """Record which ingress worker and request one invocation relays to.

        Called at submission before admission runs, so the decision relayed down when
        the handoff is minted resolves the worker and the ingress rendezvous key.
        """
        self._by_invocation[invocation_id] = _ForwardEntry(
            worker_id=worker_id, request_id=request_id, invocation_id=invocation_id
        )

    def open(
        self,
        *,
        session_id: str,
        invocation_id: str,
        idm: str,
        task_id: str,
        call_correlation: str,
        handoff: AdmissionHandoff,
        envelope: ServeRequestEnvelope,
    ) -> None:
        """Relay the admission decision down so the ingress begins its own drive.

        The envelope is not relayed: the ingress already holds the frozen request and
        drives the sidecar data-direct.
        """
        entry = self._by_invocation.get(invocation_id)
        if entry is None:
            return
        entry.session_id = session_id
        self._by_session[session_id] = entry
        self._relay(
            entry.worker_id,
            "serve_ingress_admitted",
            {
                "request_id": entry.request_id,
                "session_id": session_id,
                "task_id": task_id,
                "call_correlation": call_correlation,
                "handoff": handoff.model_dump(mode="json"),
            },
        )

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None:
        """Relay the post-acceptance route authorization down to the ingress."""
        entry = self._by_session.get(session_id)
        if entry is None:
            return
        self._relay(
            entry.worker_id,
            "serve_ingress_authorized",
            {
                "request_id": entry.request_id,
                "session_id": session_id,
                "auth": auth.model_dump(mode="json"),
            },
        )

    def close(self, session_id: str) -> None:
        """Reap the ingress drive and its rendezvous entry on a fenced terminal."""
        entry = self._by_session.pop(session_id, None)
        if entry is None:
            return
        self._by_invocation.pop(entry.invocation_id, None)
        self._relay(
            entry.worker_id,
            "serve_ingress_reaped",
            {
                "request_id": entry.request_id,
                "session_id": session_id,
                "invocation_id": entry.invocation_id,
            },
        )

    def deny(self, invocation_id: str, status: int, detail: str) -> None:
        """Refuse a tracked request that failed before its drive opened.

        A synchronous rejection or an admission that gave up before the handoff was
        minted never opened a session, so its ingress rendezvous is still waiting: relay
        a denial down so the waiting request is refused promptly rather than at its
        admission timeout, and drop the tracking entry.
        """
        entry = self._by_invocation.pop(invocation_id, None)
        if entry is None or entry.session_id is not None:
            return
        self._relay(
            entry.worker_id,
            "serve_ingress_denied",
            {
                "request_id": entry.request_id,
                "status": status,
                "detail": detail,
            },
        )

    def forget(self, invocation_id: str) -> None:
        """Drop a settled request's tracking, denying it if it never opened."""
        entry = self._by_invocation.get(invocation_id)
        if entry is not None and entry.session_id is None:
            self.deny(invocation_id, 502, "serve request could not be admitted")
        else:
            self._by_invocation.pop(invocation_id, None)
