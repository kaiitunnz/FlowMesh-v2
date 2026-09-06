"""The FabricToolBroker — control-plane policy and correlation for fabric-served tools.

The broker applies policy and correlation for a mediated tool boundary the control plane
captured server-side (today the agent-model gateway's captured ``search/v1`` facade).
External-tool egress runs only in a worker's mediated-egress sidecar, reached by the
worker-originated path; a boundary that reaches the broker has no worker origin and no
in-server egress, so the broker terminalizes it durably off the agent's lane as a typed
unavailable outcome.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from shared.outcome import InlineControl, OutcomeCarrier
from shared.tools.contract import ToolOutcome, ToolOutcomeStatus

from ..config import WebSearchConfig
from ..orchestration.tool_dispatch import ToolInvocationEnvelope

# (task_id, call_correlation, carrier) — a typed inline control datum the runtime
# settles durably.
SettleCallback = Callable[[str, str, OutcomeCarrier], None]


def inline_outcome(outcome: ToolOutcome) -> InlineControl:
    """Wrap a server-produced typed outcome as an opaque inline control datum."""
    return InlineControl(value=outcome.model_dump_json())


class FabricToolBroker:
    """Terminalize a server-captured tool boundary off the agent's lane."""

    def __init__(
        self,
        config: WebSearchConfig,
        settle: SettleCallback,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cfg = config
        self._settle = settle
        self._log = logger or logging.getLogger("fabric-tool-broker")
        self._pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="fabric-tool-broker"
        )

    @classmethod
    def build(
        cls,
        config: WebSearchConfig,
        settle: SettleCallback,
        logger: logging.Logger | None = None,
    ) -> "FabricToolBroker":
        return cls(config, settle, logger)

    def submit(self, env: ToolInvocationEnvelope) -> None:
        """Accept a server-captured tool boundary and terminalize it off the lane."""
        self._pool.submit(self._run, env)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)

    def _run(self, env: ToolInvocationEnvelope) -> None:
        self._log.info(
            "fabric tool boundary has no in-server egress interface=%s inv=%s",
            env.interface,
            env.invocation_id,
        )
        outcome = ToolOutcome(
            status=ToolOutcomeStatus.UNAVAILABLE,
            value=f"the {env.interface} tool has no in-server egress",
        )
        self._settle(env.task_id, env.call_correlation, inline_outcome(outcome))
