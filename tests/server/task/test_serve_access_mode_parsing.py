"""A serve task pins one gated exposure mode at submission.

``proxy`` and ``forward`` are the two gated HTTP ingress modes over the same binding and
claim path. ``direct`` named a raw, ungated listener and is no longer a mode at all, so
it is refused as ordinary invalid input rather than accepted and reinterpreted.
"""

import textwrap

import pytest

from server.task.parser import parse_workflow
from shared.tasks.specs.serve import ServeSpecTemplate


def _serve_workflow(access_mode: str | None) -> str:
    declared = (
        "" if access_mode is None else f"\n                accessMode: {access_mode}"
    )
    return textwrap.dedent(f"""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: serve-wf
        spec:
          stages:
            - name: serve
              spec:
                taskType: serve{declared}
                resources:
                  hardware:
                    gpu:
                      type: any
                      count: 1
                model:
                  source:
                    type: huggingface
                    identifier: Qwen/Qwen3-0.6B
        """).strip()


def _parsed_mode(access_mode: str | None) -> str | None:
    parsed = parse_workflow(_serve_workflow(access_mode), format="native")
    assert len(parsed.tasks) == 1
    spec = parsed.tasks[0].task.spec
    assert isinstance(spec, ServeSpecTemplate)
    return spec.accessMode


@pytest.mark.parametrize("access_mode", ["proxy", "forward"])
def test_both_gated_modes_parse(
    access_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both gated modes are enabled here; each mode's deployment switch is exercised
    # separately below.
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_FORWARD", True)
    assert _parsed_mode(access_mode) == access_mode


def test_a_serve_task_may_leave_the_mode_unset() -> None:
    assert _parsed_mode(None) is None


def test_direct_is_rejected_as_invalid_input() -> None:
    with pytest.raises(ValueError, match="Invalid task payload"):
        parse_workflow(_serve_workflow("direct"), format="native")


def test_proxy_is_rejected_when_the_deployment_disables_the_proxy_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An operator must be able to refuse public serve exposure at the deployment level.
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_PROXY", False)
    with pytest.raises(ValueError, match="serve accessMode 'proxy' is disabled"):
        parse_workflow(_serve_workflow("proxy"), format="native")


def test_forward_is_not_gated_by_the_proxy_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each gated mode has its own deployment switch: disabling proxy must not reject a
    # forward task when forward is enabled.
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_PROXY", False)
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_FORWARD", True)
    assert _parsed_mode("forward") == "forward"


def test_forward_is_rejected_when_the_deployment_disables_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ENABLE_SERVER_SERVE_FORWARD is a static deployment config known at submission, so
    # a forward task on a forward-disabled server fails fast rather than being adopted
    # and failing every request.
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_FORWARD", False)
    with pytest.raises(ValueError, match="serve accessMode 'forward' is disabled"):
        parse_workflow(_serve_workflow("forward"), format="native")
    # With forward enabled the same task parses.
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_FORWARD", True)
    assert _parsed_mode("forward") == "forward"
