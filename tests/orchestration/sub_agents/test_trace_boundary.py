# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What the parent records at the boundary of a sub-agent invocation.

The boundary is opened before the child starts and closed by the ``finally``
that releases the invocation's resources, so both the concurrency slots and
the open ``sub_agent.started`` must be taken inside that block — and the
close must state the outcome the controller actually reached, not the one
its result text reads like.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.platform import get_platform
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.envelope import EventDraft
from chrys.foundation.trajectory.event_types import EventType, TraceCoverage
from chrys.foundation.trajectory.writer import EmitResult
from chrys.orchestration.invoker.contracts import SubAgentFailureReason, SubAgentStatus
from chrys.orchestration.invoker.resources import PreparedAgent
from chrys.orchestration.sub_agents.tools import SubAgentTools, _close_sub_agent_trace
from chrys.service.profiles.agents.schema import AcpAgentConfig, AgentProfile
from chrys.service.tools.result_metadata import tool_error, tool_result_metadata
from chrys.service.trajectory.sub_agents import SubAgentStatus as TrajectorySubAgentStatus
from chrys.service.trajectory.sub_agents import SubAgentTrace
from tests.service.trajectory._fakes import FakeSink, make_context


class InterruptedAckSink(FakeSink):
    """Real sink shape whose awaited acknowledgement is interrupted once."""

    def __init__(self) -> None:
        super().__init__()
        self.interrupt_next_ack = True

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        if self.interrupt_next_ack:
            self.interrupt_next_ack = False
            raise asyncio.CancelledError
        return await super().emit(draft, payload_factory=payload_factory)


@pytest.mark.asyncio
async def test_an_interrupt_while_recording_the_start_releases_slot_and_boundary() -> None:
    tools = SubAgentTools(max_total_concurrency=2, event_bus=None, session_id="s-1", session_dir=None)
    tools._display_names["Explore"] = "Explore"
    tools._agent_max["Explore"] = 1
    tools._agent_active["Explore"] = 0
    tools._prepared_by_tool["Explore"] = PreparedAgent()
    tool = tools._make_acp_tool(
        "Explore",
        "Delegate to Explore",
        AgentProfile(name="Explore", acp=AcpAgentConfig(command="agent")),
        SessionEnvironment(cwd="", platform=get_platform()),
    )
    sink = InterruptedAckSink()

    with trajectory_scope(make_context(sink)), pytest.raises(asyncio.CancelledError):
        await tool.func(prompt="delegate")

    # The interrupted start is closed, not left hanging.
    finished = sink.only(EventType.SUB_AGENT_FINISHED)
    assert finished.payload["status"] == TrajectorySubAgentStatus.CANCELLED
    # And the slots it took are back, so the next delegation is not refused.
    assert tools._total_active == 0
    assert tools._agent_active["Explore"] == 0


# ------------------------------------------------------------ boundary close


class _Controller:
    """Just the surface :func:`_close_sub_agent_trace` reads."""

    def __init__(self, status: SubAgentStatus, failure_reason: SubAgentFailureReason | None = None) -> None:
        self.status = status
        self.failure_reason = failure_reason


class _AcpController:
    """The external controller: a status, and no failure reason of its own."""

    def __init__(self, status: SubAgentStatus) -> None:
        self.status = status


def _closed(controller: object | None, *, hook_status: str, failure_code: str | None) -> dict[str, Any]:
    sink = FakeSink()
    trace = SubAgentTrace(
        make_context(sink),
        invocation_id="0123456789ab",
        trace_coverage=TraceCoverage.FULL,
        parent_tool_operation_id=None,
    )
    _close_sub_agent_trace(
        trace,
        hook_status=hook_status,
        failure_code=failure_code,
        controller=controller,  # type: ignore[arg-type]
        failure_reason=controller.failure_reason if isinstance(controller, _Controller) else None,
    )
    return dict(sink.only(EventType.SUB_AGENT_FINISHED).payload)


def test_a_completed_child_is_recorded_ok_even_when_its_answer_reads_like_an_error() -> None:
    payload = _closed(
        _Controller(SubAgentStatus.COMPLETED),
        hook_status="failed",
        failure_code="tool_error",
    )
    assert payload["status"] == TrajectorySubAgentStatus.OK
    assert "failure_reason_code" not in payload


def test_a_completed_child_that_gave_up_is_recorded_failed() -> None:
    """An ACP child that cannot start still reaches COMPLETED, and records the
    failure structurally on its way out — that verdict is the one to report."""
    metadata: dict[str, Any] = {}
    token = tool_result_metadata.set(metadata)
    try:
        tool_error("sub_agent_acp_spawn", "the agent command did not start")
        payload = _closed(
            _AcpController(SubAgentStatus.COMPLETED),
            hook_status="failed",
            failure_code="tool_error",
        )
    finally:
        tool_result_metadata.reset(token)

    assert payload["status"] == TrajectorySubAgentStatus.FAILED
    assert payload["failure_reason_code"] == "tool_error"


def test_an_aborted_child_is_recorded_failed_even_when_its_answer_reads_clean() -> None:
    payload = _closed(_AcpController(SubAgentStatus.ABORTED), hook_status="ok", failure_code=None)
    assert payload["status"] == TrajectorySubAgentStatus.FAILED
    assert payload["failure_reason_code"] == SubAgentStatus.ABORTED.value


def test_the_controller_names_the_reason_it_failed_for() -> None:
    payload = _closed(
        _Controller(SubAgentStatus.ABORTED, SubAgentFailureReason.STREAM_STALL),
        hook_status="failed",
        failure_code="tool_error",
    )
    assert payload["status"] == TrajectorySubAgentStatus.FAILED
    assert payload["failure_reason_code"] == SubAgentFailureReason.STREAM_STALL.value


def test_a_setup_failure_without_a_controller_keeps_its_own_status() -> None:
    payload = _closed(None, hook_status="failed", failure_code="setup_failed")
    assert payload["status"] == TrajectorySubAgentStatus.SETUP_FAILED
    assert payload["failure_reason_code"] == "setup_failed"


def test_a_cancelled_invocation_is_cancelled_whatever_the_controller_reached() -> None:
    payload = _closed(_Controller(SubAgentStatus.COMPLETED), hook_status="cancelled", failure_code=None)
    assert payload["status"] == TrajectorySubAgentStatus.CANCELLED
