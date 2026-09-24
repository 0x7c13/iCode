# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Approval observer hooks follow the actual wait and cannot decide approvals."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalRequest, ApprovalResponse
from chrys.foundation.tool_kinds import KIND_MCP, set_tool_kind
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.metadata import OPERATION_ID_KEY
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.kernel.tools import FunctionTool
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.agent_middleware.events.hook_dispatch import set_call_id
from chrys.service.approval.judge import ApprovalJudge, JudgeVerdict
from chrys.service.approval.policy import ApprovalMode, ApprovalPolicy
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.schema import HookDecision
from chrys.service.profiles.agents.schema import ApprovalConfig
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.waiting import wait_for


@pytest.mark.parametrize(
    ("mode", "judge_approved", "human_approved", "expects_wait"),
    [
        (ApprovalMode.MANUAL, None, True, True),
        (ApprovalMode.MANUAL, None, False, True),
        (ApprovalMode.AUTO, False, True, True),
        (ApprovalMode.AUTO, True, True, False),
        (ApprovalMode.BYPASS, None, True, False),
    ],
)
async def test_approval_hooks_follow_human_wait_before_tool_execution(
    mode: ApprovalMode,
    judge_approved: bool | None,
    human_approved: bool,
    expects_wait: bool,
) -> None:
    bus = EventBus()
    hooks = create_autospec(HookManager, instance=True)
    hooks.has_hooks_for.side_effect = lambda event: (
        event
        in {
            HookEvent.APPROVAL_REQUESTED,
            HookEvent.APPROVAL_RESOLVED,
        }
    )
    notified = asyncio.Event()
    observed: list[tuple[HookEvent, dict[str, Any]]] = []
    order: list[str] = []

    async def observe(
        event: HookEvent, payload: dict[str, Any], *, target_operation_id: str | None = None
    ) -> HookDecision:
        observed.append((event, payload))
        order.append(str(event))
        if event == HookEvent.APPROVAL_REQUESTED:
            notified.set()
        # Even a hostile observer cannot change the user's decision.
        return HookDecision(blocked=True, block_reason="observer must not gate", args_override={"value": "changed"})

    hooks.fire.side_effect = observe
    judge = None
    if judge_approved is not None:
        judge = create_autospec(ApprovalJudge, instance=True)
        judge.evaluate.return_value = JudgeVerdict(approved=judge_approved, reason="test verdict")
    function = FunctionTool(name="change_test_value")
    set_tool_kind(function, KIND_MCP)
    context = FunctionInvocationContext(function, {"value": "original"})
    context.metadata[OPERATION_ID_KEY] = "b" * 32
    set_call_id(context, "call-test")
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default="require"), tools=[function]),
        bus,
        approval_mode=mode,
        approval_judge=judge,
        hook_manager=hooks,
        session_id="session-test",
        profile_name="Code",
        workspace_cwd="/workspace",
    )

    async def execute() -> None:
        order.append("execute")

    sink = FakeSink()
    with trajectory_scope(make_context(sink)):
        task = asyncio.create_task(middleware.process(context, execute))
    try:
        if expects_wait:
            await wait_for(lambda: notified.is_set() or task.done())
            if task.done():
                await task
            assert notified.is_set()
            assert order == ["approval_requested"]
            assert not task.done()
            payload = observed[0][1]
            assert payload["tool"] == {
                "name": "change_test_value",
                "kind": "mcp",
                "call_id": "call-test",
                "args": {"value": "original"},
            }
            assert payload["session_id"] == "session-test"
            assert payload["profile"] == "Code"
            await bus.publish(ApprovalResponse(request_id=payload["request_id"], approved=human_approved))
        await wait_for(task.done)
        await task
        assert context.arguments == {"value": "original"}
        if expects_wait:
            assert observed[1][0] == HookEvent.APPROVAL_RESOLVED
            assert observed[1][1]["approved"] is human_approved
            assert observed[1][1]["request_id"] == observed[0][1]["request_id"]
            interval_id = sink.only(EventType.APPROVAL_REQUESTED).payload["approval_request_id"]
            assert interval_id != context.metadata[OPERATION_ID_KEY]
            assert all(call.kwargs["target_operation_id"] == interval_id for call in hooks.fire.await_args_list)
            assert order == ["approval_requested", "approval_resolved", *(["execute"] if human_approved else [])]
        else:
            assert observed == []
            assert order == ["execute"]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await middleware.close()


async def test_cancelled_approval_hook_wait_releases_response_subscription() -> None:
    bus = EventBus()
    notified = asyncio.Event()
    hooks = create_autospec(HookManager, instance=True)
    hooks.has_hooks_for.return_value = True

    async def observe(
        event: HookEvent, payload: dict[str, Any], *, target_operation_id: str | None = None
    ) -> HookDecision:
        notified.set()
        return HookDecision()

    hooks.fire.side_effect = observe
    function = FunctionTool(name="change_test_value")
    set_tool_kind(function, KIND_MCP)
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default="require"), tools=[function]),
        bus,
        hook_manager=hooks,
    )

    async def execute() -> None:
        pytest.fail("Cancelled approval must not execute the tool")

    task = asyncio.create_task(middleware.process(FunctionInvocationContext(function, {}), execute))
    try:
        await wait_for(lambda: notified.is_set() or task.done())
        if task.done():
            await task
        assert notified.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not bus._handlers.get(ApprovalResponse)
        assert hooks.fire.await_count == 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await middleware.close()


async def test_synchronous_approval_response_does_not_claim_a_human_wait() -> None:
    bus = EventBus()
    hooks = create_autospec(HookManager, instance=True)
    hooks.has_hooks_for.return_value = True

    async def approve(event: ApprovalRequest) -> None:
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True))

    await bus.subscribe(ApprovalRequest, approve)
    function = FunctionTool(name="change_test_value")
    set_tool_kind(function, KIND_MCP)
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default="require"), tools=[function]),
        bus,
        hook_manager=hooks,
    )
    executed = False

    async def execute() -> None:
        nonlocal executed
        executed = True

    try:
        await middleware.process(FunctionInvocationContext(function, {}), execute)
        assert executed
        hooks.fire.assert_not_awaited()
    finally:
        await bus.unsubscribe(ApprovalRequest, approve)
        await middleware.close()
