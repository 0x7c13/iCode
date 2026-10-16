# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACP risk checks, empty-context review, and human-priority audit."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import create_autospec

import pytest
from acp.schema import PermissionOption, ToolCallUpdate

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalRequest, ApprovalResponse, ApprovalReviewed
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import ChatResponse, Message
from chrys.orchestration.invoker.acp_protocol import AcpPermissionBroker, AcpUpdateTranslator
from chrys.service.approval import judge as judge_module
from chrys.service.approval.policy import ApprovalMode
from chrys.service.approval.turn_context import TurnContextHolder
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support import formal_approval as core
from tests.support.event_capture import capture_event_sequence
from tests.support.waiting import wait_for


@pytest.mark.parametrize("kind", [None, "other"], ids=["missing-kind", "unmapped-kind"])
@pytest.mark.parametrize("triggered", [False, True])
@pytest.mark.parametrize("human_approved", [False, True])
async def test_missing_acp_kind_is_reviewed_by_the_predicate_model(monkeypatch, kind, triggered, human_approved):
    bus = EventBus()
    sink = FakeSink()
    context = TurnContextHolder()
    context.replace(["Inspect source"])
    values = core.predicate_values() | {"external_action": triggered}
    async with core.judge_client(monkeypatch, [json.dumps(values)]) as (judge, calls):
        broker = AcpPermissionBroker(
            event_bus=bus,
            session_id="parent",
            caller_name="External",
            mode_getter=lambda: ApprovalMode.AUTO,
            turn_context=context,
            workspace_roots=["/workspace"],
            workspace_cwd="/workspace",
            approval_judge=judge,
            ask_user_timeout_seconds=None,
            trajectory_context=make_context(sink),
            trajectory_boundary_operation_id=new_analytics_id(),
        )

        async def human_decline(event: ApprovalReviewed):
            if triggered:
                assert event.approved is False
                assert "external_action" in event.reason
                await bus.publish(
                    ApprovalResponse(request_id=event.request_id, approved=human_approved, session_id="parent")
                )

        await bus.subscribe(ApprovalReviewed, human_decline)
        task = asyncio.create_task(
            broker.on_permission_request(
                ToolCallUpdate(toolCallId="call", title="Inspect source", kind=kind, rawInput={"path": "main.py"}),
                [
                    PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
                    PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
                ],
            )
        )
        try:
            await wait_for(task.done, description="ACP permission resolved")
            decision = await task
            assert len(calls) == 1
            assert decision.action == ("deny" if triggered and not human_approved else "allow")
            resolved = sink.only(EventType.APPROVAL_RESOLVED).payload
            assert resolved["decider"] == ("user" if triggered else "judge")
            assert resolved["formal_judge"]["approved"] is (not triggered)
            assert resolved["formal_judge"]["predicate_results"] == [
                {"id": key, "value": value} for key, value in values.items()
            ]
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await broker.close()
            await bus.unsubscribe(ApprovalReviewed, human_decline)


@pytest.mark.parametrize("kind", [None, "other", "read"], ids=["missing-kind", "unmapped-kind", "read-kind"])
@pytest.mark.parametrize("messages", [[], [" ", "\t"]], ids=["empty-context", "blank-context"])
@pytest.mark.parametrize(
    "triggered,direct_approved",
    [(True, False), (False, False), (False, True)],
    ids=["predicate-requires-human", "direct-requires-human", "direct-approves"],
)
async def test_empty_formal_context_is_reviewed(monkeypatch, kind, messages, triggered, direct_approved):
    bus, sink = EventBus(), FakeSink()
    context = TurnContextHolder()
    context.replace(messages)
    values = core.predicate_values() | {"external_action": triggered}
    replies = [json.dumps(values)]
    if not triggered:
        replies.append(json.dumps({"approved": direct_approved, "reason": "Reviewed without user context"}))
    approved = not triggered and direct_approved
    async with core.judge_client(monkeypatch, replies) as (judge, calls):
        broker = AcpPermissionBroker(
            event_bus=bus,
            session_id="parent",
            caller_name="External",
            mode_getter=lambda: ApprovalMode.AUTO,
            turn_context=context,
            workspace_roots=["/workspace"],
            workspace_cwd="/workspace",
            approval_judge=judge,
            ask_user_timeout_seconds=None,
            trajectory_context=make_context(sink),
            trajectory_boundary_operation_id=new_analytics_id(),
        )
        translator = AcpUpdateTranslator(
            event_bus=bus,
            session_id="parent",
            agent_name="External",
            invocation_id="inv",
            attempt=1,
            origin=InvocationOrigin("sub_agent", "parent", "inv", None),
        )
        broker.set_translator(translator)

        async def human_approve(event: ApprovalReviewed):
            if not event.approved:
                await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, session_id="parent"))

        await bus.subscribe(ApprovalReviewed, human_approve)
        async with capture_event_sequence(bus, ApprovalRequest, ApprovalReviewed) as events:
            task = asyncio.create_task(
                broker.on_permission_request(
                    ToolCallUpdate(toolCallId="call", title="Inspect source", kind=kind, rawInput={"path": "main.py"}),
                    [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
                )
            )
            try:
                await wait_for(task.done, description="ACP review without user context settled")
                decision = await task
                assert decision.action == "allow"
                request, review = events
                assert isinstance(request, ApprovalRequest)
                assert isinstance(review, ApprovalReviewed)
                assert request.judging is True
                assert review.request_id == request.request_id
                assert review.approved is approved
                assert len(calls) == (1 if triggered else 2)
                prompts = [json.loads(call.content)["messages"][1]["content"] for call in calls]
                assert all("Current-turn user prompts:\n(no user message available)" in prompt for prompt in prompts)
                assert all("Latest user prompt:\n(no user message available)" in prompt for prompt in prompts)
                if not triggered:
                    assert json.loads(calls[1].content)["messages"][0]["content"] == judge_module._SYSTEM_PROMPT
                reviews = [
                    item["update"]
                    for item in translator.translated_updates
                    if item["update"]["sessionUpdate"] == "permission_review"
                ]
                assert len(reviews) == 1
                assert "formal_audit" not in reviews[0]
                audit = reviews[0]["formal_judge"]
                assert audit["failure_reason"] is None
                assert audit["approved"] is approved
                assert audit["stages"] == (["predicate"] if triggered else ["predicate", "reasoning"])
                assert audit["predicate_results"] == [{"id": key, "value": value} for key, value in values.items()]
                resolved = sink.only(EventType.APPROVAL_RESOLVED).payload
                assert resolved["decider"] == ("judge" if approved else "user")
                assert resolved["formal_judge"]["approved"] is approved
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await broker.close()
                await bus.unsubscribe(ApprovalReviewed, human_approve)


@pytest.mark.parametrize("interrupt", [False, True])
async def test_acp_terminal_keeps_formal_audit_when_human_or_interrupt_wins(monkeypatch, interrupt):
    bus, sink = EventBus(), FakeSink()
    context = TurnContextHolder()
    context.replace(["Inspect source"])
    entered, release = asyncio.Event(), asyncio.Event()
    async with capture_event_sequence(bus, ApprovalRequest, ApprovalReviewed) as events:

        async def response(client, messages, *, stream, options, timeout):
            entered.set()
            if interrupt:
                await release.wait()
            else:
                request = next(event for event in events if isinstance(event, ApprovalRequest))
                await bus.publish(ApprovalResponse(request_id=request.request_id, approved=False, session_id="parent"))
            return ChatResponse(messages=[Message("assistant", [json.dumps(core.predicate_values())])])

        monkeypatch.setattr(
            judge_module, "get_final_response", create_autospec(judge_module.get_final_response, side_effect=response)
        )
        async with core.judge_client(monkeypatch, []) as (judge, _):
            broker = AcpPermissionBroker(
                event_bus=bus,
                session_id="parent",
                caller_name="External",
                mode_getter=lambda: ApprovalMode.AUTO,
                turn_context=context,
                workspace_roots=[],
                workspace_cwd="",
                approval_judge=judge,
                ask_user_timeout_seconds=None,
                trajectory_context=make_context(sink),
                trajectory_boundary_operation_id=new_analytics_id(),
            )
            task = asyncio.create_task(
                broker.on_permission_request(
                    ToolCallUpdate(toolCallId="call", title="Inspect source", rawInput={}),
                    [
                        PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
                        PermissionOption(optionId="deny", name="Deny", kind="reject_once"),
                    ],
                )
            )
            try:
                await wait_for(lambda: entered.is_set() or task.done(), description="ACP Formal request started")
                if interrupt:
                    assert not task.done()
                    task.cancel()
                await wait_for(task.done, description="ACP permission settled")
                decision = await task
                assert decision.action == ("cancelled" if interrupt else "deny")
                resolved = sink.only(EventType.APPROVAL_RESOLVED).payload
                assert resolved["formal_judge"].get("failure_reason") == ("cancelled" if interrupt else None)
                assert resolved["formal_judge"]["approved"] is (not interrupt)
                assert not any(isinstance(event, ApprovalReviewed) for event in events)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await broker.close()
