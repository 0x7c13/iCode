# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AcpPermissionBroker: permission and ask-user bridging, wait cancellation, and abort latching."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    PermissionOption,
    ToolCallUpdate,
)

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ApprovalAutoFulfillBlocked,
    ApprovalCancelled,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalReviewed,
    AskUserResponse,
    AskUserTimedOut,
    QuestionToUser,
)
from chrys.foundation.models.ask_user import AskUserAnswer, AskUserOption, AskUserQuestion
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.trajectory.event_types import EventType, WaitCategory
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.orchestration.invoker.acp_protocol import AcpPermissionBroker, AcpUpdateTranslator
from chrys.orchestration.invoker.contracts import AbortCause
from chrys.orchestration.invoker.resources import OperationLifetime
from chrys.service.approval.judge import JudgeVerdict
from chrys.service.approval.policy import ApprovalMode
from chrys.service.approval.turn_context import TurnContextHolder
from chrys.service.trajectory.approvals import ApprovalDecider
from chrys.service.trajectory.waits import WaitOutcome
from tests.orchestration.sub_agents._acp_fakes import make_broker, make_controller
from tests.service.trajectory._fakes import CancelAckSink, FakeSink, make_context
from tests.support.event_capture import capture_events
from tests.support.waiting import wait_for

_ALLOW = PermissionOption(optionId="allow", name="Allow", kind="allow_once")


async def _request_permission(broker: AcpPermissionBroker) -> Any:
    return await broker.on_permission_request(ToolCallUpdate(toolCallId="call", title="Run", rawInput={}), [_ALLOW])


async def _request_input(broker: AcpPermissionBroker) -> Any:
    return await broker.on_ext_method(
        "chrys/request_input",
        {"sessionId": "remote", "requestId": "remote-request", "questions": [{"question": "Continue?"}]},
    )


async def test_owner_close_synchronously_revokes_already_approved_permission(tmp_path) -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=None)
    commits: list[bool] = []
    controller = make_controller(
        bus, tmp_path, broker=broker, parent_interrupted_result_commit=lambda: commits.append(True)
    )
    operation = OperationLifetime()
    # The pass has returned; the real shell lifetime still covers permission
    # cleanup. No artificial pause decision future is installed in this window.
    operation.begin_cleanup()
    controller.attach_operation(operation)
    snapshots: list[tuple[bool, bool, bool, bool]] = []

    async def approve_then_close(event: ApprovalRequest) -> None:
        future = broker._permission_waits[event.request_id]
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, session_id="parent"))
        controller.request_close(AbortCause.OWNER_CLOSE)
        snapshots.append(
            (
                controller._cascade_requested,
                commits == [True] and controller._parent_interrupted_result_commit is None,
                broker._aborted,
                future.done() and future.result().approved,
            )
        )

    await bus.subscribe(ApprovalRequest, approve_then_close)
    try:
        result = await _request_permission(broker)
        assert result.action == "cancelled"
        assert len(snapshots) == 1
        for value in snapshots[0]:
            assert value is True, snapshots
        assert controller._pending_decision is None
        assert broker._permission_waits == {}
    finally:
        operation.finish()
        await asyncio.wait_for(controller.drained, 5)
        await bus.unsubscribe(ApprovalRequest, approve_then_close)
        await broker.close()


@dataclass(frozen=True)
class _WaitChannel:
    """One of the broker's two user-facing wait channels and the events it publishes.

    The permission and ask-user paths open a dialog, register a future, and
    tear both down on cancellation through the same skeleton; the cancellation
    tests below run once per channel with the channel supplying the request
    call, the dialog events, the pending-wait registry, and the trajectory
    events it is expected to leave behind.
    """

    name: str
    request: Callable[[AcpPermissionBroker], Awaitable[Any]]
    is_cancelled: Callable[[Any], bool]
    opened: type[Any]  # published when the dialog opens
    closed: type[Any]  # published when an opened dialog is torn down
    pending: Callable[[AcpPermissionBroker], dict[str, Any]]  # the broker's registry of pending futures
    started_event: EventType
    finished_event: EventType
    cancelled_payload: tuple[str, Any]  # (payload key, value) the finish event carries after a cancel
    answer: Callable[[str], Any]  # the user's positive response to a pending request


_PERMISSION = _WaitChannel(
    name="permission",
    request=_request_permission,
    is_cancelled=lambda decision: decision.action == "cancelled",
    opened=ApprovalRequest,
    closed=ApprovalCancelled,
    pending=lambda broker: broker._permission_waits,
    started_event=EventType.APPROVAL_REQUESTED,
    finished_event=EventType.APPROVAL_RESOLVED,
    cancelled_payload=("decider", ApprovalDecider.NONE),
    answer=lambda request_id: ApprovalResponse(request_id=request_id, approved=True, session_id="parent"),
)
_INPUT = _WaitChannel(
    name="input",
    request=_request_input,
    is_cancelled=lambda response: response == {"cancelled": True},
    opened=QuestionToUser,
    closed=AskUserTimedOut,
    pending=lambda broker: broker._ask_waits,
    started_event=EventType.WAIT_STARTED,
    finished_event=EventType.WAIT_FINISHED,
    cancelled_payload=("outcome", WaitOutcome.CANCELLED),
    answer=lambda request_id: AskUserResponse(request_id=request_id, answers=(AskUserAnswer(values=("proceed",)),)),
)
_channels = pytest.mark.parametrize("channel", [_PERMISSION, _INPUT], ids=["permission", "input"])


async def test_permission_polarity_preflight_runs_before_bypass() -> None:
    bus = EventBus()
    requests = await capture_events(bus, ApprovalRequest)
    broker = make_broker(bus, [ApprovalMode.BYPASS])
    tool_call = ToolCallUpdate(toolCallId="call", title="Spoof", kind="execute", rawInput={"command": "x"})
    reject = PermissionOption(optionId="reject", name="Reject", kind="reject_once")

    decision = await broker.on_permission_request(tool_call, [reject])
    empty_decision = await broker.on_permission_request(tool_call, [])

    assert decision.action == "deny"
    assert decision.option_id == "reject"
    assert empty_decision.action == "cancelled"
    assert requests == []


async def test_permission_request_uses_spoof_proof_presentation_and_user_decision_wins() -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL])
    requests = await capture_events(bus, ApprovalRequest)

    async def respond(event: ApprovalRequest) -> None:
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=False,
                reason="no",
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalRequest, respond)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="write_file", kind="edit", rawInput={"path": "x"}),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
        ],
    )
    assert decision.action == "deny"
    assert requests[0].tool_name == "acp:write_file"
    assert requests[0].tool_kind == ""
    assert requests[0].presentation_kind == "filesystem.write"
    assert requests[0].user_message == "original prompt"

    # A kind chrys does not recognize still yields a non-empty presentation
    # hint so the dialog can tell bridged requests from local ones.
    await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call2", title="mystery", kind="think", rawInput={}),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
        ],
    )
    assert requests[1].presentation_kind == "remote"


async def test_auto_permission_reuses_judge_race_with_separate_judge_and_presentation_fields() -> None:
    bus = EventBus()
    context = TurnContextHolder()
    context.replace(["review this"])

    class Judge:
        called_with: dict[str, Any] | None = None

        async def evaluate(self, **kwargs: Any) -> JudgeVerdict:
            self.called_with = kwargs
            return JudgeVerdict(approved=True, reason="safe")

    judge = Judge()
    broker = AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: ApprovalMode.AUTO,
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=judge,
        ask_user_timeout_seconds=1,
    )
    requests = await capture_events(bus, ApprovalRequest)

    async def reject_after_review(event: ApprovalReviewed) -> None:
        await bus.publish(ApprovalAutoFulfillBlocked(request_id=event.request_id))
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=False,
                reason="user wins",
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalReviewed, reject_after_review)
    decision = await broker.on_permission_request(
        ToolCallUpdate(
            toolCallId="call",
            title="outer intent",
            kind="other",
            rawInput={"command": "echo ok"},
            _meta={"chrys": {"tool_name": "zsh", "tool_kind": "shell"}},
        ),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="deny", name="Deny", kind="reject_once"),
        ],
    )
    assert decision.action == "deny"
    assert requests[0].tool_name == "acp:outer intent"
    assert requests[0].tool_kind == ""
    # Presentation rides the _meta-enhanced judge kind, never tool_kind.
    assert requests[0].presentation_kind == "shell"
    assert judge.called_with is not None
    assert judge.called_with["tool_name"] == "zsh"
    assert judge.called_with["tool_kind"] == "shell"
    assert judge.called_with["args"] == {"command": "echo ok"}
    assert judge.called_with["log_dir"] is None


async def test_auto_permission_records_user_when_user_beats_enabled_judge() -> None:
    bus = EventBus()
    sink = FakeSink()
    context = TurnContextHolder()
    context.replace(["review this"])

    class Judge:
        async def evaluate(self, **_kwargs: Any) -> JudgeVerdict:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    broker = AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: ApprovalMode.AUTO,
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=Judge(),
        ask_user_timeout_seconds=1,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    async def approve(event: ApprovalRequest) -> None:
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, session_id=event.session_id))

    await bus.subscribe(ApprovalRequest, approve)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
        [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
    )

    assert decision.action == "allow"
    assert sink.only(EventType.APPROVAL_RESOLVED).payload["decider"] == ApprovalDecider.USER


async def test_judge_and_dialog_receive_untruncated_raw_arguments() -> None:
    """A destructive suffix beyond the preview cap must stay visible (§9.2).

    The judge and the published approval both authorize what the remote will
    execute; a truncating canonicalization here would let AUTO approve the
    benign prefix of a command whose tail it never saw.
    """
    bus = EventBus()
    context = TurnContextHolder()
    context.replace(["review this"])

    class Judge:
        called_with: dict[str, Any] | None = None

        async def evaluate(self, **kwargs: Any) -> JudgeVerdict:
            self.called_with = kwargs
            return JudgeVerdict(approved=True, reason="safe")

    judge = Judge()
    broker = AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: ApprovalMode.AUTO,
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=judge,
        ask_user_timeout_seconds=1,
    )
    requests = await capture_events(bus, ApprovalRequest)
    command = "echo " + "x" * 5_000 + " && rm -rf target"
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="run", kind="execute", rawInput={"command": command}),
        [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
    )
    assert decision.action == "allow"
    assert judge.called_with is not None
    assert judge.called_with["args"] == {"command": command}
    assert requests[0].args == {"command": command}


async def test_standalone_permission_requests_enter_the_audit_trail() -> None:
    """Permission requests with no surrounding ToolCallStart still get audited.

    ACP permits ``session/request_permission`` before any tool-call update;
    judge logging is deliberately ``log_dir=None``, so the translator audit
    ring is the only durable home for the request's rawInput and outcome.
    """
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL])
    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id="parent",
        agent_name="External",
        invocation_id="inv",
        attempt=1,
        origin=InvocationOrigin("sub_agent", "parent", "inv", None),
    )
    broker.set_translator(translator)

    async def respond(event: ApprovalRequest) -> None:
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=False,
                reason="no",
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalRequest, respond)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="write_file", kind="edit", rawInput={"path": "secrets.txt"}),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
        ],
    )
    assert decision.action == "deny"
    entries = translator.translated_updates
    assert entries
    assert entries[-1]["attempt"] == 1
    recorded = entries[-1]["update"]
    assert recorded["sessionUpdate"] == "permission_request"
    assert recorded["title"] == "write_file"
    assert recorded["kind"] == "filesystem.write"
    assert recorded["rawInput"] == {"path": "secrets.txt"}
    assert recorded["outcome"] == "denied"


async def test_permission_abandonment_and_ask_user_timeout_publish_clear_events() -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=0.01)
    cancelled = await capture_events(bus, ApprovalCancelled)
    timed_out = await capture_events(bus, AskUserTimedOut)
    questions = await capture_events(bus, QuestionToUser)
    allow = PermissionOption(optionId="allow", name="Allow", kind="allow_once")
    permission_task = asyncio.create_task(
        broker.on_permission_request(
            ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
            [allow],
        )
    )
    await wait_for(lambda: bool(broker._permission_waits), description="permission wait")
    await broker.cancel_pending_waits()
    assert (await permission_task).action == "cancelled"
    assert cancelled

    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "questions": [{"question": "Continue?", "options": [{"label": "yes"}, {"label": "no"}]}],
            "callerName": "Remote",
        },
    )
    assert result == {"cancelled": True}
    assert questions
    assert timed_out


async def test_request_input_normalizes_option_labels_before_controller_publication() -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=None)
    published: list[QuestionToUser] = []

    async def respond(event: QuestionToUser) -> None:
        published.append(event)
        await bus.publish(
            AskUserResponse(
                request_id=event.request_id,
                answers=(AskUserAnswer(values=("A",)),),
                session_id=event.session_id,
            )
        )

    await bus.subscribe(QuestionToUser, respond)
    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "questions": [
                {
                    "question": "Pick?",
                    "options": [{"label": "  A\u200b  "}, {"label": "\u200b"}, {"label": "   "}],
                }
            ],
        },
    )

    assert result == {"answers": [{"values": ["A"], "note": ""}], "cancelled": False}
    assert len(published) == 1
    assert published[0].questions == (AskUserQuestion(question="Pick?", options=(AskUserOption("A"),)),)


async def test_legacy_request_input_shape_is_rejected() -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=None)
    questions = await capture_events(bus, QuestionToUser)

    with pytest.raises(RequestError):
        await broker.on_ext_method(
            "chrys/request_input",
            {
                "sessionId": "remote",
                "requestId": "remote-request",
                "question": "Pick?",
                "options": ["A"],
            },
        )

    assert questions == []
    assert broker._ask_waits == {}


async def test_acp_permission_and_input_waits_target_the_boundary() -> None:
    bus = EventBus()
    sink = FakeSink()
    boundary = new_analytics_id()
    broker = make_broker(
        bus,
        [ApprovalMode.MANUAL],
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=boundary,
    )

    async def approve(event: ApprovalRequest) -> None:
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, session_id=event.session_id))

    async def answer(event: QuestionToUser) -> None:
        await bus.publish(
            AskUserResponse(
                request_id=event.request_id,
                answers=(AskUserAnswer(values=("yes",)),),
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalRequest, approve)
    await bus.subscribe(QuestionToUser, answer)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
        [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
    )
    response = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "questions": [{"question": "Continue?", "options": [{"label": "yes"}, {"label": "no"}]}],
        },
    )

    assert decision.action == "allow"
    assert response == {"answers": [{"values": ["yes"], "note": ""}], "cancelled": False}
    approval_events = [
        event
        for event in sink.drafts
        if event.event_type in {EventType.APPROVAL_REQUESTED, EventType.APPROVAL_RESOLVED}
    ]
    assert len(approval_events) == 2
    assert all(event.parent_operation_id == boundary for event in approval_events)
    assert all(event.payload["target_operation_id"] == boundary for event in approval_events)
    assert all(event.operation_id == event.payload["approval_request_id"] for event in approval_events)
    assert all(event.operation_id != event.parent_operation_id for event in approval_events)
    wait_events = [
        event for event in sink.drafts if event.event_type in {EventType.WAIT_STARTED, EventType.WAIT_FINISHED}
    ]
    assert len(wait_events) == 2
    assert all(event.payload["category"] == WaitCategory.USER_INPUT for event in wait_events)
    assert all(event.parent_operation_id == boundary for event in wait_events)
    sink.assert_operations_settled()


async def test_ask_timeout_notification_cancellation_cannot_leave_wait_open() -> None:
    bus = EventBus()
    sink = FakeSink()
    broker = make_broker(
        bus,
        [ApprovalMode.MANUAL],
        timeout=0,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    async def cancel_notification(_event: AskUserTimedOut) -> None:
        raise asyncio.CancelledError

    await bus.subscribe(AskUserTimedOut, cancel_notification)
    with pytest.raises(asyncio.CancelledError):
        await broker.on_ext_method(
            "chrys/request_input",
            {
                "sessionId": "remote",
                "requestId": "remote-request",
                "questions": [{"question": "Continue?"}],
            },
        )

    assert sink.only(EventType.WAIT_FINISHED).payload["outcome"] == WaitOutcome.TIMED_OUT
    sink.assert_operations_settled()


async def test_ask_user_is_suppressed_when_owner_is_noninteractive() -> None:
    bus = EventBus()
    questions = await capture_events(bus, QuestionToUser)
    broker = make_broker(
        bus,
        [ApprovalMode.MANUAL],
        timeout=None,
        allow_user_interaction=False,
    )

    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "questions": [{"question": "Continue?", "options": [{"label": "yes"}, {"label": "no"}]}],
            "callerName": "Remote",
        },
    )

    assert result == {"cancelled": True}
    assert questions == []
    assert broker._ask_waits == {}


async def test_ask_user_none_timeout_preserves_wait_until_response() -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=None)

    async def respond(event: QuestionToUser) -> None:
        await bus.publish(AskUserResponse(request_id=event.request_id, answers=(AskUserAnswer(values=("yes",)),)))

    await bus.subscribe(QuestionToUser, respond)
    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "questions": [{"question": "Continue?"}],
            "callerName": "Remote",
        },
    )
    assert result == {"answers": [{"values": ["yes"], "note": ""}], "cancelled": False}


async def test_ask_user_round_trip_is_structured_and_modal_only() -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=None)
    questions = (
        AskUserQuestion("Pick?", "Choice", (AskUserOption("A", "First"),)),
        AskUserQuestion("Targets?", "Targets", (AskUserOption("TUI"), AskUserOption("ACP")), True),
        AskUserQuestion("Note?", "Note"),
    )
    answers = (
        AskUserAnswer(("A",), "because"),
        AskUserAnswer(("TUI", "ACP")),
        AskUserAnswer(("Canary",)),
    )

    async def respond(event: QuestionToUser) -> None:
        assert event.questions == questions
        assert event.call_id == ""
        await bus.publish(AskUserResponse(request_id=event.request_id, answers=answers))

    await bus.subscribe(QuestionToUser, respond)
    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "questions": [
                {
                    "question": "Pick?",
                    "header": "Choice",
                    "options": [{"label": "A", "description": "First"}],
                    "multiSelect": False,
                },
                {
                    "question": "Targets?",
                    "header": "Targets",
                    "options": [{"label": "TUI", "description": ""}, {"label": "ACP", "description": ""}],
                    "multiSelect": True,
                },
                {"question": "Note?", "header": "Note", "options": [], "multiSelect": False},
            ],
        },
    )

    assert result == {
        "answers": [
            {"values": ["A"], "note": "because"},
            {"values": ["TUI", "ACP"], "note": ""},
            {"values": ["Canary"], "note": ""},
        ],
        "cancelled": False,
    }


@_channels
async def test_start_ack_cancellation_clears_registered_future(channel: _WaitChannel) -> None:
    bus = EventBus()
    sink = CancelAckSink(at=1)
    broker = make_broker(
        bus,
        [ApprovalMode.MANUAL],
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    outcome = await channel.request(broker)

    assert channel.is_cancelled(outcome)
    assert channel.pending(broker) == {}
    assert sink.event_types == [channel.started_event, channel.finished_event]
    sink.assert_operations_settled()


@_channels
async def test_publish_cancellation_closes_opened_dialog(channel: _WaitChannel) -> None:
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=None)
    opened = await capture_events(bus, channel.opened)
    closed = await capture_events(bus, channel.closed)
    publication_blocked = asyncio.Event()
    never = asyncio.Event()

    async def block_later_handler(_event: Any) -> None:
        publication_blocked.set()
        await never.wait()

    await bus.subscribe(channel.opened, block_later_handler)
    task = asyncio.create_task(channel.request(broker))
    await publication_blocked.wait()
    task.cancel()
    outcome = await task

    assert channel.is_cancelled(outcome)
    assert len(opened) == 1
    assert [event.request_id for event in closed] == [opened[0].request_id]
    assert channel.pending(broker) == {}


@_channels
async def test_cancel_notification_cancellation_cannot_leave_wait_open(channel: _WaitChannel) -> None:
    bus = EventBus()
    sink = FakeSink()
    broker = make_broker(
        bus,
        [ApprovalMode.MANUAL],
        timeout=None,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    async def cancel_notification(_event: Any) -> None:
        raise asyncio.CancelledError

    await bus.subscribe(channel.closed, cancel_notification)
    task = asyncio.create_task(channel.request(broker))
    await wait_for(lambda: bool(channel.pending(broker)), description=f"{channel.name} wait")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    key, value = channel.cancelled_payload
    assert sink.only(channel.finished_event).payload[key] == value
    sink.assert_operations_settled()


@_channels
async def test_settled_answer_is_dropped_when_abort_latches_before_resume(channel: _WaitChannel) -> None:
    """A settled "allow" or answer racing ahead of the backgrounded cancel must not win.

    cascade_abort latches broker.mark_aborted() synchronously; a request whose
    response future was already resolved — and whose resolving continuation is
    queued ahead of cancel_pending_waits — must still resolve to cancelled,
    else the remote runs a tool or proceeds past the global interrupt.
    """
    bus = EventBus()
    broker = make_broker(bus, [ApprovalMode.MANUAL], timeout=None)
    task = asyncio.create_task(channel.request(broker))
    await wait_for(lambda: bool(channel.pending(broker)), description=f"{channel.name} wait")
    request_id = next(iter(channel.pending(broker)))

    # The user's response settles the future BEFORE the abort lands...
    await bus.publish(channel.answer(request_id))
    # ...but the abort latches before the awaiter's continuation resumes.
    broker.mark_aborted()

    assert channel.is_cancelled(await task)


@_channels
async def test_request_after_abort_latched_is_refused_without_publishing(channel: _WaitChannel) -> None:
    """Once aborted, a fresh request — even a BYPASS-mode auto-allow — is refused with no dialog published."""
    bus = EventBus()
    opened = await capture_events(bus, channel.opened)
    # Finite timeout so that if the entry-guard regresses the call still returns
    # (via timeout) instead of hanging — the published-dialog assertion is what
    # actually pins the invariant.
    broker = make_broker(bus, [ApprovalMode.BYPASS], timeout=0.1)
    broker.mark_aborted()

    assert channel.is_cancelled(await channel.request(broker))
    assert opened == []
