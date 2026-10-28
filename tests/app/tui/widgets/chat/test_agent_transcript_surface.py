# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared transcript contracts independent of any workflow controller or runner."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import AgentMessageChunk, TextContentBlock, ToolCallProgress, ToolCallStart
from rich.text import Text
from textual.widgets import Button, Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.main.state import MainScreenState, RunState
from chrys.app.tui.util.context_pressure import context_pressure_message
from chrys.app.tui.widgets.chat import tool_call as tool_call_module
from chrys.app.tui.widgets.chat.agent_transcript_surface import (
    AgentTranscriptJournal,
    AgentTranscriptSurface,
    AnswerSegments,
    TranscriptAssistantOp,
    TranscriptCompactionFinishedOp,
    TranscriptCompactionStartOp,
    TranscriptErrorOp,
    TranscriptInterruptedOp,
    TranscriptPausedOp,
    TranscriptResumedOp,
    TranscriptRetryOp,
    TranscriptToolResultOp,
    TranscriptToolStartOp,
    TranscriptToolStatusOp,
    TranscriptUserOp,
    TranscriptWarningOp,
)
from chrys.app.tui.widgets.chat.compaction_card import CompactionCard
from chrys.app.tui.widgets.chat.messages import (
    INTERRUPTED_REASON_PLAIN_MESSAGE,
    AgentMessage,
    ErrorMessage,
    InterruptedMessage,
    RetryMessage,
    SystemMessage,
    UserMessage,
)
from chrys.app.tui.widgets.chat.renderers import sleep as sleep_module
from chrys.app.tui.widgets.chat.renderers.sleep import SleepToolCall
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard, ToolGroup, ToolGroupTitle
from chrys.app.tui.widgets.workflow.projector import INVOCATION_EVENTS, journal_operation
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationContentDelta,
    InvocationContentFinished,
    InvocationEvent,
    InvocationToolCallStart,
)
from chrys.foundation.i18n import DisplayBlock
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.invocations import ContentChannel, ContentStatus, InvocationOrigin
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_READ, KIND_FILESYSTEM_WRITE, KIND_SLEEP
from chrys.orchestration.invoker.acp_protocol import AcpUpdateTranslator
from chrys.service.session.sub_agent_transcript import project_agent_transcript
from tests.orchestration.sub_agents._acp_fakes import session_notification
from tests.support.tui_helpers import LocalizedWidgetApp, make_backend_handler, title_text
from tests.support.waiting import wait_for

_TURN = InvocationOrigin("turn", "s1", "main", None)
_CHILD = InvocationOrigin("sub_agent", "s1", "child", _TURN)
_OTHER = InvocationOrigin("sub_agent", "s1", "other", _TURN)


def _delta(
    segment_id: str, text: str, channel: ContentChannel = "answer", origin: InvocationOrigin = _CHILD
) -> InvocationContentDelta:
    return InvocationContentDelta(origin=origin, segment_id=segment_id, channel=channel, text=text)


def _end(
    segment_id: str,
    status: ContentStatus = "completed",
    channel: ContentChannel = "answer",
    origin: InvocationOrigin = _CHILD,
) -> InvocationContentFinished:
    return InvocationContentFinished(origin=origin, segment_id=segment_id, channel=channel, status=status)


def _shown(answers: AnswerSegments, events: list[InvocationContentDelta | InvocationContentFinished]) -> list[str]:
    return [f"{op.segment_id}: {op.text}" for event in events if (op := answers.take(event)) is not None]


def test_answer_segments_show_each_completed_answer_whole_once() -> None:
    shown = _shown(
        AnswerSegments(),
        [
            _delta("a", "Hello "),
            _delta("x", "Other ", origin=_OTHER),
            _delta("a", "world"),
            _end("a"),
            _delta("r", "Hmm", "reasoning"),
            _end("r", channel="reasoning"),
            _delta("b", "Checking sources."),
            _end("b"),
            _delta("x", "answer", origin=_OTHER),
            _end("x", origin=_OTHER),
        ],
    )

    assert shown == ["a: Hello world", "b: Checking sources.", "x: Other answer"]


@pytest.mark.parametrize(
    ("text", "status"),
    [("Half an", "failed"), ("Half an", "interrupted"), (" \n", "completed")],
    ids=["failed", "interrupted", "blank"],
)
def test_answer_segments_never_show_a_cut_off_or_blank_answer(text: str, status: ContentStatus) -> None:
    answers = AnswerSegments()

    # Its text is gone with it: nothing later shows it.
    assert _shown(answers, [_delta("a", text), _end("a", status), _end("a")]) == []


async def _subscribe_workflow_transcript(bus: EventBus, shown: list[str]) -> None:
    answers = AnswerSegments()

    async def record(event: InvocationEvent) -> None:
        operation = journal_operation(event, answers)
        if isinstance(operation, TranscriptAssistantOp):
            shown.append(operation.text)
        elif isinstance(operation, TranscriptToolStartOp) and "tool" not in shown:
            shown.append("tool")

    for event_type in (InvocationContentDelta, InvocationContentFinished, InvocationToolCallStart):
        await bus.subscribe(event_type, record)


async def _subscribe_chat_card(bus: EventBus, shown: list[str]) -> None:
    class Panel:
        def add_sub_agent_answer(self, agent_name: str, invocation_id: str, segment_id: str, text: str) -> None:
            shown.append(text)

        async def add_sub_agent_tool_start(
            self,
            agent_name: str,
            invocation_id: str,
            tool_name: str,
            args: dict[str, object],
            call_id: str,
            *,
            tool_kind: str,
            provider_hosted: bool = False,
            hosted_family: str = "",
            provider: str = "",
            provider_item_type: str = "",
            provider_status: str = "",
            provider_call_id: str = "",
        ) -> None:
            if "tool" not in shown:
                shown.append("tool")

    handler = make_backend_handler(
        SimpleNamespace(_state=MainScreenState(run=RunState(agent_running=True)), query_one=lambda _: Panel())
    )
    # As the main screen subscribes them.
    await bus.subscribe(InvocationContentDelta, handler.on_agent_content)
    await bus.subscribe(InvocationContentFinished, handler.on_agent_content)
    await bus.subscribe(InvocationToolCallStart, handler.on_tool_start)


@pytest.mark.parametrize("consumer", [_subscribe_workflow_transcript, _subscribe_chat_card], ids=["workflow", "chat"])
@pytest.mark.parametrize("hosted", [False, True], ids=["local", "hosted"])
@pytest.mark.parametrize("tool_end", ["completed", "transport_failure"])
async def test_acp_tool_updates_leave_the_answer_they_interrupt_to_its_segment_end(
    tool_end: str, hosted: bool, consumer: Callable[[EventBus, list[str]], Awaitable[None]]
) -> None:
    """An ACP agent re-sends a call's start to update its card: the answer streaming around it shows once, whole."""
    bus = EventBus()
    shown: list[str] = []
    await consumer(bus, shown)
    translator = AcpUpdateTranslator(
        event_bus=bus, session_id="s1", agent_name="External", invocation_id="child", attempt=0, origin=_CHILD
    )

    def message(text: str) -> AgentMessageChunk:
        return AgentMessageChunk(
            sessionUpdate="agent_message_chunk", messageId="m1", content=TextContentBlock(type="text", text=text)
        )

    # An iCode agent marks the calls its provider runs; the translator keeps that on every start it re-sends.
    meta = {"chrys": {"provider_hosted": True, "hosted_family": "web_search"}} if hosted else None
    updates: list[ToolCallStart | ToolCallProgress | AgentMessageChunk] = [
        ToolCallStart(
            sessionUpdate="tool_call", toolCallId="read", title="Read", kind="read", status="in_progress", _meta=meta
        ),
        message("Half an "),
    ]
    if tool_end == "completed":
        updates += [
            ToolCallProgress(
                sessionUpdate="tool_call_update", toolCallId="read", title="Read file", status="completed", _meta=meta
            ),
            message("answer."),
        ]
    for seq, update in enumerate(updates, start=1):
        await translator.put(seq, session_notification(update))
    if tool_end == "completed":
        await translator.finish_content("completed")
        assert shown == ["tool", "Half an answer."]
    else:
        # The cut-off answer never shows, though the call's interrupted card re-sends its start.
        await translator.flush_interrupted()
        assert shown == ["tool"]


@pytest.mark.parametrize("again", ["before_result", "new_input"])
async def test_an_acp_call_started_again_after_a_later_card_keeps_its_one_card(again: str) -> None:
    """An ACP agent re-sends a running call's start after later output ended the call's tool group.

    Live and reopened views update the call's card where it is, and never open another.
    """
    bus = EventBus()
    journal = AgentTranscriptJournal()
    answers = AnswerSegments()

    async def record(event: InvocationEvent) -> None:
        if (operation := journal_operation(event, answers)) is not None:
            journal.record(operation)

    for event_type in INVOCATION_EVENTS:
        await bus.subscribe(event_type, record)
    translator = AcpUpdateTranslator(
        event_bus=bus, session_id="s1", agent_name="External", invocation_id="child", attempt=1, origin=_CHILD
    )

    def start(call_id: str) -> ToolCallStart:
        return ToolCallStart(
            sessionUpdate="tool_call",
            toolCallId=call_id,
            title="Read",
            kind="read",
            status="in_progress",
            rawInput={"path": f"{call_id}.txt"},
        )

    if again == "before_result":
        update = ToolCallProgress(sessionUpdate="tool_call_update", toolCallId="first", status="completed")
        shown = ("Read", KIND_FILESYSTEM_READ, {"path": "first.txt"})
        expected = [(0, False, shown, shown)]
    else:
        update = ToolCallProgress(
            sessionUpdate="tool_call_update",
            toolCallId="first",
            title="Write",
            kind="edit",
            rawInput={"path": "new.txt"},
        )
        shown = ("Write", KIND_FILESYSTEM_WRITE, {"path": "new.txt"})
        expected = [(0, True, shown, shown)]
    message = AgentMessageChunk(
        sessionUpdate="agent_message_chunk", messageId="m1", content=TextContentBlock(type="text", text="Next one.")
    )

    def first_cards(view: AgentTranscriptSurface) -> list[tuple[object, ...]]:
        """Each card of the first call: its group's position, whether it runs, and its name, kind and arguments.

        Name, kind and arguments come twice: the call's record, then what its card shows (a collapsed group mounts
        its cards from their records when it opens).
        """
        cards: list[tuple[object, ...]] = []
        for position, group in enumerate(view.query(ToolGroup)):
            if (record := group._tool_records.get("a1:first")) is None:
                continue
            recorded = (record.tool_name, record.tool_kind, record.args)
            card = group.get_tool("a1:first")
            shown = (card.tool_name, card.tool_kind, card.args) if isinstance(card, BaseToolCard) else recorded
            cards.append((position, group.is_tool_running("a1:first"), recorded, shown))
        return cards

    live = AgentTranscriptSurface(journal)
    async with LocalizedWidgetApp(lambda: live).run_test() as pilot:
        for seq, update_or_text in enumerate([start("first"), message, start("second"), update], start=1):
            await translator.put(seq, session_notification(update_or_text))
        await wait_for(lambda: expected[0][1:] in [card[1:] for card in first_cards(live)], pilot=pilot)
        assert first_cards(live) == expected
    reopened = AgentTranscriptSurface(journal)
    async with LocalizedWidgetApp(lambda: reopened).run_test() as pilot:
        await wait_for(lambda: expected[0][1:] in [card[1:] for card in first_cards(reopened)], pilot=pilot)
        assert first_cards(reopened) == expected


@pytest.mark.parametrize("archived", [False, True], ids=["live", "archive"])
async def test_input_intermediate_tools_and_final_keep_their_order(archived: bool) -> None:
    journal = AgentTranscriptJournal()
    journal.record(TranscriptUserOp("Review [literal] input"))
    messages = [{"role": "user", "contents": [{"type": "text", "text": "Review [literal] input"}]}]
    for index in range(2):
        call_id, text = f"read-{index}", f"Inspection {index}."
        journal.record(TranscriptAssistantOp(text, segment_id=f"segment-{index}"))
        journal.record(TranscriptToolStartOp(call_id, "read_file", KIND_FILESYSTEM_READ, {"path": "README.md"}))
        journal.record(TranscriptToolResultOp(call_id, "read_file", "contents"))
        messages.extend(
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text", "text": text},
                        {
                            "type": "function_call",
                            "name": "read_file",
                            "call_id": call_id,
                            "arguments": {"path": "README.md"},
                        },
                    ],
                },
                {"role": "tool", "contents": [{"type": "function_result", "call_id": call_id, "result": "contents"}]},
            ]
        )
    # The final answer showed as a segment first; its final op only completes it.
    journal.record(TranscriptAssistantOp("Final answer.", segment_id="answer"))
    journal.record(TranscriptAssistantOp("Final answer.", final=True))
    messages.append({"role": "assistant", "contents": [{"type": "text", "text": "Final answer."}]})
    replay = project_agent_transcript(
        {
            "meta": {"runner": "kernel", "status": "completed", "agent_display_name": "QA"},
            "state": {"messages": messages},
        }
    )
    assert replay is not None
    surface = AgentTranscriptSurface(
        journal,
        persisted_replay=replay if archived else None,
        fallback_final_text="Final answer.",
    )
    async with LocalizedWidgetApp(lambda: surface).run_test() as pilot:
        await wait_for(lambda: len(surface.query(AgentMessage)) == 3, pilot=pilot)
        assert [message._text for message in surface.query(UserMessage)] == ["Review [literal] input"]
        assert [message.text for message in surface.query(AgentMessage)] == [
            "Inspection 0.",
            "Inspection 1.",
            "Final answer.",
        ]
        assert [type(child) for child in surface.direct_children()] == [
            UserMessage,
            AgentMessage,
            ToolGroup,
            AgentMessage,
            ToolGroup,
            AgentMessage,
        ]
        assert all(group.all_complete for group in surface.query(ToolGroup))


@pytest.mark.parametrize("cancelled", [False, True], ids=["error", "interrupted"])
async def test_terminal_status_settles_tools_and_rejects_late_activity(cancelled: bool) -> None:
    journal = AgentTranscriptJournal()
    surface = AgentTranscriptSurface(journal)
    status_type = InterruptedMessage if cancelled else ErrorMessage
    async with LocalizedWidgetApp(lambda: surface).run_test() as pilot:
        journal.record(TranscriptUserOp("Input"))
        journal.record(TranscriptToolStartOp("pending", "read_file", KIND_FILESYSTEM_READ))
        reason = INTERRUPTED_REASON_PLAIN_MESSAGE.bind(reason=DisplayBlock("Cancelled [literal]"))
        journal.record(TranscriptInterruptedOp(reason) if cancelled else TranscriptErrorOp("Failed [literal]"))
        journal.record(TranscriptToolStartOp("late", "read_file", KIND_FILESYSTEM_READ))
        await wait_for(lambda: not surface._pending and not surface._draining, pilot=pilot)
        group = surface.query_one(ToolGroup)
        assert group.all_complete and "late" not in group._tool_records
        assert not surface.query_one(status_type).query(Button)

        journal.record(TranscriptResumedOp())
        journal.record(TranscriptAssistantOp("Recovered", final=True))
        if cancelled:
            # Cancellation is terminal; retries of failures retain the same journal.
            assert not any(isinstance(op, TranscriptResumedOp) for op in journal.operations)
        else:
            await wait_for(lambda: bool(surface.query(AgentMessage)) and not surface.query(ErrorMessage), pilot=pilot)
            assert surface.query_one(AgentMessage).text == "Recovered"
        reopened = AgentTranscriptSurface(journal)
        await surface.remove()
        await pilot.app.mount(reopened)
        await wait_for(lambda: bool(reopened.query(InterruptedMessage if cancelled else AgentMessage)), pilot=pilot)
        assert len(reopened.query(UserMessage)) == 1
        assert bool(reopened.query(status_type)) is cancelled


async def test_retry_and_context_warning_preserve_running_activity() -> None:
    journal = AgentTranscriptJournal()
    surface = AgentTranscriptSurface(journal)
    async with LocalizedWidgetApp(lambda: surface).run_test() as pilot:
        journal.record(TranscriptRetryOp("Transient [error]", attempt=1, max_attempts=3, delay_seconds=0))
        await wait_for(lambda: bool(surface.query(RetryMessage)), pilot=pilot)
        journal.record(TranscriptToolStartOp("read", "read_file", KIND_FILESYSTEM_READ))
        journal.record(TranscriptWarningOp(context_pressure_message("no_progress", source="sub_agent")))
        await wait_for(lambda: bool(surface.query(SystemMessage)), pilot=pilot)
        assert not surface.query(RetryMessage)
        assert surface.query_one(ToolGroup).is_tool_running("read")
        assert surface.query_one(SystemMessage).has_class("-warning")


async def test_compaction_retry_is_rendered_inside_its_card() -> None:
    journal = AgentTranscriptJournal()
    surface = AgentTranscriptSurface(journal)
    async with LocalizedWidgetApp(lambda: surface).run_test() as pilot:
        journal.record(TranscriptCompactionStartOp("compact"))
        for attempt in range(1, CompactionCard.QUIET_RETRY_NOTICES + 2):
            journal.record(TranscriptRetryOp("Rate limit [literal]", attempt, 7, 1, compaction=True))
        await wait_for(lambda: bool(surface.query(CompactionCard)), pilot=pilot)
        card = surface.query_one(CompactionCard)
        await wait_for(
            lambda: "Rate limit [literal]" in str(card.query_one("#compaction-retry-notice", Static).content),
            pilot=pilot,
        )
        assert not surface.query(RetryMessage)
        journal.record(TranscriptCompactionFinishedOp("compact", "ok"))
        await wait_for(lambda: card.status != "running", pilot=pilot)


async def test_replay_tail_adds_status_after_archived_content() -> None:
    replay = project_agent_transcript(
        {
            "meta": {"runner": "kernel", "status": "failed"},
            "state": {"messages": [{"role": "assistant", "contents": [{"type": "text", "text": "Saved prefix"}]}]},
        }
    )
    assert replay is not None
    journal = AgentTranscriptJournal()
    journal.record(TranscriptAssistantOp("Stale live prefix"))
    surface = AgentTranscriptSurface(
        journal, persisted_replay=replay, replay_tail=(TranscriptErrorOp("Saved failure"),)
    )
    async with LocalizedWidgetApp(lambda: surface).run_test() as pilot:
        await wait_for(lambda: bool(surface.query(ErrorMessage)), pilot=pilot)
        assert [type(child) for child in surface.direct_children()] == [AgentMessage, ErrorMessage]
        assert surface.query_one(AgentMessage).text == "Saved prefix"
        assert surface.query_one(ErrorMessage)._text == "Saved failure"
        assert journal.operations == ()


@pytest.mark.parametrize("with_warning", [False, True], ids=["adjacent-input", "warning-boundary"])
async def test_new_input_cleans_up_failed_prompt_without_crossing_warning(with_warning: bool) -> None:
    journal = AgentTranscriptJournal()
    surface = AgentTranscriptSurface(journal)
    async with LocalizedWidgetApp(lambda: surface).run_test() as pilot:
        journal.record(TranscriptUserOp("Failed input"))
        if with_warning:
            journal.record(TranscriptWarningOp(context_pressure_message("no_progress", source="sub_agent")))
        journal.record(TranscriptRetryOp("Temporary failure", attempt=1, max_attempts=3, delay_seconds=0))
        await wait_for(lambda: bool(surface.query(RetryMessage)), pilot=pilot)
        journal.record(TranscriptUserOp("New input"))
        await wait_for(lambda: not surface._pending and not surface._draining, pilot=pilot)

        expected_inputs = ["Failed input", "New input"] if with_warning else ["New input"]
        assert [message._text for message in surface.query(UserMessage)] == expected_inputs
        assert not surface.query(RetryMessage)
        assert bool(surface.query(SystemMessage)) is with_warning


@pytest.mark.parametrize("archived", [False, True], ids=["live", "archive"])
@pytest.mark.parametrize("locale, header", [("en", "⚠ Interrupted"), ("zh-Hans", "⚠ 已中断")])
async def test_interruption_header_uses_mount_locale(locale: str, header: str, archived: bool) -> None:
    journal = AgentTranscriptJournal()
    reason = "Cancelled [literal]"
    replay = None
    if archived:
        replay = project_agent_transcript(
            {
                "meta": {"runner": "kernel", "status": "cancelled"},
                "state": {
                    "messages": [
                        {
                            "role": "assistant",
                            "contents": [{"type": "text", "text": reason}],
                            "additional_properties": {
                                HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED,
                                "_interrupted_by": "system",
                            },
                        }
                    ]
                },
            }
        )
        assert replay is not None
    else:
        journal.record(TranscriptInterruptedOp(INTERRUPTED_REASON_PLAIN_MESSAGE.bind(reason=DisplayBlock(reason))))
    surface = AgentTranscriptSurface(journal, persisted_replay=replay)
    app = LocalizedWidgetApp(lambda: surface)
    app.locale_controller = LocaleController(Settings(locale=locale))
    async with app.run_test() as pilot:
        await wait_for(lambda: bool(surface.query(InterruptedMessage)), pilot=pilot)
        body = surface.query_one(InterruptedMessage).query_one(".status-body", Static).content
        assert isinstance(body, Text)
        assert body.plain == f"{header}\n{reason}"


def test_a_tool_start_records_when_it_started() -> None:
    before = time.monotonic()
    start = TranscriptToolStartOp("s1", "sleep", KIND_SLEEP, {"seconds": 60})

    assert before <= start.started_at <= time.monotonic()


def test_a_tool_status_records_when_it_came() -> None:
    before = time.monotonic()
    status = TranscriptToolStatusOp("s1", "interrupted")

    assert before <= status.stopped_at <= time.monotonic()


async def test_a_view_opened_during_a_sleep_counts_down_from_when_it_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """A view built later, as a sub-agent's details or a workflow node's transcript, counts from the recorded start."""
    clock = ModuleType("time")
    clock.monotonic = lambda: 1040.0  # type: ignore[attr-defined]
    monkeypatch.setattr(tool_call_module, "time", clock)
    monkeypatch.setattr(sleep_module, "time", clock)
    journal = AgentTranscriptJournal()
    journal.record(TranscriptToolStartOp("s1", "sleep", KIND_SLEEP, {"seconds": 60, "reason": "CI"}, started_at=1000.0))
    surface = AgentTranscriptSurface(journal, profile_name="Explore")

    async with LocalizedWidgetApp(lambda: surface).run_test(size=(100, 30)) as pilot:
        await wait_for(lambda: bool(surface.query(SleepToolCall)), pilot=pilot, description="the sleep's card")
        assert title_text(surface.query_one(ToolGroupTitle)) == "Sleeping: CI · 20s left"
        assert surface.query_one(SleepToolCall)._start_time == 1000.0


@pytest.mark.parametrize(
    ("stop", "title"),
    [
        ("interrupted", "▸ Slept 10s · 1 interrupted"),
        ("error", "▸ Slept 10s · 1 interrupted"),
        ("resumed", "▸ Slept 10s · 1 interrupted"),
        # A sub-agent cancelled, or ended before its calls' results came, settles them by status.
        ("status_interrupted", "▸ Slept 10s · 1 interrupted"),
        ("status_completed", "▸ Slept 10s"),
    ],
)
async def test_a_view_opened_later_counts_a_stopped_sleep_up_to_when_it_stopped(
    monkeypatch: pytest.MonkeyPatch, stop: str, title: str
) -> None:
    """A view built later from what the journal recorded counts how long a sleep ran up to its recorded stop."""
    clock = ModuleType("time")
    clock.monotonic = lambda: 1300.0  # type: ignore[attr-defined]
    monkeypatch.setattr(tool_call_module, "time", clock)
    monkeypatch.setattr(sleep_module, "time", clock)
    journal = AgentTranscriptJournal()
    journal.record(TranscriptToolStartOp("s1", "sleep", KIND_SLEEP, {"seconds": 600}, started_at=1000.0))
    if stop == "interrupted":
        reason = INTERRUPTED_REASON_PLAIN_MESSAGE.bind(reason=DisplayBlock("Cancelled"))
        journal.record(TranscriptInterruptedOp(reason, stopped_at=1010.0))
    elif stop == "error":
        journal.record(TranscriptErrorOp("Failed", stopped_at=1010.0))
    elif stop.startswith("status_"):
        # As a sub-agent's card ends: it settles the calls still running by status, then records the answer.
        journal.record(TranscriptToolStatusOp("s1", stop.removeprefix("status_"), stopped_at=1010.0))
        journal.record(TranscriptAssistantOp(text="Done.", final=True, streamed=False))
    else:
        # A paused attempt resumed: the retry stops the calls its failed attempt left running.
        journal.record(TranscriptPausedOp())
        journal.record(TranscriptResumedOp(stopped_at=1010.0))
    surface = AgentTranscriptSurface(journal, profile_name="Explore")

    async with LocalizedWidgetApp(lambda: surface).run_test(size=(100, 30)) as pilot:
        await wait_for(lambda: not surface._pending and not surface._draining, pilot=pilot)
        assert title_text(surface.query_one(ToolGroupTitle)) == title
