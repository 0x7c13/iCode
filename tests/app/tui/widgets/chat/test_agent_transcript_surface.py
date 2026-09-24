# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared transcript contracts independent of any workflow controller or runner."""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.widgets import Button, Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.util.context_pressure import context_pressure_message
from chrys.app.tui.widgets.chat.agent_transcript_surface import (
    AgentTranscriptJournal,
    AgentTranscriptSurface,
    TranscriptAssistantOp,
    TranscriptCompactionFinishedOp,
    TranscriptCompactionStartOp,
    TranscriptErrorOp,
    TranscriptInterruptedOp,
    TranscriptResumedOp,
    TranscriptRetryOp,
    TranscriptToolResultOp,
    TranscriptToolStartOp,
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
from chrys.app.tui.widgets.chat.tool_call import ToolGroup
from chrys.foundation.config.settings import Settings
from chrys.foundation.i18n import DisplayBlock
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_READ
from chrys.service.session.sub_agent_transcript import project_agent_transcript
from tests.support.tui_helpers import LocalizedWidgetApp
from tests.support.waiting import wait_for


@pytest.mark.parametrize("archived", [False, True], ids=["live", "archive"])
async def test_input_intermediate_tools_and_final_keep_their_order(archived: bool) -> None:
    journal = AgentTranscriptJournal()
    journal.record(TranscriptUserOp("Review [literal] input"))
    messages = [{"role": "user", "contents": [{"type": "text", "text": "Review [literal] input"}]}]
    for index in range(2):
        call_id, text = f"read-{index}", f"Inspection {index}."
        journal.record(TranscriptAssistantOp(text))
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
        assert [message._text for message in surface.query(AgentMessage)] == [
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
            assert surface.query_one(AgentMessage)._text == "Recovered"
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
        assert surface.query_one(AgentMessage)._text == "Saved prefix"
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
