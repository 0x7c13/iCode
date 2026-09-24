# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the sub-agent detail transcript: final-result placement, transcript journal retention, persisted ACP transcript replay, and terminal audit fallbacks."""

from __future__ import annotations

import asyncio
import json

from textual.app import ComposeResult
from textual.widgets import Static

from chrys.app.tui.widgets.chat.agent_transcript_surface import (
    AgentTranscriptJournal,
    AgentTranscriptOp,
    AgentTranscriptSurface,
    TranscriptAssistantOp,
    TranscriptCompactionFinishedOp,
    TranscriptCompactionStartOp,
    TranscriptToolStartOp,
)
from chrys.app.tui.widgets.chat.messages import (
    AgentMessage,
    UserMessage,
)
from chrys.app.tui.widgets.chat.renderers.sub_agent import SubAgentToolCall
from chrys.app.tui.widgets.chat.tool_call import (
    BaseToolCard,
    ToolCardHeader,
    ToolGroup,
)
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.foundation.events.types import (
    ProvisionalPresentation,
)
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.tool_kinds import (
    KIND_FILESYSTEM_READ,
    KIND_FILESYSTEM_WRITE,
    KIND_SEARCH,
)
from chrys.foundation.tool_result_metadata import (
    TOOL_INTERRUPTED_METADATA_KEY,
)
from chrys.foundation.util.sub_agent_context import SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY
from chrys.service.session.sub_agent_logs import sessions_dir
from chrys.service.session.sub_agent_transcript import (
    PersistedSubAgentTranscript,
    load_persisted_sub_agent_transcript,
)
from tests.support.tui_helpers import (
    LocalizedApp,
    LocalizedWidgetApp,
    mount_sub_agent_detail,
)
from tests.support.waiting import wait_for


async def test_sub_agent_final_result_is_detail_only_and_detail_stays_live() -> None:
    """The compact summary omits final prose while a detail clone stays live."""

    persisted_loads: list[str] = []

    async def load_persisted(log_file: str) -> PersistedSubAgentTranscript | None:
        persisted_loads.append(log_file)
        return None

    class ToolApp(LocalizedApp):
        def compose(self) -> ComposeResult:
            card = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
            card.configure_transcript_loader(load_persisted)
            yield card

    final_result = "# Result\n\nThe complete answer"
    async with ToolApp().run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        card.add_assistant_message("I will inspect the tests.")
        await card.add_inner_tool_start(
            "inner1",
            "read_file",
            {"path": "tests/test_app.py"},
            tool_kind=KIND_FILESYSTEM_READ,
        )

        latest = card.query_one("#sa-activity-text", Static)
        assert latest.render().plain == "Reading tests/test_app.py"
        assert not list(card.query(AgentMessage))
        assert not list(card.query(ToolGroup))

        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: len(detail.query(AgentMessage)) == 1 and len(detail.query(ToolGroup)) == 1,
            pilot=pilot,
            description="live detail transcript snapshot",
        )
        detail_group = detail.query_one(ToolGroup)
        assert detail_group.collapsed is True
        detail_group.collapsed = False

        card.update_inner_tool_progress("inner1", ["Reading tests/test_app.py"])
        await pilot.pause()
        assert detail_group.collapsed is False
        card.complete_inner_tool("inner1", "done", 20)
        card.set_complete(
            final_result,
            50,
            metadata={
                "sub_agent_log_file": "Explore_a1b2c3d4e5f6.json",
                "sub_agent_audit_complete": True,
            },
        )
        assert card._transcript_journal.operations == ()

        await wait_for(
            lambda: len(detail.query(AgentMessage)) == 2 and detail_group.all_complete,
            pilot=pilot,
            description="live detail terminal update",
        )
        assert [message._text for message in detail.query(AgentMessage)] == [
            "I will inspect the tests.",
            final_result,
        ]
        assert detail_group.collapsed is True
        assert latest.render().plain == "Completed"
        header = card.query_one(ToolCardHeader)
        assert header.actions_visible is True
        assert header.copy_action_visible is True
        assert card.query_one("#sa-activity-indicator", ChrysLoadingIndicator).display is False
        assert persisted_loads == []


async def test_sub_agent_transcript_result_records_only_unpublished_final_segment() -> None:
    async with LocalizedWidgetApp(
        lambda: SubAgentToolCall("c1", "External", args={"prompt": "investigate"})
    ).run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        card.add_assistant_message("First published segment.")
        card.set_complete(
            "First published segment.\n\nFinal unpublished segment.",
            50,
            metadata={SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY: "Final unpublished segment."},
        )

        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: len(detail.query(AgentMessage)) == 2,
            pilot=pilot,
            description="deduplicated ACP transcript result",
        )

        assert [message._text for message in detail.query(AgentMessage)] == [
            "First published segment.",
            "Final unpublished segment.",
        ]
        assert card.result_text == "First published segment.\n\nFinal unpublished segment."


async def test_sub_agent_transcript_result_does_not_repeat_when_no_final_segment_remains() -> None:
    async with LocalizedWidgetApp(
        lambda: SubAgentToolCall("c1", "External", args={"prompt": "investigate"})
    ).run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        card.add_assistant_message("Already published segment.")
        card.set_complete(
            "Already published segment.",
            50,
            metadata={SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY: ""},
        )

        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: (
                len(detail.query(AgentMessage)) == 1
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="ACP transcript without duplicate terminal segment",
        )

        assert [message._text for message in detail.query(AgentMessage)] == ["Already published segment."]
        assert card.result_text == "Already published segment."


async def test_sub_agent_empty_override_keeps_parent_fallback_when_durable_replay_is_unavailable() -> None:
    async def load(_log_file: str) -> PersistedSubAgentTranscript | None:
        return None

    class ToolApp(LocalizedApp):
        def compose(self) -> ComposeResult:
            card = SubAgentToolCall("c1", "External", args={"prompt": "investigate"})
            card.configure_transcript_loader(load)
            card.add_assistant_message("Already published segment.")
            card.set_complete(
                "Already published segment.",
                50,
                metadata={
                    SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY: "",
                    "sub_agent_log_file": "External_a1b2c3d4e5f6.json",
                    "sub_agent_audit_complete": True,
                },
            )
            yield card

    async with ToolApp().run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        assert card._transcript_journal.operations == ()
        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: (
                len(detail.query(AgentMessage)) == 1
                and not detail._loading_persisted_replay
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="parent fallback after unavailable durable ACP replay",
        )

        assert [message._text for message in detail.query(AgentMessage)] == ["Already published segment."]


async def test_sub_agent_persisted_acp_uses_only_successful_attempt_for_parent_result(tmp_path) -> None:
    log_file = "Remote_a1b2c3d4e5f6.json"
    path = sessions_dir(tmp_path) / log_file
    path.parent.mkdir(parents=True)
    atomic_write_owner_only_text(
        path,
        json.dumps(
            {
                "meta": {
                    "record_type": "sub_agent_session",
                    "runner": "acp",
                    "agent_display_name": "Remote",
                    "status": "completed",
                },
                "acp_state": {
                    "successful_attempt": 2,
                    "translated_updates": [
                        {
                            "attempt": 1,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "messageId": "old",
                                "content": {"type": "text", "text": "Old failed attempt."},
                            },
                        },
                        {
                            "attempt": 2,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "messageId": "new-1",
                                "content": {"type": "text", "text": "New first."},
                            },
                        },
                        {
                            "attempt": 2,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "messageId": "new-2",
                                "content": {"type": "text", "text": "New second."},
                            },
                        },
                        {
                            "attempt": 2,
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "terminal",
                                "title": "Read",
                                "kind": "read",
                                "status": "completed",
                            },
                        },
                    ],
                },
            }
        ),
    )

    async def load(persisted_log_file: str) -> PersistedSubAgentTranscript | None:
        return await load_persisted_sub_agent_transcript(tmp_path, persisted_log_file)

    class ToolApp(LocalizedApp):
        def compose(self) -> ComposeResult:
            card = SubAgentToolCall("c1", "Remote", args={"prompt": "investigate"})
            card.configure_transcript_loader(load)
            card.set_complete(
                "New first.\n\nNew second.",
                50,
                metadata={
                    "sub_agent_log_file": log_file,
                    "sub_agent_audit_complete": True,
                },
            )
            yield card

    async with ToolApp().run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: (
                len(detail.query(AgentMessage)) == 3
                and len(detail.query(ToolGroup)) == 1
                and not detail._loading_persisted_replay
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="persisted ACP transcript without duplicate parent fallback",
        )

        assert [message._text for message in detail.query(AgentMessage)] == [
            "Old failed attempt.",
            "New first.",
            "New second.",
        ]


async def test_sub_agent_persisted_cancelled_acp_interrupts_unfinished_tool(tmp_path) -> None:
    log_file = "Remote_cancelled_a1b2c3d4e5f6.json"
    path = sessions_dir(tmp_path) / log_file
    path.parent.mkdir(parents=True)
    atomic_write_owner_only_text(
        path,
        json.dumps(
            {
                "meta": {
                    "record_type": "sub_agent_session",
                    "runner": "acp",
                    "agent_display_name": "Remote",
                    "status": "cancelled",
                },
                "acp_state": {
                    "translated_updates": [
                        {
                            "attempt": 1,
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "unfinished",
                                "title": "Read",
                                "kind": "read",
                                "status": "in_progress",
                            },
                        }
                    ]
                },
            }
        ),
    )

    async def load(persisted_log_file: str) -> PersistedSubAgentTranscript | None:
        return await load_persisted_sub_agent_transcript(tmp_path, persisted_log_file)

    class ToolApp(LocalizedApp):
        def compose(self) -> ComposeResult:
            card = SubAgentToolCall("c1", "Remote", args={"prompt": "investigate"})
            card.configure_transcript_loader(load)
            card.set_complete(
                "cancelled",
                50,
                metadata={
                    TOOL_INTERRUPTED_METADATA_KEY: True,
                    "sub_agent_log_file": log_file,
                    "sub_agent_audit_complete": True,
                },
            )
            yield card

    async with ToolApp().run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: (
                len(detail.query(ToolGroup)) == 1
                and not detail._loading_persisted_replay
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="persisted cancelled ACP tool replay",
        )

        record = next(iter(detail.query_one(ToolGroup)._tool_records.values()))
        assert record.status == "error"
        assert record.canonical_status == "interrupted"


async def test_sub_agent_surface_preserves_message_tool_boundaries_and_retracts_provisional_text() -> None:
    async with LocalizedWidgetApp(
        lambda: SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    ).run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        surface = await mount_sub_agent_detail(card, pilot)
        card.add_assistant_message("First plan")
        await card.add_inner_tool_start("inner-1", "read_file", {"path": "a.py"})
        card.complete_inner_tool("inner-1", "ok", 10)
        card.add_assistant_message("Second plan")
        await card.add_inner_tool_start("inner-2", "grep", {"pattern": "needle"})

        await wait_for(
            lambda: len(surface.direct_children()) == 5,
            pilot=pilot,
            description="sub-agent message and tool boundaries",
        )
        assert [type(widget) for widget in surface.direct_children()] == [
            UserMessage,
            AgentMessage,
            ToolGroup,
            AgentMessage,
            ToolGroup,
        ]

        card.add_assistant_message(
            "Discard me",
            presentation=ProvisionalPresentation("attempt-1", "segment-1"),
        )
        await wait_for(
            lambda: any(message._text == "Discard me" for message in surface.query(AgentMessage)),
            pilot=pilot,
            description="provisional sub-agent message",
        )
        card.reject_presentation_attempt("attempt-1")
        await wait_for(
            lambda: not any(message._text == "Discard me" for message in surface.query(AgentMessage)),
            pilot=pilot,
            description="retracted sub-agent message",
        )


async def test_sub_agent_compact_activity_keeps_only_latest_assistant_or_tool() -> None:
    async with LocalizedWidgetApp(
        lambda: SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    ).run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        latest = card.query_one("#sa-activity-text", Static)

        card.add_assistant_message("First plan")
        assert latest.render().plain == "First plan"

        await card.add_inner_tool_start(
            "inner-1",
            "grep",
            {"pattern": "needle"},
            tool_kind=KIND_SEARCH,
        )
        assert latest.render().plain == "Searching for needle"

        card.complete_inner_tool("inner-1", "ok", 10)
        assert latest.render().plain == "Searching for needle"

        card.add_assistant_message("Second plan")
        assert latest.render().plain == "Second plan"
        assert not list(card.query(AgentMessage))
        assert not list(card.query(ToolGroup))


async def test_sub_agent_parent_completion_settles_running_nested_detail() -> None:
    async with LocalizedWidgetApp(
        lambda: SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    ).run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        await card.add_inner_tool_start(
            "inner-1",
            "grep",
            {"pattern": "needle"},
            tool_kind=KIND_SEARCH,
        )
        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: len(detail.query(ToolGroup)) == 1,
            pilot=pilot,
            description="running nested tool detail group",
        )
        group = detail.query_one(ToolGroup)
        group.collapsed = False
        await wait_for(
            lambda: isinstance(group.get_tool("inner-1"), BaseToolCard),
            pilot=pilot,
            description="running nested tool card",
        )
        nested = group.get_tool("inner-1")
        assert isinstance(nested, BaseToolCard)
        assert nested.status == "running"

        card.set_complete("Finished without a nested result event", 50)

        await wait_for(
            lambda: group.all_complete and nested.status == "complete",
            pilot=pilot,
            description="parent completion settled nested tool card",
        )
        assert card.query_one("#sa-activity-text", Static).render().plain == "Completed"
        assert card.query_one("#sa-activity-indicator", ChrysLoadingIndicator).display is False


async def test_sub_agent_detail_replays_persisted_transcript_without_duplicate_final() -> None:
    loaded: list[str] = []
    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "investigate"}]},
        {
            "role": "assistant",
            "contents": [
                {"type": "text", "text": "I will inspect."},
                {
                    "type": "function_call",
                    "name": "read_file",
                    "call_id": "inner-1",
                    "arguments": {"path": "README.md"},
                },
            ],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "inner-1", "result": "ok"}]},
        {"role": "assistant", "contents": [{"type": "text", "text": "Persisted final."}]},
    ]

    async def load(log_file: str) -> PersistedSubAgentTranscript:
        loaded.append(log_file)
        return PersistedSubAgentTranscript(
            messages=messages,
            profile_name="Explore",
            includes_final_message=True,
        )

    class ToolApp(LocalizedApp):
        def compose(self) -> ComposeResult:
            card = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
            card.configure_transcript_loader(load)
            card.set_complete(
                "Persisted final.",
                50,
                metadata={
                    "sub_agent_log_file": "Explore_a1b2c3d4e5f6.json",
                    "sub_agent_audit_complete": True,
                },
            )
            yield card

    async with ToolApp().run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        header = card.query_one(ToolCardHeader)
        assert header.actions_visible is True
        assert header.copy_action_visible is True
        detail = card._tool_view_output_widgets()[0]
        assert isinstance(detail, AgentTranscriptSurface)
        await pilot.app.mount(detail)
        await wait_for(
            lambda: len(detail.query(AgentMessage)) == 2 and len(detail.query(ToolGroup)) == 1,
            pilot=pilot,
            description="persisted sub-agent detail replay",
        )

        assert loaded == ["Explore_a1b2c3d4e5f6.json"]
        assert [message._text for message in detail.query(AgentMessage)] == [
            "I will inspect.",
            "Persisted final.",
        ]
        assert [message._text for message in detail.query(UserMessage)] == ["investigate"]
        assert detail.query_one(ToolGroup).collapsed is True


async def test_sub_agent_detail_stops_pending_tool_mount_when_pruning() -> None:
    """A scheduled drain must stop when its transcript surface starts pruning."""
    entered = asyncio.Event()
    release = asyncio.Event()
    journal = AgentTranscriptJournal()

    class BlockingSurface(AgentTranscriptSurface):
        async def _apply(self, operation: AgentTranscriptOp) -> None:
            if isinstance(operation, TranscriptAssistantOp) and operation.text == "blocked presentation":
                entered.set()
                await asyncio.wait_for(release.wait(), timeout=5)
            await super()._apply(operation)

    surface = BlockingSurface(journal)

    async with LocalizedWidgetApp(lambda: surface).run_test() as pilot:
        journal.record(TranscriptAssistantOp("blocked presentation"))
        journal.record(
            TranscriptToolStartOp(
                "inner-write",
                "write_file",
                KIND_FILESYSTEM_WRITE,
                {"path": "generated.py", "content": "pass"},
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        removal = surface.remove()
        assert surface._pruning is True
        release.set()

        await removal
        await pilot.pause()

        assert surface.is_attached is False
        assert not surface._pending
        assert surface._draining is False


async def test_sub_agent_detail_keeps_parent_result_when_acp_replay_tail_is_truncated() -> None:
    async def load(_log_file: str) -> PersistedSubAgentTranscript:
        return PersistedSubAgentTranscript(
            messages=[{"role": "assistant", "contents": [{"type": "text", "text": "answer tail"}]}],
            profile_name="Remote",
            includes_final_message=True,
            requires_result_fingerprint=True,
        )

    class ToolApp(LocalizedApp):
        def compose(self) -> ComposeResult:
            card = SubAgentToolCall("c1", "Remote", args={"prompt": "investigate"})
            card.configure_transcript_loader(load)
            card.set_complete(
                "Authoritative beginning and answer tail",
                50,
                metadata={
                    "sub_agent_log_file": "Remote_truncated.json",
                    "sub_agent_audit_complete": True,
                },
            )
            yield card

    async with ToolApp().run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: (
                len(detail.query(AgentMessage)) == 2
                and not detail._loading_persisted_replay
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="authoritative parent result after truncated ACP replay",
        )

        assert [message._text for message in detail.query(AgentMessage)] == [
            "answer tail",
            "Authoritative beginning and answer tail",
        ]


async def test_sub_agent_detail_does_not_load_incomplete_audit_snapshot() -> None:
    loaded: list[str] = []

    async def load(log_file: str) -> PersistedSubAgentTranscript:
        loaded.append(log_file)
        return PersistedSubAgentTranscript(
            messages=[{"role": "assistant", "contents": [{"type": "text", "text": "Stale running text."}]}],
            profile_name="Explore",
            terminal_audit=False,
        )

    class ToolApp(LocalizedApp):
        def compose(self) -> ComposeResult:
            card = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
            card.configure_transcript_loader(load)
            card.set_complete(
                "Authoritative parent result.",
                50,
                metadata={"sub_agent_log_file": "Explore_incomplete.json"},
            )
            yield card

    async with ToolApp().run_test() as pilot:
        card = pilot.app.query_one(SubAgentToolCall)
        detail = card._tool_view_output_widgets()[0]
        assert isinstance(detail, AgentTranscriptSurface)
        await pilot.app.mount(detail)
        await wait_for(
            lambda: any(message._text == "Authoritative parent result." for message in detail.query(AgentMessage)),
            pilot=pilot,
            description="authoritative parent fallback rendered",
        )

        assert loaded == ["Explore_incomplete.json"]
        assert [message._text for message in detail.query(AgentMessage)] == ["Authoritative parent result."]


async def test_running_sub_agent_detail_stays_live_when_terminal_audit_is_already_readable() -> None:
    loaded: list[str] = []

    async def load(log_file: str) -> PersistedSubAgentTranscript:
        loaded.append(log_file)
        return PersistedSubAgentTranscript(
            messages=[{"role": "assistant", "contents": [{"type": "text", "text": "Persisted partial work."}]}],
            profile_name="Explore Agent",
            terminal_audit=True,
        )

    card = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    card.configure_transcript_loader(load)
    card.claim_invocation("invocation", "Explore Agent", "Explore_terminal.json")

    async with LocalizedWidgetApp(lambda: card).run_test() as pilot:
        detail = await mount_sub_agent_detail(card, pilot)
        card.add_assistant_message("Live work after opening.")
        card.set_error("Error: failed after the detail modal opened")
        await wait_for(
            lambda: (
                len(detail.query(AgentMessage)) == 2
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="live sub-agent terminal result drained",
        )

        assert loaded == []
        assert [message._text for message in detail.query(AgentMessage)] == [
            "Live work after opening.",
            "Error: failed after the detail modal opened",
        ]


async def test_running_sub_agent_detail_captures_completion_before_surface_mount() -> None:
    card = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    card.claim_invocation("invocation", "Explore Agent", "Explore_terminal.json")

    async with LocalizedWidgetApp(lambda: card).run_test() as pilot:
        detail = card._tool_view_output_widgets()[0]
        assert isinstance(detail, AgentTranscriptSurface)
        card.set_complete(
            "Completed before the detail surface mounted.",
            50,
            metadata={
                "sub_agent_log_file": "Explore_terminal.json",
                "sub_agent_audit_complete": True,
            },
        )
        await pilot.app.mount(detail)
        await wait_for(
            lambda: (
                len(detail.query(AgentMessage)) == 1
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="pre-mount sub-agent completion drained",
        )

        assert [message._text for message in detail.query(AgentMessage)] == [
            "Completed before the detail surface mounted."
        ]


async def test_terminal_sub_agent_keeps_bounded_fallback_with_display_name() -> None:
    card = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    card.claim_invocation("invocation", "Explore Agent")
    card.add_assistant_message("working", profile_name="explore_agent")
    card.set_complete("final answer", metadata={"sub_agent_log_file": "incomplete.json"})

    assert card._transcript_profile_name == "Explore Agent"
    assert card._transcript_journal.operations == (
        TranscriptAssistantOp("working"),
        TranscriptAssistantOp("final answer", final=True),
    )

    async with LocalizedWidgetApp(lambda: card).run_test() as pilot:
        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: len(detail.query(AgentMessage)) == 2,
            pilot=pilot,
            description="terminal sub-agent final result fallback",
        )

        messages = list(detail.query(AgentMessage))
        assert [message._text for message in messages] == ["working", "final answer"]
        assert {message._profile_name for message in messages} == {"Explore Agent"}
        assert [item._text for item in detail.query(UserMessage)] == ["investigate"]


def test_terminal_sub_agent_journal_is_bounded_and_rejects_late_events() -> None:
    journal = AgentTranscriptJournal()
    oversized = "x" * 20_000
    for _ in range(300):
        journal.record(TranscriptAssistantOp(oversized))

    journal.finalize_retention(durable_replay_available=False)
    retained = journal.operations
    journal.record(TranscriptAssistantOp("late"))

    assert len(retained) == 256
    assert all(isinstance(operation, TranscriptAssistantOp) for operation in retained)
    assert max(len(operation.text) for operation in retained if isinstance(operation, TranscriptAssistantOp)) < len(
        oversized
    )
    assert journal.operations == retained


async def test_terminal_audit_from_invocation_start_lazily_releases_cancelled_card_fallback() -> None:
    loaded: list[str] = []

    async def load(log_file: str) -> PersistedSubAgentTranscript:
        loaded.append(log_file)
        return PersistedSubAgentTranscript(
            messages=[{"role": "assistant", "contents": [{"type": "text", "text": "Persisted work."}]}],
            profile_name="Explore Agent",
            terminal_audit=True,
        )

    card = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    card.configure_transcript_loader(load)
    card.claim_invocation("invocation", "Explore Agent", "Explore_terminal.json")
    for index in range(300):
        card.add_assistant_message(f"message-{index}")
    card.set_cascade_aborted()

    assert card._transcript_journal.operations

    async with LocalizedWidgetApp(lambda: card).run_test() as pilot:
        detail = await mount_sub_agent_detail(card, pilot)
        await wait_for(
            lambda: (
                len(detail.query(AgentMessage)) == 2
                and not detail._loading_persisted_replay
                and not detail._pending
                and not detail._draining
                and not detail._drain_scheduled
            ),
            pilot=pilot,
            description="terminal audit loaded through invocation-start reference",
        )

        assert loaded == ["Explore_terminal.json"]
        assert card._transcript_journal.operations == ()
        assert [message._text for message in detail.query(AgentMessage)] == [
            "Persisted work.",
            "Error: cancelled (global interrupt)",
        ]


def test_terminal_sub_agent_journal_retains_compaction_start_at_tail_boundary() -> None:
    journal = AgentTranscriptJournal()
    journal.record(TranscriptCompactionStartOp("compaction-1"))
    journal.record(TranscriptCompactionFinishedOp("compaction-1", "ok"))
    for index in range(255):
        journal.record(TranscriptAssistantOp(f"message-{index}"))

    journal.finalize_retention(durable_replay_available=False)

    assert journal.operations[:2] == (
        TranscriptCompactionStartOp("compaction-1"),
        TranscriptCompactionFinishedOp("compaction-1", "ok"),
    )


async def test_sub_agent_rejected_lazy_completion_bounds_terminal_journal() -> None:
    async with LocalizedWidgetApp(
        lambda: SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    ).run_test() as pilot:
        tc = pilot.app.query_one(SubAgentToolCall)
        await tc.add_inner_tool_start("inner1", "read_file", {"path": "a.py"}, tool_kind=KIND_FILESYSTEM_READ)
        result = "# Rejected\n\n" + "\n".join(f"- reason {idx}" for idx in range(5000))
        tc.set_complete(
            result,
            approval="user_rejected",
            lazy=True,
        )
        await pilot.pause()

        assert tc.status == "rejected"
        assert not tc._inner_tools
        final_operations = [
            operation
            for operation in tc._transcript_journal.operations
            if isinstance(operation, TranscriptAssistantOp) and operation.final
        ]
        assert len(final_operations) == 1
        assert len(final_operations[0].text) < len(result)
