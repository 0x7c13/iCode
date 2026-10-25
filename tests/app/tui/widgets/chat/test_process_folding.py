# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Completed process folding preserves user boundaries, copy and logical selection order."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from chrys.app.tui.widgets.chat.messages import AgentMessage, InterruptedMessage, ThinkingMessage, UserMessage
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.process_block import ProcessBlock
from chrys.app.tui.widgets.chat.process_controller import process_entry
from chrys.app.tui.widgets.chat.process_model import ProcessEntry, plan_process_segments
from chrys.app.tui.widgets.chat.replay_mount import ReplayMountController, estimated_rows
from chrys.app.tui.widgets.chat.selection_controller import ChatSelectionController
from chrys.app.tui.widgets.chat.tool_call import ToolGroup
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Content, Message
from chrys.service.context.providers.history import CompressedBlock
from tests.support.tui_helpers import ChatPanelApp, click_when_settled, rich_plain
from tests.support.waiting import wait_for


def _history() -> list[dict[str, Any]]:
    return [
        Message("user", [Content.from_text("prompt")]).to_dict(),
        Message(
            "assistant",
            [
                Content.from_text_reasoning(text="Look."),
                Content.from_text("Checking."),
                Content.from_function_call(call_id="c1", name="read_file", arguments="{}"),
            ],
        ).to_dict(),
        Message("tool", [Content.from_function_result(call_id="c1", result="ok")]).to_dict(),
        Message(
            "user", [Content.from_text("also this")], additional_properties={HistoryMarkerKind.INJECTED_KEY: True}
        ).to_dict(),
        Message(
            "assistant",
            [
                Content.from_text("One more."),
                Content.from_function_call(call_id="c2", name="read_file", arguments="{}"),
            ],
        ).to_dict(),
        Message("tool", [Content.from_function_result(call_id="c2", result="ok")]).to_dict(),
        Message("assistant", [Content.from_text("Done.")]).to_dict(),
    ]


def _archive(history: list[dict[str, Any]]) -> tuple[dict[str, Any], CompressedBlock]:
    block = CompressedBlock(compressed_context_id="archive", messages=[Message.from_dict(item) for item in history])
    summary = Message(
        "assistant",
        [Content.from_text("Condensed")],
        additional_properties={HistoryMarkerKind.KEY: HistoryMarkerKind.SUMMARY, "_block_id": "archive"},
    ).to_dict()
    return summary, block


def _terminal_status(marker: str, status_code: str | None) -> dict[str, Any]:
    return Message(
        "assistant",
        [Content.from_text("Stopped before completing the reply")],
        additional_properties={
            HistoryMarkerKind.KEY: marker,
            "_interrupted_by": "error",
            **({HistoryMarkerKind.STATUS_CODE_KEY: status_code} if status_code is not None else {}),
            "_invocation_ids": ["child"],
        },
    ).to_dict()


@pytest.mark.parametrize("inside", [True, False], ids=["inside-archive", "after-archive"])
@pytest.mark.parametrize(
    ("marker", "status_code"),
    [
        (HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.STATUS_EXECUTION_FAILED),
        (HistoryMarkerKind.AWAITING_SUB_AGENTS, HistoryMarkerKind.STATUS_AWAITING_SUB_AGENTS),
        (HistoryMarkerKind.AWAITING_SUB_AGENTS, None),
    ],
)
async def test_compressed_terminal_status_prevents_folding(inside: bool, marker: str, status_code: str | None) -> None:
    history = _history()
    history[-1] = Message("assistant", [Content.from_text("Partial answer")]).to_dict()
    status = _terminal_status(marker, status_code)
    summary, block = _archive([*history, status] if inside else history)
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history([summary] if inside else [summary, status], compressed_blocks={"archive": block})
        await panel.wait_replay_complete()
        assert not panel.query(ProcessBlock)
        assert all(message.display for message in panel.query(AgentMessage))
        if inside:
            assert not panel.query(InterruptedMessage)
            assert all(not message.is_status for message in panel.query(AgentMessage))
            assert panel.get_agent_responses() == [
                ("Agent", "Checking."),
                ("Agent", "One more."),
                ("Agent", "Partial answer"),
            ]
            assert panel.get_all_messages() == [
                ("You", "prompt"),
                ("Agent", "Checking."),
                ("You", "also this"),
                ("Agent", "One more."),
                ("Agent", "Partial answer"),
            ]


@pytest.mark.parametrize("answer_inside", [True, False], ids=["success-inside-archive", "success-after-archive"])
async def test_compressed_failure_followed_by_success_still_folds(answer_inside: bool) -> None:
    history = _history()
    final = history.pop()
    history.append(Message("assistant", [Content.from_text("Partial answer")]).to_dict())
    history.append(_terminal_status(HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.STATUS_EXECUTION_FAILED))
    summary, block = _archive([*history, final] if answer_inside else history)
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(
            [summary] if answer_inside else [summary, final], compressed_blocks={"archive": block}
        )
        await panel.wait_replay_complete()
        assert _summary(panel) == [(0, 0, True, True), (1, 0, True, True)]
        assert not panel.query(InterruptedMessage)
        assert panel.get_agent_responses() == [
            ("Agent", "Checking."),
            ("Agent", "One more."),
            ("Agent", "Partial answer"),
            ("Agent", "Done."),
        ]
        assert panel.toggle_fold_all() is False
        assert all(message.display for message in panel.query(AgentMessage))
        await wait_for(
            lambda: all(message.query(VirtualizedMarkdown) for message in panel.query(ThinkingMessage)),
            pilot=pilot,
            description="expanded archived thinking body restored",
        )


@pytest.mark.parametrize("thinking", [False, True])
async def test_omitted_status_never_creates_an_empty_process_or_wraps_lone_thinking(thinking: bool) -> None:
    history = [Message("user", [Content.from_text("prompt")]).to_dict()]
    if thinking:
        history.append(Message("assistant", [Content.from_text_reasoning(text="Evidence")]).to_dict())
    history.extend(
        [
            _terminal_status(HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.STATUS_EXECUTION_FAILED),
            Message("assistant", [Content.from_text("Done.")]).to_dict(),
        ]
    )
    summary, block = _archive(history)
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history([summary], compressed_blocks={"archive": block})
        await panel.wait_replay_complete()
        assert not panel.query(ProcessBlock)
        assert not panel.query(InterruptedMessage)
        assert len(panel.query(ThinkingMessage)) == int(thinking)
        assert all(message.collapsed and message.display for message in panel.query(ThinkingMessage))
        assert panel.get_all_messages() == [("You", "prompt"), ("Agent", "Done.")]


@pytest.mark.parametrize("boundary", ["thinking", "injection"])
async def test_omitted_failure_after_a_non_answer_still_ends_the_turn(boundary: str) -> None:
    history = _history()
    if boundary == "thinking":
        history.append(Message("assistant", [Content.from_text_reasoning(text="Checking again")]).to_dict())
    else:
        history.append(
            Message(
                "user",
                [Content.from_text("one last check")],
                additional_properties={HistoryMarkerKind.INJECTED_KEY: True},
            ).to_dict()
        )
    history.append(_terminal_status(HistoryMarkerKind.INTERRUPTED, None))
    summary, block = _archive(history)
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history([summary], compressed_blocks={"archive": block})
        await panel.wait_replay_complete()
        assert not panel.query(ProcessBlock)
        assert all(message.display for message in panel.query(AgentMessage))


async def test_omitted_failure_survives_setting_changes_until_a_later_success() -> None:
    enabled = False
    history = _history()
    history.append(_terminal_status(HistoryMarkerKind.INTERRUPTED, None))
    summary, block = _archive(history)
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: enabled).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history([summary], compressed_blocks={"archive": block})
        await panel.wait_replay_complete()
        enabled = True
        await panel.add_agent_message("")
        assert not panel.query(ProcessBlock)
        await panel.add_agent_message("Recovered.")
        assert _summary(panel) == [(0, 0, True, True), (1, 0, True, True)]
        assert panel.get_agent_responses()[-1] == ("Agent", "Recovered.")


async def _live(panel: ChatPanel) -> None:
    await panel.add_user_message("prompt")
    await panel.add_thinking_delta("r1", "reasoning", "Look.")
    panel.finish_thinking_segment("r1")
    for call_id, text in [("c1", "Checking."), ("c2", "One more.")]:
        if call_id == "c2":
            await panel.add_user_message("also this", is_injection=True)
        await panel.add_answer_delta(call_id, text)
        panel.finish_answer_segment(call_id, completed=True)
        await panel.add_tool_start(call_id, "read_file", "filesystem.read", args={})
        await panel.add_tool_result(call_id, "read_file", "ok")
        await panel.add_system("notice")
    await panel.add_answer_delta("final", "Done.")
    panel.finish_answer_segment("final", completed=True)
    await panel.add_agent_message("Done.")


def _summary(panel: ChatPanel) -> list[tuple[int, int, bool, bool]]:
    return [
        (block.segment_index, block.tool_count, block.is_compressed or block.tool_count == 0, block.collapsed)
        for block in panel.query(ProcessBlock)
    ]


async def test_live_and_replay_share_segments_counts_and_default_fold_state() -> None:
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await _live(panel)
        await pilot.pause()
        expected = _summary(panel)
        assert expected == [(0, 1, False, True), (1, 1, False, True)]
        copied = panel.get_all_messages()
        assert copied == [
            ("You", "prompt"),
            ("Agent", "Checking."),
            ("You", "also this"),
            ("Agent", "One more."),
            ("Agent", "Done."),
        ]
        assert all(message.display for message in panel.query(UserMessage))
        assert [message.text for message in panel.query(AgentMessage) if message.display] == ["Done."]
        await wait_for(
            lambda: not panel.query(ThinkingMessage).first().query(VirtualizedMarkdown),
            pilot=pilot,
            description="folded thinking body removed",
        )
        await panel.clear()
        await panel.replay_history(_history())
        await panel.wait_replay_complete()
        assert _summary(panel) == expected
        assert panel.get_all_messages() == copied


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("failed", [False, True])
async def test_archived_turn_does_not_build_omitted_local_tool_groups(fold: bool, failed: bool) -> None:
    history = _history()
    if failed:
        history[-1] = Message(
            "assistant",
            [Content.from_text("Failed")],
            additional_properties={HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED, "_interrupted_by": "error"},
        ).to_dict()
    archived = CompressedBlock(compressed_context_id="archive", messages=[Message.from_dict(item) for item in history])
    summary = Message(
        "assistant",
        [Content.from_text("Condensed")],
        additional_properties={HistoryMarkerKind.KEY: HistoryMarkerKind.SUMMARY, "_block_id": "archive"},
    ).to_dict()
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: fold).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history([summary], compressed_blocks={"archive": archived})
        await panel.wait_replay_complete()
        assert not panel.query(ToolGroup)
        assert all(block.tool_count == 0 for block in panel.query(ProcessBlock))
        if failed:
            assert not panel.query(ProcessBlock)
        if fold and not failed:
            assert _summary(panel) == [(0, 0, True, True), (1, 0, True, True)]


@pytest.mark.parametrize("retained_kind", ["hosted", "informational"])
async def test_mixed_compressed_turn_always_uses_activity_title(retained_kind: str) -> None:
    history = _history()
    if retained_kind == "hosted":
        retained = [
            Content.from_hosted_tool_call(
                "search", tool_name="web_search", status="completed", hosted_family="search", hosted_provider="openai"
            ).to_dict(),
            Content.from_hosted_tool_result(
                "search",
                tool_name="web_search",
                result="found",
                status="completed",
                hosted_family="search",
                hosted_provider="openai",
            ).to_dict(),
        ]
    else:
        call = Content.from_function_call(call_id="server", name="server_task", arguments="{}").to_dict()
        call["informational_only"] = True
        retained = [call]
    # Both process segments retain a call, while their local calls stay omitted.
    history[1]["contents"].extend(retained)
    history[4]["contents"].extend(retained)
    archived = CompressedBlock(compressed_context_id="mixed", messages=[Message.from_dict(item) for item in history])
    summary = Message(
        "assistant",
        [Content.from_text("Condensed")],
        additional_properties={HistoryMarkerKind.KEY: HistoryMarkerKind.SUMMARY, "_block_id": "mixed"},
    ).to_dict()
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history([summary], compressed_blocks={"mixed": archived})
        await panel.wait_replay_complete()
        blocks = list(panel.query(ProcessBlock))
        assert len(blocks) == 2
        assert all(block.tool_count == 1 and block.is_compressed for block in blocks)
        assert [rich_plain(block.render()).strip() for block in blocks] == ["▶ Activity", "▶ Activity"]
        assert panel.toggle_fold_all() is False
        assert [rich_plain(block.render()).strip() for block in blocks] == ["▼ Activity", "▼ Activity"]
        await wait_for(
            lambda: all(group._content_mounted and group._tools for group in panel.query(ToolGroup)),
            pilot=pilot,
            description="retained tool details mounted",
        )


def test_compressed_opener_marks_every_process_segment() -> None:
    segments = plan_process_segments(
        [
            ProcessEntry("user", compressed=True),
            ProcessEntry("tool", 1),
            ProcessEntry("injection"),
            ProcessEntry("tool", 3),
            ProcessEntry("answer"),
        ]
    )
    assert [segment.compressed for segment in segments] == [True, True]


@pytest.mark.parametrize("status_code", [None, HistoryMarkerKind.STATUS_AWAITING_SUB_AGENTS])
async def test_restored_awaiting_sub_agents_status_cannot_complete_a_turn(status_code: str | None) -> None:
    history = _history()
    history.append(
        Message(
            "assistant",
            [Content.from_text("Awaiting 2 sub-agents")],
            additional_properties={
                HistoryMarkerKind.KEY: HistoryMarkerKind.AWAITING_SUB_AGENTS,
                **({HistoryMarkerKind.STATUS_CODE_KEY: status_code} if status_code is not None else {}),
                "_invocation_ids": ["a", "b"],
            },
        ).to_dict()
    )
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(history)
        await panel.wait_replay_complete()
        status = panel.query(AgentMessage).last()
        assert status.is_status and not status.is_process_final
        assert process_entry(status).kind == "failure"
        assert not panel.query(ProcessBlock)
        assert all(group.display for group in panel.query(ToolGroup))


async def test_failed_header_mount_keeps_members_visible_and_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        mount = panel.mount_transcript_after

        async def fail_first_segment(anchor, block):
            assert all(member.display for member in block.members)
            if block.segment_index == 0:
                return False
            return await mount(anchor, block)

        monkeypatch.setattr(panel, "mount_transcript_after", fail_first_segment)
        await _live(panel)
        assert _summary(panel) == [(1, 1, False, True)]
        assert panel.query(ToolGroup).first().display
        existing = panel.query_one(ProcessBlock)
        monkeypatch.setattr(panel, "mount_transcript_after", mount)
        await panel._process_controller.complete_live()
        assert _summary(panel) == [(0, 1, False, True), (1, 1, False, True)]
        assert panel.query(ProcessBlock).last() is existing


async def test_unowned_tool_state_notifications_do_not_enter_batch_update() -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await _live(panel)
        group = panel.query(ToolGroup).first()
        with patch.object(pilot.app, "batch_update", wraps=pilot.app.batch_update) as batches:
            panel.on_tool_group_process_state_changed(ToolGroup.ProcessStateChanged(ToolGroup(live=False)))
            batches.assert_not_called()
            panel.on_tool_group_process_state_changed(ToolGroup.ProcessStateChanged(group))
            batches.assert_called_once()


async def test_tool_free_activity_uses_fallback_title_and_single_thinking_has_no_wrapper() -> None:
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_answer_delta("middle", "An intermediate update.")
        panel.finish_answer_segment("middle", completed=True)
        await panel.add_thinking_delta("r", "reasoning", "Thinking.")
        panel.finish_thinking_segment("r")
        await panel.add_answer_delta("final", "Done.")
        panel.finish_answer_segment("final", completed=True)
        await panel.add_agent_message("Done.")
        assert _summary(panel) == [(0, 0, True, True)]
        await panel.clear()
        await panel.add_user_message("prompt")
        await panel.add_thinking_delta("only", "reasoning", "Thinking.")
        panel.finish_thinking_segment("only")
        await panel.add_agent_message("Done.")
        assert not panel.query(ProcessBlock)
        assert panel.query_one(ThinkingMessage).collapsed


async def test_failure_prompt_cleanup_has_no_precreated_header() -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("failed prompt")
        await panel.add_error("failed")
        await panel.add_user_message("replacement")
        assert [message.text for message in panel.query(UserMessage)] == ["replacement"]
        assert not panel.query(ProcessBlock)


async def test_manual_expand_persists_and_fold_toggles_processes_with_their_contents() -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await _live(panel)
        block = panel.query(ProcessBlock).first()
        await click_when_settled(pilot, block)
        assert not block.collapsed
        await panel.add_user_message("next")
        assert not block.collapsed
        assert panel.toggle_fold_all()
        assert block.collapsed
        assert panel.toggle_fold_all() is False
        assert all(not item.collapsed for item in panel.query(ProcessBlock))
        assert all(group.display and not group.collapsed for group in panel.query(ToolGroup))


async def test_fold_can_open_existing_processes_after_automatic_folding_is_disabled() -> None:
    enabled = True
    async with ChatPanelApp(show_thinking=lambda: True, fold_finished_turns=lambda: enabled).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await _live(panel)
        enabled = False
        thinking = panel.query_one(ThinkingMessage)
        await wait_for(lambda: not thinking.query(VirtualizedMarkdown), pilot=pilot, description="thinking released")
        assert panel.toggle_fold_all() is False
        assert all(not block.collapsed for block in panel.query(ProcessBlock))
        assert all(member.display for block in panel.query(ProcessBlock) for member in block.members)
        await wait_for(
            lambda: bool(thinking.query(VirtualizedMarkdown)),
            pilot=pilot,
            description="keyboard expansion restores thinking",
        )
        assert not thinking.collapsed
        assert panel.toggle_fold_all() is True
        assert all(block.collapsed for block in panel.query(ProcessBlock))


@pytest.mark.parametrize("lock", [False, True], ids=["running-tool", "ask-user-lock"])
async def test_terminal_member_notification_retries_deferred_folding(lock: bool) -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_tool_start("c1", "read_file", "filesystem.read")
        group = panel.query_one(ToolGroup)
        if lock:
            await panel.add_tool_result("c1", "read_file", "ok")
            group.lock_collapse_for("question")
        await panel.add_agent_message("Done.")
        block = panel.query_one(ProcessBlock)
        assert not block.collapsed and group.display
        if lock:
            group.unlock_collapse_for("question")
        else:
            await panel.add_tool_result("c1", "read_file", "ok")
        await wait_for(lambda: block.collapsed, pilot=pilot, description="last blocker notification folds process")
        assert not group.display
        group.set_content_visible(True)
        assert not group.display


@pytest.mark.parametrize("lock", [False, True], ids=["running-tool", "ask-user-lock"])
async def test_fold_expansion_survives_the_last_blocker_finishing(lock: bool) -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await _live(panel)
        await panel.add_user_message("next")
        await panel.add_tool_start("pending", "read_file", "filesystem.read")
        group = panel.query(ToolGroup).last()
        if lock:
            await panel.add_tool_result("pending", "read_file", "ok")
            group.lock_collapse_for("question")
        await panel.add_agent_message("Done.")
        block = panel.query(ProcessBlock).last()
        assert block.blocked and block.requested_collapsed and not block.collapsed
        assert panel.toggle_fold_all() is False
        assert not block.requested_collapsed
        if lock:
            group.unlock_collapse_for("question")
        else:
            await panel.add_tool_result("pending", "read_file", "ok")
        await pilot.pause()
        assert not block.blocked and not block.collapsed and not block.requested_collapsed
        assert group.display
        await wait_for(
            lambda: all(item._content_mounted and item._tools for item in panel.query(ToolGroup)),
            pilot=pilot,
            description="expanded tool details mounted",
        )


async def test_fold_with_only_blocked_sections_preserves_pending_intent() -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_tool_start("pending", "read_file", "filesystem.read")
        await panel.add_tool_result("pending", "read_file", "ok")
        group = panel.query_one(ToolGroup)
        group.lock_collapse_for("question")
        await panel.add_agent_message("Done.")
        block = panel.query_one(ProcessBlock)
        assert block.requested_collapsed and not block.collapsed
        assert panel.toggle_fold_all() is False
        assert block.requested_collapsed and not block.collapsed and group.display
        group.unlock_collapse_for("question")
        await wait_for(lambda: block.collapsed, pilot=pilot, description="preserved pending fold applied")


async def test_sidecar_last_turn_and_disabled_setting_leave_process_visible() -> None:
    enabled = True
    async with ChatPanelApp(fold_finished_turns=lambda: enabled).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(_history(), recovered_from_sidecar=True)
        await panel.wait_replay_complete()
        assert not panel.query(ProcessBlock)
        await panel.clear()
        enabled = False
        await _live(panel)
        assert not panel.query(ProcessBlock)


@pytest.mark.parametrize("replay", [False, True], ids=["live-add", "replay-add"])
async def test_adding_tools_cannot_override_process_visibility(replay: bool) -> None:
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        group = ToolGroup(live=False)
        group.set_process_hidden(True)
        await panel.mount(group)
        if replay:
            await group.add_collapsed_replay_tool("c1", "read_file", "filesystem.read", result="ok", lazy=True)
        else:
            await group.add_tool("c1", "read_file", "filesystem.read")
        assert not group.display
        group.set_process_hidden(False)
        assert group.display


async def test_inserted_header_preserves_cross_entry_selection_and_prepend_anchor() -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await _live(panel)
        await pilot.pause()
        children = panel.transcript_content_children()
        stamps = [panel.transcript_seq(child) for child in children]
        assert None not in stamps
        assert stamps == sorted(stamps)
        first = panel.query(UserMessage).first().query_one(VirtualizedMarkdown)
        last = panel.query(AgentMessage).last().query_one(VirtualizedMarkdown)
        selection = ChatSelectionController(pilot.app.screen)
        start = selection._leaf_anchor(panel, first, None)
        end = selection._leaf_anchor(panel, last, None)
        assert start is not None and end is not None
        leaves = [leaf for leaf, _ in selection._iter_range(panel, start, end)]
        assert leaves[-1] is last
        user = panel.query(UserMessage).first()
        assert panel.transcript_entry_after(user) is panel.query(ProcessBlock).first()


def test_empty_segments_and_transient_notices_never_renumber_segments() -> None:
    entries = [
        ProcessEntry("user"),
        ProcessEntry("other"),
        ProcessEntry("injection"),
        ProcessEntry("retry"),
        ProcessEntry("tool", 2),
        ProcessEntry("other"),
        ProcessEntry("answer"),
    ]
    planned = plan_process_segments(entries)
    assert [(segment.index, segment.tool_count, segment.members) for segment in planned] == [(0, 0, ()), (1, 2, (3, 4))]


def test_hidden_members_use_no_tail_budget_and_do_not_exhaust_first_batch_limit() -> None:
    widgets = [AgentMessage("visible\n" * 40)]
    for _ in range(80):
        hidden = AgentMessage("hidden\n" * 1000)
        hidden.set_process_hidden(True)
        widgets.append(hidden)
    widgets.append(AgentMessage("final"))
    assert estimated_rows(widgets[1], 100) == 0
    start = ReplayMountController._batch_start(
        widgets, len(widgets), row_budget=40, min_entries=2, width=100, max_entries=128
    )
    assert start == 0


def test_foreground_budget_caps_visible_entries_and_total_hidden_entries() -> None:
    visible = [AgentMessage("short") for _ in range(40)]
    start = ReplayMountController._batch_start(
        visible, len(visible), row_budget=1000, min_entries=2, width=100, max_entries=128
    )
    assert len(visible) - start == 32
    hidden = [AgentMessage("hidden") for _ in range(300)]
    for widget in hidden:
        widget.set_process_hidden(True)
    start = ReplayMountController._batch_start(
        hidden, len(hidden), row_budget=1000, min_entries=2, width=100, max_entries=128
    )
    assert len(hidden) - start == 128


@pytest.mark.parametrize("ending", ["error", "interrupted", "empty"])
async def test_unsuccessful_or_empty_final_never_hides_the_last_process(ending: str) -> None:
    async with ChatPanelApp(fold_finished_turns=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_tool_start("c", "read_file", "filesystem.read")
        await panel.add_tool_result("c", "read_file", "ok")
        if ending == "error":
            await panel.add_error("failed")
        elif ending == "interrupted":
            await panel.add_interrupted()
        else:
            await panel.add_agent_message("")
        assert not panel.query(ProcessBlock)
        assert panel.query_one(ToolGroup).display
