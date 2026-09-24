# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for replay batch_id merge logic and _excluded message filtering.

Covers:
- Double-break fix: batch_id merge must walk back through tool-group entries
- _excluded messages filtered in load_session_raw
- Parallel tool calls grouped correctly during replay
- Intermediate text between batches doesn't break batch grouping
- _excluded messages don't appear as phantom tool groups in replay
"""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Content, Message
from chrys.service.state.store import JsonFileStateStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fc(call_id: str, name: str) -> dict[str, Any]:
    """Create a function_call content dict."""
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _fr(call_id: str, result: str = "ok") -> dict[str, Any]:
    """Create a function_result content dict."""
    return {"type": "function_result", "call_id": call_id, "result": result}


def _text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def _asst_msg(
    contents: list[dict],
    *,
    batch_id: int | None = None,
    intermediate_text: str | None = None,
    excluded: bool = False,
) -> dict[str, Any]:
    """Create an assistant message dict with optional metadata."""
    msg: dict[str, Any] = {"role": "assistant", "contents": contents}
    ap: dict[str, Any] = {}
    if batch_id is not None:
        ap["_batch_id"] = batch_id
    if intermediate_text is not None:
        ap["_intermediate_text"] = intermediate_text
    if excluded:
        ap["_excluded"] = True
    if ap:
        msg["additional_properties"] = ap
    return msg


def _tool_msg(contents: list[dict], *, excluded: bool = False) -> dict[str, Any]:
    """Create a tool-role message dict."""
    msg: dict[str, Any] = {"role": "tool", "contents": contents}
    if excluded:
        msg["additional_properties"] = {"_excluded": True}
    return msg


def _user_msg(text: str, *, has_group: bool = True) -> dict[str, Any]:
    """Create a user message dict."""
    msg: dict[str, Any] = {"role": "user", "contents": [_text(text)]}
    if has_group:
        msg["additional_properties"] = {"_group": {"id": f"g_{text}"}}
    return msg


def _run_replay_merge(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the replay_history merge logic from ChatPanel (batch_id aware).

    This faithfully mirrors the merge loop in ChatPanel.replay_history,
    including batch_id grouping, intermediate text handling, and
    injection-aware merging.
    """
    merged: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "tool":
            continue
        if role == "assistant":
            contents = msg.get("contents", [])
            extra = msg.get("additional_properties", {})

            if extra.get(HistoryMarkerKind.KEY):
                merged.append(msg)
                continue

            def _is_tool(c: Any) -> bool:
                return isinstance(c, dict) and c.get("type") in ("function_call", "legacy")

            text_part = []
            tool_part = []
            for c in contents:
                if _is_tool(c):
                    tool_part.append(c)
                else:
                    if tool_part:
                        break
                    text_part.append(c)

            has_text = any(
                (isinstance(c, dict) and c.get("type") == "text" and c.get("text")) or isinstance(c, str)
                for c in text_part
            )
            embedded_itext = extra.get("_intermediate_text")

            if has_text:
                text_entry: dict[str, Any] = {"role": "assistant", "contents": list(text_part)}
                if tool_part:
                    text_entry["_is_intermediate"] = True
                merged.append(text_entry)

            if tool_part:
                batch_id = extra.get("_batch_id")

                merge_target = None
                if batch_id is not None:
                    for j in range(len(merged) - 1, -1, -1):
                        prev = merged[j]
                        if prev.get("_merged_role") == "assistant_tools" and prev.get("_batch_id") == batch_id:
                            merge_target = j
                            break
                        # Stop at the first non-tool-group entry to avoid
                        # merging across user/text boundaries.
                        if prev.get("_merged_role") != "assistant_tools":
                            break
                if merge_target is not None:
                    merged[merge_target]["contents"].extend(tool_part)
                    if embedded_itext:
                        merged[merge_target].setdefault("_intermediate_texts", []).append(embedded_itext)
                else:
                    entry: dict[str, Any] = {
                        "_merged_role": "assistant_tools",
                        "role": "assistant",
                        "contents": list(tool_part),
                    }
                    if batch_id is not None:
                        entry["_batch_id"] = batch_id
                    if has_text:
                        entry["_has_preceding_text"] = True
                    if embedded_itext and not has_text:
                        entry["_intermediate_texts"] = [embedded_itext]
                    merged.append(entry)
            if not has_text and not tool_part:
                merged.append(msg)
            continue
        merged.append(msg)
    return merged


# ---------------------------------------------------------------------------
# 1. Double-break fix: batch_id merge walks back through tool groups
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. _excluded messages filtered from load_session_raw
# ---------------------------------------------------------------------------


class TestExcludedMessageFiltering:
    """Tests that _excluded messages are filtered in load_session_raw."""

    @pytest.mark.asyncio
    async def test_load_raw_skips_excluded_messages(self, tmp_path) -> None:
        """Messages with _excluded=True should not appear in load_session_raw."""
        store = JsonFileStateStore(tmp_path)
        m1 = Message("user", ["hello"])
        m2 = Message("assistant", [])
        fc = Content(type="function_call", call_id="c1", name="glob")
        fc.arguments = '{"pattern": "*.py"}'
        m2.contents.append(fc)
        m2.additional_properties["_excluded"] = True
        m3 = Message("tool", [])
        fr = Content(type="function_result", call_id="c1", result="found files")
        m3.contents.append(fr)
        m3.additional_properties["_excluded"] = True
        m4 = Message("assistant", ["response text"])

        state: dict[str, Any] = {"messages": [m1, m2, m3, m4], "compressed_msgs": []}
        await store.save_session("test_excl", state, agent_profile="code")

        result = await store.load_session_raw("test_excl")
        assert result is not None
        # Only the user message and the final assistant text should remain
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_load_raw_preserves_non_excluded(self, tmp_path) -> None:
        """Messages with _excluded=False or no _excluded key are preserved."""
        store = JsonFileStateStore(tmp_path)
        m1 = Message("user", ["hello"])
        m2 = Message("assistant", ["response"])
        m2.additional_properties["_excluded"] = False
        m3 = Message("assistant", ["another"])

        state: dict[str, Any] = {"messages": [m1, m2, m3], "compressed_msgs": []}
        await store.save_session("test_keep", state, agent_profile="code")

        result = await store.load_session_raw("test_keep")
        assert result is not None
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_excluded_tool_calls_dont_appear_in_replay(self, tmp_path) -> None:
        """Excluded tool call + result pairs should not create phantom tool groups."""
        store = JsonFileStateStore(tmp_path)
        messages_objs = [
            Message("user", ["go"]),
            # Non-excluded tool call
            _make_fc_message("c1", "tool_a"),
            _make_fr_message("c1", "ok"),
            # Excluded tool call (compacted)
            _make_fc_message("c2", "compacted_tool", excluded=True),
            _make_fr_message("c2", "old result", excluded=True),
            # Non-excluded final text
            Message("assistant", ["done"]),
        ]
        state: dict[str, Any] = {"messages": messages_objs, "compressed_msgs": []}
        await store.save_session("test_phantom", state, agent_profile="code")

        raw = await store.load_session_raw("test_phantom")
        assert raw is not None

        # 4 messages: user, assistant(tool_a), tool(c1 result), assistant(done)
        # The excluded compacted_tool assistant + tool pair is gone
        assert len(raw) == 4
        tool_calls = [
            c for m in raw for c in m.get("contents", []) if isinstance(c, dict) and c.get("type") == "function_call"
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "tool_a"

        # Running merge on the filtered messages should produce 1 tool group
        merged = _run_replay_merge(raw)
        tool_groups = [m for m in merged if m.get("_merged_role") == "assistant_tools"]
        assert len(tool_groups) == 1


# ---------------------------------------------------------------------------
# 3. End-to-end: replay through ChatPanel widget
# ---------------------------------------------------------------------------


class TestReplayWidget:
    """Tests that exercise the real ChatPanel.replay_history method."""

    @pytest.mark.asyncio
    async def test_replay_batch_merge_across_other_tool_group(self) -> None:
        """Split messages with the same batch_id still merge (first-pass rule),
        and consecutive tool groups then collapse into one (second-pass rule)
        since no intermediate text separates them.
        """
        messages = [
            _user_msg("go"),
            _asst_msg([_fc("c1", "tool_a")], batch_id=1),
            _tool_msg([_fr("c1")]),
            _asst_msg([_fc("c2", "tool_b")], batch_id=2),
            _tool_msg([_fr("c2")]),
            # Resubmitted — first-pass merges with batch_id=1 entry above
            _asst_msg([_fc("c3", "tool_c")], batch_id=1),
            _tool_msg([_fr("c3")]),
            _asst_msg([_text("done")]),
        ]
        result = await _run_replay(messages)
        # Second-pass merges consecutive tool groups into one: [tool_a, tool_c, tool_b]
        assert result["tool_group_count"] == 1
        assert result["tool_group_sizes"] == [3]


# ---------------------------------------------------------------------------
# Helpers for Message object construction
# ---------------------------------------------------------------------------


def _make_fc_message(call_id: str, name: str, *, excluded: bool = False) -> Message:
    """Create an assistant Message with a function_call Content."""
    msg = Message("assistant", [])
    fc = Content(type="function_call", call_id=call_id, name=name)
    fc.arguments = "{}"
    msg.contents.append(fc)
    if excluded:
        msg.additional_properties["_excluded"] = True
    return msg


def _make_fr_message(call_id: str, result: str, *, excluded: bool = False) -> Message:
    """Create a tool Message with a function_result Content."""
    msg = Message("tool", [])
    fr = Content(type="function_result", call_id=call_id, result=result)
    msg.contents.append(fr)
    if excluded:
        msg.additional_properties["_excluded"] = True
    return msg


# ---------------------------------------------------------------------------
# Replay widget helper (mirrors test_pipeline_tool_calls._run_replay)
# ---------------------------------------------------------------------------


async def _run_replay(raw_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Run replay_history in a headless Textual app and inspect widget tree."""
    from textual.app import App, ComposeResult

    from chrys.app.tui.widgets.chat.panel import ChatPanel

    class ReplayApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with ReplayApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.replay_history(raw_messages)
        await pilot.pause()

        from chrys.app.tui.widgets.chat.messages import AgentMessage as AgentMessageWidget
        from chrys.app.tui.widgets.chat.messages import UserMessage as UserMessageWidget
        from chrys.app.tui.widgets.chat.tool_call import ToolGroup

        user_msgs = cp.query(UserMessageWidget)
        tool_groups = cp.query(ToolGroup)
        agent_msgs = cp.query(AgentMessageWidget)

        tool_group_sizes = []
        for tg in tool_groups:
            tool_group_sizes.append(len(tg._tool_records))

        return {
            "user_count": len(user_msgs),
            "tool_group_count": len(tool_groups),
            "tool_group_sizes": tool_group_sizes,
            "agent_message_count": len(agent_msgs),
        }
