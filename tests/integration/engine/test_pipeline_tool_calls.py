# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pipeline integration tests for tool-call shape: sequencing, batching, replay.

Scenario families S1 (multi-turn sequential calls), S2 (parallel calls in one
LLM response), S3 (intermediate text alongside tool calls) and S7 (three-way
consistency between events, session state and TUI replay).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from chrys.service.llm.mock import MockResponse
from tests.support.pipeline_helpers import (
    extract_final_messages,
    extract_intermediate_messages,
    extract_session_batch_ids,
    extract_session_intermediate_texts,
    extract_session_tool_names,
    extract_tool_results,
    extract_tool_starts,
)

# ---------------------------------------------------------------------------
# S1: Multi-turn sequential tool calls
# ---------------------------------------------------------------------------


class TestMultiTurnSequential:
    """Two turns, each with sequential tool calls from different LLM responses."""

    @pytest.fixture
    async def ctx(self, make_pipeline_ctx):
        responses = [
            # Turn 1: echo → concat → final text
            MockResponse(tool_calls=[("echo", "c1", {"message": "hello"})]),
            MockResponse(tool_calls=[("concat", "c2", {"a": "x", "b": "y"})]),
            MockResponse(text="Turn 1 complete"),
            # Turn 2: echo → final text
            MockResponse(tool_calls=[("echo", "c3", {"message": "world"})]),
            MockResponse(text="Turn 2 complete"),
        ]
        return await make_pipeline_ctx(responses)

    async def test_events(self, ctx):
        """Events: correct ToolCallStart/Result pairs and final messages."""
        await ctx.send_message("Do turn 1")
        await ctx.send_message("Do turn 2")

        starts = extract_tool_starts(ctx.events)
        results = extract_tool_results(ctx.events)
        finals = extract_final_messages(ctx.events)

        assert len(starts) == 3
        assert len(results) == 3
        assert starts[0]["tool_name"] == "echo"
        assert starts[1]["tool_name"] == "concat"
        assert starts[2]["tool_name"] == "echo"

        assert len(finals) == 2
        assert finals[0] == "Turn 1 complete"
        assert finals[1] == "Turn 2 complete"

    async def test_session_batch_ids(self, ctx):
        """Session: sequential tool calls get distinct batch_ids per LLM response."""
        await ctx.send_message("Do turn 1")
        await ctx.send_message("Do turn 2")

        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)
        batch_ids = extract_session_batch_ids(raw)

        assert tool_names == ["echo", "concat", "echo"]
        assert len(batch_ids) == 3
        assert all(b is not None for b in batch_ids)
        # echo and concat are from different LLM responses → distinct batch_ids
        assert batch_ids[0] != batch_ids[1]


class TestSlowTurnBoundary:
    """A turn outlasting wait_for_idle's historical 5s ceiling stays one turn.

    wait_for_idle used to silently suppress its 5s timeout: on an overloaded
    runner send_message returned mid-turn, so the test's NEXT message was
    consumed as an injection into turn 1 instead of starting turn 2, and
    turn 2's tool call vanished from the persisted session (the recurring
    Windows CI failure).  The wait is now generous and fails loudly instead;
    this test pins that by keeping turn 1 active beyond the old ceiling.
    """

    async def test_turn_slower_than_legacy_wait_ceiling_keeps_turn_boundary(self, make_pipeline_ctx):
        from chrys.kernel import FunctionTool

        async def _stalled_echo(message: str) -> str:
            # Beyond the old 5s wait_for_idle ceiling, well under the new 30s.
            await asyncio.sleep(6.0)
            return f"echo: {message}"

        async def _echo(message: str) -> str:
            return f"echo: {message}"

        stalled_echo = FunctionTool(func=_stalled_echo, name="stalled_echo", description="Echo after a stall")
        echo = FunctionTool(func=_echo, name="echo", description="Echo")

        responses = [
            # Turn 1: one tool call that stalls past the legacy ceiling
            MockResponse(tool_calls=[("stalled_echo", "sl1", {"message": "slow"})]),
            MockResponse(text="Turn 1 complete"),
            # Turn 2: a fast tool call that must land in a NEW turn
            MockResponse(tool_calls=[("echo", "sl2", {"message": "fast"})]),
            MockResponse(text="Turn 2 complete"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[stalled_echo, echo])
        await ctx.send_message("Do turn 1")
        await ctx.send_message("Do turn 2")

        # Two distinct final messages prove message 2 started its own turn
        # rather than being injected into the still-running turn 1.
        finals = extract_final_messages(ctx.events)
        assert finals == ["Turn 1 complete", "Turn 2 complete"]

        raw = await ctx.get_session_messages()
        assert extract_session_tool_names(raw) == ["stalled_echo", "echo"]


# ---------------------------------------------------------------------------
# S2: Parallel tool calls (single LLM response)
# ---------------------------------------------------------------------------


class TestParallelToolCalls:
    """One LLM response returns multiple tool_calls simultaneously."""

    @pytest.fixture
    async def ctx(self, make_pipeline_ctx):
        responses = [
            MockResponse(
                tool_calls=[
                    ("echo", "p1", {"message": "a"}),
                    ("concat", "p2", {"a": "b", "b": "c"}),
                    ("uppercase", "p3", {"text": "hello"}),
                ]
            ),
            MockResponse(text="All done"),
        ]
        return await make_pipeline_ctx(responses)

    async def test_events(self, ctx):
        """Events: 3 starts + 3 results from single LLM response."""
        await ctx.send_message("Run three tools")

        starts = extract_tool_starts(ctx.events)
        results = extract_tool_results(ctx.events)

        assert len(starts) == 3
        assert len(results) == 3
        tool_names = {s["tool_name"] for s in starts}
        assert tool_names == {"echo", "concat", "uppercase"}

    async def test_session_batch_ids(self, ctx):
        """Session: all parallel tools share the same batch_id."""
        await ctx.send_message("Run three tools")

        raw = await ctx.get_session_messages()
        batch_ids = extract_session_batch_ids(raw)

        # All 3 tool calls are in a single assistant message with one batch_id
        assert len(batch_ids) == 1
        assert batch_ids[0] is not None


# ---------------------------------------------------------------------------
# S3: Intermediate text alongside tool calls
# ---------------------------------------------------------------------------


class TestIntermediateText:
    """LLM returns text alongside tool calls — mock client now fires callbacks."""

    async def test_intermediate_text_callback(self, make_pipeline_ctx):
        """Mock client fires intermediate text callback for text+tools response."""
        responses = [
            MockResponse(text="Let me search...", tool_calls=[("echo", "it1", {"message": "query"})]),
            MockResponse(text="Found the answer!"),
        ]
        ctx = await make_pipeline_ctx(responses)

        await ctx.send_message("Search for something")

        # Intermediate text should be captured via the callback
        intermediates = extract_intermediate_messages(ctx.events)
        assert len(intermediates) >= 1

        # Final message should arrive
        finals = extract_final_messages(ctx.events)
        assert len(finals) == 1
        assert finals[0] == "Found the answer!"

    async def test_no_duplicate_intermediate_text(self, make_pipeline_ctx):
        """When text is already in message contents, _intermediate_text is NOT set."""
        responses = [
            MockResponse(text="Thinking...", tool_calls=[("echo", "it2", {"message": "q"})]),
            MockResponse(text="Done"),
        ]
        ctx = await make_pipeline_ctx(responses)

        await ctx.send_message("Think and act")

        raw = await ctx.get_session_messages()
        # The text is in the message contents (text + fn_calls),
        # so _intermediate_text should NOT be set (would cause duplication)
        itexts = extract_session_intermediate_texts(raw)
        assert len(itexts) == 0

        # But the text should be in the message contents
        for msg in raw:
            if msg.get("role") == "assistant":
                contents = msg.get("contents", [])
                has_fc = any(isinstance(c, dict) and c.get("type") == "function_call" for c in contents)
                text_parts = [c.get("text", "") for c in contents if isinstance(c, dict) and c.get("type") == "text"]
                if has_fc and text_parts:
                    assert "Thinking" in text_parts[0]
                    break

    async def test_multi_turn_intermediate_text_isolation(self, make_pipeline_ctx):
        """Intermediate texts from turn 2 must NOT be assigned to turn 1 messages.

        Regression test: _persist_intermediate_texts used to iterate from the
        start of the message list, so turn 2's texts would be assigned to
        unmatched assistant messages from turn 1.
        """
        responses = [
            # Turn 1: tool-only response (no intermediate text)
            MockResponse(tool_calls=[("echo", "mt1", {"message": "a"})]),
            MockResponse(text="Turn 1 done"),
            # Turn 2: text + tools (intermediate text)
            MockResponse(text="Planning...", tool_calls=[("echo", "mt2", {"message": "b"})]),
            MockResponse(text="Turn 2 done"),
        ]
        ctx = await make_pipeline_ctx(responses)

        # Turn 1
        await ctx.send_message("First turn")
        raw1 = await ctx.get_session_messages()
        itexts1 = extract_session_intermediate_texts(raw1)
        # Turn 1 had no intermediate text
        assert len(itexts1) == 0, f"Turn 1 should have no intermediate texts, got {itexts1}"

        # Turn 2
        await ctx.send_message("Second turn")
        raw2 = await ctx.get_session_messages()
        # Crucially, turn 1's messages must NOT have _intermediate_text
        for msg in raw2:
            if msg.get("role") != "assistant":
                continue
            extra = msg.get("additional_properties", {}) or {}
            batch_id = extra.get("_batch_id")
            itext = extra.get("_intermediate_text")
            # Turn 1 messages (batch_id from turn 1) must not have itext
            # from turn 2.  If itext is set, it should match the batch's
            # own response, not a later turn's text.
            if itext and batch_id is not None:
                # Verify the text makes sense for this batch
                assert itext != "Planning...", (
                    f"Turn 2 intermediate text leaked to turn 1 message (batch_id={batch_id})"
                )

    async def test_parallel_tools_intermediate_text_not_duplicated(self, make_pipeline_ctx):
        """When parallel tool calls are split into separate messages, only the
        first message of each batch should get intermediate text.

        Regression test: _persist_intermediate_texts used to assign one text
        per assistant message (not per batch), so text from batch 2 would leak
        to extra messages from batch 1's parallel tool calls.
        """
        responses = [
            # Batch 1: text + 3 parallel tools
            MockResponse(
                text="Step 1: searching...",
                tool_calls=[
                    ("echo", "p1", {"message": "a"}),
                    ("concat", "p2", {"a": "b", "b": "c"}),
                    ("uppercase", "p3", {"text": "d"}),
                ],
            ),
            # Batch 2: text + 1 tool
            MockResponse(text="Step 2: reading...", tool_calls=[("echo", "p4", {"message": "e"})]),
            MockResponse(text="Done"),
        ]
        ctx = await make_pipeline_ctx(responses)

        await ctx.send_message("Do work")
        raw = await ctx.get_session_messages()

        # At most 1 intermediate text should be set (for batch 1's first msg).
        # Batch 2's text should NOT leak to batch 1's other messages.
        # (Batch 2 might have 0 or 1 depending on whether the provider
        # preserves text in contents.)
        for msg in raw:
            if msg.get("role") != "assistant":
                continue
            extra = msg.get("additional_properties", {}) or {}
            itext = extra.get("_intermediate_text")
            if itext:
                assert itext != "Step 2: reading...", (
                    f"Batch 2 text leaked to batch 1 message (batch_id={extra.get('_batch_id')})"
                )

    async def test_batch_boundary_without_text(self, make_pipeline_ctx):
        """Tool-only responses still fire empty callback → batch_id increments."""
        responses = [
            MockResponse(tool_calls=[("echo", "b1", {"message": "first"})]),
            MockResponse(tool_calls=[("echo", "b2", {"message": "second"})]),
            MockResponse(text="Done"),
        ]
        ctx = await make_pipeline_ctx(responses)

        await ctx.send_message("Run two tools")

        raw = await ctx.get_session_messages()
        batch_ids = extract_session_batch_ids(raw)

        assert len(batch_ids) == 2
        assert all(b is not None for b in batch_ids)
        # Two separate LLM responses → distinct batch_ids
        assert batch_ids[0] != batch_ids[1]


# ---------------------------------------------------------------------------
# S7: Three-way consistency (events ↔ session ↔ replay)
# ---------------------------------------------------------------------------


class TestThreeWayConsistency:
    """Verify events, session state, and replay all agree on tool call structure."""

    async def test_sequential_consistency(self, make_pipeline_ctx):
        """Sequential tool calls: events and session match, distinct batch_ids."""
        responses = [
            MockResponse(tool_calls=[("echo", "s1", {"message": "a"})]),
            MockResponse(tool_calls=[("concat", "s2", {"a": "b", "b": "c"})]),
            MockResponse(text="Sequential done"),
        ]
        ctx = await make_pipeline_ctx(responses)

        await ctx.send_message("Run sequential tools")

        # 1. Event tool names
        event_tools = [s["tool_name"] for s in extract_tool_starts(ctx.events)]

        # 2. Session tool names
        raw = await ctx.get_session_messages()
        session_tools = extract_session_tool_names(raw)

        # Events and session match
        assert event_tools == ["echo", "concat"]
        assert session_tools == ["echo", "concat"]

        # Distinct batch_ids for sequential tools
        batch_ids = extract_session_batch_ids(raw)
        assert all(b is not None for b in batch_ids)
        assert batch_ids[0] != batch_ids[1]

    async def test_parallel_consistency(self, make_pipeline_ctx):
        """Parallel tool calls: events and session match, shared batch_id."""
        responses = [
            MockResponse(
                tool_calls=[
                    ("echo", "p1", {"message": "x"}),
                    ("uppercase", "p2", {"text": "y"}),
                ]
            ),
            MockResponse(text="Parallel done"),
        ]
        ctx = await make_pipeline_ctx(responses)

        await ctx.send_message("Run parallel tools")

        event_tools = [s["tool_name"] for s in extract_tool_starts(ctx.events)]
        raw = await ctx.get_session_messages()
        session_tools = extract_session_tool_names(raw)

        assert set(event_tools) == {"echo", "uppercase"}
        assert set(session_tools) == {"echo", "uppercase"}

        # Parallel tools share one batch_id
        batch_ids = extract_session_batch_ids(raw)
        assert len(batch_ids) == 1
        assert batch_ids[0] is not None

    async def test_replay_groups_match_batch_ids(self, make_pipeline_ctx):
        """Replay produces one merged tool group when no intermediate text separates batches."""
        responses = [
            MockResponse(tool_calls=[("echo", "r1", {"message": "one"})]),
            MockResponse(
                tool_calls=[
                    ("concat", "r2", {"a": "a", "b": "b"}),
                    ("uppercase", "r3", {"text": "test"}),
                ]
            ),
            MockResponse(text="Done with all tools"),
        ]
        ctx = await make_pipeline_ctx(responses)

        await ctx.send_message("Run mixed tools")

        raw = await ctx.get_session_messages()

        # Verify session structure: all 3 tools present
        session_tools = extract_session_tool_names(raw)
        assert session_tools == ["echo", "concat", "uppercase"]

        # echo has batch_id=1, concat+uppercase have batch_id=2
        batch_ids = extract_session_batch_ids(raw)
        assert len(batch_ids) == 2
        assert batch_ids[0] != batch_ids[1]

        # Run replay through ChatPanel
        replay_result = await _run_replay(raw)

        # No intermediate text separates the two batches, so replay
        # merges them into one 3-tool group.
        assert replay_result["user_count"] == 1
        assert replay_result["tool_group_count"] == 1
        assert replay_result["tool_group_sizes"] == [3]
        assert replay_result["agent_message_count"] >= 1


# ---------------------------------------------------------------------------
# Replay helper (minimal Textual app)
# ---------------------------------------------------------------------------


async def _run_replay(raw_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Run replay_history in a headless Textual app and inspect widget tree.

    Returns:
        Dict with: user_count, tool_group_count, tool_group_sizes,
        agent_message_count.
    """
    from textual.app import App, ComposeResult

    from chrys.app.tui.theme import TuiVariableDefaultsMixin
    from chrys.app.tui.widgets.chat.panel import ChatPanel

    class ReplayApp(TuiVariableDefaultsMixin, App):
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
