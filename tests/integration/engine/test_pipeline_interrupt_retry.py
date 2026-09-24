# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pipeline integration tests for interrupt, error and resume (scenario family S6).

Completed tool-loop iterations must survive an interrupt or a non-retryable
LLM error, a response that lands after the user pressed Stop must not reach
session state, and a retry or a new message must continue from the preserved
work.
"""

from __future__ import annotations

from chrys.foundation.events.types import InvocationToolCallResult, InvocationToolCallStart, UserMessage
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.service.llm.mock import MockResponse
from tests.support.pipeline_helpers import (
    _wait_for_loop_snapshot,
    error_on_nth,
    extract_final_messages,
    extract_session_batch_ids,
    extract_session_real_results,
    extract_session_tool_names,
    interrupt_on_nth,
    slow_echo_tool,
    wait_for_event,
    wait_for_idle,
)

# ---------------------------------------------------------------------------
# S6: Interrupt + resume
# ---------------------------------------------------------------------------


class TestInterruptAndResume:
    """Interrupt after first tool completes, then resume."""

    async def test_interrupt_preserves_completed_work(self, make_pipeline_ctx):
        """Interrupt mid-run: completed tools preserved, engine can resume."""
        slow_echo = slow_echo_tool(0.2, name="echo")

        # 3 sequential tool calls — we'll interrupt after the 1st
        responses = [
            MockResponse(tool_calls=[("echo", "i1", {"message": "first"})]),
            MockResponse(tool_calls=[("echo", "i2", {"message": "second"})]),
            MockResponse(tool_calls=[("echo", "i3", {"message": "third"})]),
            MockResponse(text="All three done"),
            # Resume responses: continue from where we left off
            MockResponse(tool_calls=[("echo", "i4", {"message": "resumed"})]),
            MockResponse(text="Resumed and done"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        await ctx.bus.publish(UserMessage(text="Run three tools"))

        # Wait for the second tool to START — this proves FunctionInvocationLayer
        # has committed the first tool's result to message history.
        # (InvocationToolCallResult fires from middleware BEFORE FunctionInvocationLayer
        # appends the result to messages, so waiting on ToolCallResult
        # alone is racy on macOS CI.)
        await wait_for_event(ctx.events, InvocationToolCallStart, timeout=20.0, min_count=2)

        # Interrupt
        await ctx.send_interrupt()
        await wait_for_idle(ctx)

        # Verify completed work preserved in session
        raw_after_interrupt = await ctx.get_session_messages()
        tool_names_after = extract_session_tool_names(raw_after_interrupt)
        assert len(tool_names_after) >= 1
        assert "echo" in tool_names_after

        # Verify batch_ids exist on completed work
        batch_ids_after = extract_session_batch_ids(raw_after_interrupt)
        completed_bids = [b for b in batch_ids_after if b is not None]
        assert len(completed_bids) >= 1
        pre_resume_max = max(completed_bids)

        # Resume
        ctx.events.clear()
        await ctx.send_retry()

        # Verify resumed work completes
        finals = extract_final_messages(ctx.events)
        assert len(finals) >= 1

        # Verify batch_ids don't collide after resume
        raw_after_resume = await ctx.get_session_messages()
        all_batch_ids = extract_session_batch_ids(raw_after_resume)
        non_none_bids = [b for b in all_batch_ids if b is not None]
        post_resume_bids = non_none_bids[len(completed_bids) :]
        if post_resume_bids:
            assert all(b > pre_resume_max for b in post_resume_bids), (
                f"Resume batch_ids {post_resume_bids} should be > pre-interrupt max {pre_resume_max}"
            )


# ---------------------------------------------------------------------------
# S6b: Interrupt preserves completed iterations in multi-iteration tool loop
# ---------------------------------------------------------------------------


class TestInterruptMultiIteration:
    """When interrupt fires mid-tool-loop, completed iterations must survive."""

    async def test_completed_iterations_preserved_on_interrupt(self, make_pipeline_ctx):
        """Multi-iteration tool loop: interrupt after 2 iterations, 1st preserved."""
        # Async tool that sleeps long enough for the interrupt to land
        # between iterations. InvocationToolCallResult fires when the
        # middleware sees the result, but LoopRecorder may not have
        # snapshotted it into recoverable state until the next LLM call.
        # _wait_for_loop_snapshot waits for that checkpoint before interrupting;
        # the tool delay leaves room for the interrupt to land.
        slow_echo = slow_echo_tool(0.1)

        # 3 sequential iterations + final text.
        # Interrupt should fire after iteration 0 completes.
        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="All done"),
            # Resume responses
            MockResponse(tool_calls=[("slow_echo", "c4", {"message": "resumed"})]),
            MockResponse(text="Resume done"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        # Start the run
        await ctx.bus.publish(UserMessage(text="Run three tools"))

        # Wait for first tool result, then for FunctionInvocationLayer to enter the
        # next tool loop iteration so LoopRecorder snapshots the
        # completed iteration's messages into recoverable state.  Polling
        # the recorder's snapshot (instead of a fixed sleep) avoids racing
        # the background checkpoint on slow CI.
        await wait_for_event(ctx.events, InvocationToolCallResult, timeout=20.0, min_count=1)
        await _wait_for_loop_snapshot(ctx, min_iterations=1)

        # Interrupt — will fire before the next tool executes
        await ctx.send_interrupt()
        await wait_for_idle(ctx)

        # Session must contain the completed tool call from iteration 0
        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)
        assert "slow_echo" in tool_names, f"Completed tool call lost on interrupt. Session tools: {tool_names}"

        # Verify at least 1 function_result has actual content (not None)
        assert extract_session_real_results(raw), "No completed tool results found in session"

        # Resume and verify it works
        ctx.events.clear()
        await ctx.send_retry()
        finals = extract_final_messages(ctx.events)
        assert len(finals) >= 1

    async def test_multiple_completed_iterations_preserved(self, make_pipeline_ctx):
        """Interrupt after 2+ iterations: all completed iterations preserved."""
        slow_echo = slow_echo_tool(0.1)

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(tool_calls=[("slow_echo", "c4", {"message": "fourth"})]),
            MockResponse(text="All done"),
            # Resume
            MockResponse(text="Resume done"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        await ctx.bus.publish(UserMessage(text="Run four tools"))

        # Wait for 2 tool results, then for the LoopRecorder to snapshot
        # both completed iterations into recoverable state.  Polling the
        # recorder's snapshot replaces a fixed settle that raced on CI.
        await wait_for_event(ctx.events, InvocationToolCallResult, timeout=20.0, min_count=2)
        await _wait_for_loop_snapshot(ctx, min_iterations=2)

        await ctx.send_interrupt()
        await wait_for_idle(ctx)

        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)

        # At least the first 2 completed iterations should be preserved
        assert len(tool_names) >= 2, f"Expected >= 2 completed tools, got {len(tool_names)}: {tool_names}"

        # Verify results are real (not None from interrupt)
        real_results = extract_session_real_results(raw)
        assert len(real_results) >= 2, f"Expected >= 2 real results, got {len(real_results)}: {real_results}"


# ---------------------------------------------------------------------------
# S6c: Error during tool loop + retry preserves completed work
# ---------------------------------------------------------------------------


class TestErrorAndRetry:
    """Error during multi-iteration tool loop: completed work survives, retry sees it."""

    async def test_error_preserves_completed_iterations(self, make_pipeline_ctx):
        """Network error during LLM call: completed iterations preserved in session."""
        slow_echo = slow_echo_tool(0.0)

        # 2 iterations succeed, then error on 3rd LLM call.
        # Retry responses follow.
        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            # 3rd LLM call will throw (patched below)
            # After retry:
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="All done after retry"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        # Patch mock client to throw on the 3rd LLM call
        error_on_nth(ctx, 3)

        # Run — should fail on 3rd LLM call after 2 iterations complete
        await ctx.send_message("Run three tools")

        # Engine should have set run_failed
        assert ctx.engine.current.loaded.bindings.state.run_failed, "Expected run_failed after non-retryable error"

        # Session must have both completed iterations
        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)
        assert len(tool_names) >= 2, f"Expected >= 2 tools preserved after error, got {tool_names}"

        # Verify results are real (not empty)
        real_results = extract_session_real_results(raw)
        assert len(real_results) >= 2, f"Expected >= 2 real results, got {real_results}"

    async def test_retry_after_error_sends_preserved_work_to_llm(self, make_pipeline_ctx):
        """On retry after error, LLM receives previously completed tool results."""
        slow_echo = slow_echo_tool(0.0)

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            # 3rd call errors
            # Retry:
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="Retry complete"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        restore_client = error_on_nth(ctx, 3)

        await ctx.send_message("Run tools")
        assert ctx.engine.current.loaded.bindings.state.run_failed

        # Restore normal client for retry
        restore_client()

        # Retry
        ctx.events.clear()
        await ctx.send_retry()

        # Verify retry completes
        finals = extract_final_messages(ctx.events)
        assert len(finals) >= 1, "Retry should produce a final message"

        # Verify the LLM received preserved work on retry.
        # The FIRST LLM call of the retry should contain function_result
        # contents from the completed iterations.
        retry_first_call_msgs = ctx.mock_client.call_history[-2][0]  # -2 = first retry LLM call
        preserved_results = []
        for msg in retry_first_call_msgs:
            for c in getattr(msg, "contents", []):
                if getattr(c, "type", "") == "function_result":
                    result = getattr(c, "result", "") or ""
                    if "echo:" in result:
                        preserved_results.append(result)
        assert len(preserved_results) >= 2, f"LLM should see >= 2 preserved results on retry, got {preserved_results}"

    async def test_retry_after_error_completes_full_session(self, make_pipeline_ctx):
        """Full round-trip: error → retry → final session has all work."""
        slow_echo = slow_echo_tool(0.0)

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            # 3rd call errors
            # Retry:
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="All done"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        restore_client = error_on_nth(ctx, 3)
        await ctx.send_message("Run tools")
        assert ctx.engine.current.loaded.bindings.state.run_failed

        restore_client()
        ctx.events.clear()
        await ctx.send_retry()

        # Final session should have original user message + all tool calls + final text
        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)
        # Should have at least the 2 from before error + 1 from retry = 3
        assert len(tool_names) >= 3, f"Expected >= 3 tools in final session, got {tool_names}"

        # Verify final text is present
        has_final = any(
            msg.get("role") == "assistant"
            and any(
                isinstance(c, dict) and c.get("type") == "text" and "All done" in (c.get("text") or "")
                for c in msg.get("contents", [])
            )
            for msg in raw
        )
        assert has_final, "Final text 'All done' not found in session"


# ---------------------------------------------------------------------------
# S6d: Interrupt during final LLM response drops leaked text
# ---------------------------------------------------------------------------


class TestInterruptDuringFinalResponse:
    """Interrupt during the LLM's final text response (no tool calls).

    The InterruptMiddleware only fires at tool call boundaries.  When the LLM
    is generating its final text-only response, the interrupt flag is set but
    the response completes and after_run stores it.  The engine must detect
    and remove this leaked response from session state.
    """

    async def test_interrupt_during_final_response_drops_text(self, make_pipeline_ctx):
        """Interrupt during final LLM call: text response not in session."""
        slow_echo = slow_echo_tool(0.0)

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            # 2nd LLM call returns text-only (final response).
            # Interrupt will be set during this call (see monkey-patch below).
            MockResponse(text="This is the final answer that should be dropped"),
            # Resume responses
            MockResponse(text="Resumed answer"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        # Monkey-patch mock client to set interrupt flag during the 2nd
        # LLM call, simulating the user pressing Stop while the LLM is
        # generating its final text response.
        interrupt_on_nth(ctx, 2)

        await ctx.send_message("Run one tool then respond")

        # Engine should have detected the interrupt
        assert ctx.engine.current.loaded.bindings.state.was_interrupted, (
            "Expected was_interrupted after interrupt during final response"
        )

        # Session must NOT contain the leaked final text
        raw = await ctx.get_session_messages()
        for msg in raw:
            for c in msg.get("contents", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    text = c.get("text", "")
                    assert "final answer that should be dropped" not in text, (
                        f"Leaked final response found in session: {text}"
                    )

        # Session should still have the completed tool call
        tool_names = extract_session_tool_names(raw)
        assert "slow_echo" in tool_names, f"Completed tool call should be preserved: {tool_names}"

    async def test_interrupt_before_any_tools_drops_text(self, make_pipeline_ctx):
        """Interrupt during first LLM call (text-only, no tools): session has only user msg."""
        responses = [
            # LLM returns text-only on first call (no tools needed).
            # Interrupt set during this call.
            MockResponse(text="Text response that should be dropped"),
            # Resume
            MockResponse(text="Resumed"),
        ]
        ctx = await make_pipeline_ctx(responses)

        interrupt_on_nth(ctx, 1)

        await ctx.send_message("Just explain something")

        assert ctx.engine.current.loaded.bindings.state.was_interrupted

        raw = await ctx.get_session_messages()
        # Should have user message + interrupted marker + turn marker, but NO assistant text
        for msg in raw:
            role = msg.get("role", "")
            if role == "assistant":
                extra = msg.get("additional_properties", {})
                chrys_kind = extra.get(HistoryMarkerKind.KEY, "")
                # Only chrys markers (interrupted, turn_marker) allowed
                assert chrys_kind in (HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.TURN), (
                    f"Unexpected assistant message in session: contents={msg.get('contents')}"
                )

    async def test_normal_completion_preserves_final_text(self, make_pipeline_ctx):
        """Regression: normal completion (no interrupt) keeps the final text."""
        slow_echo = slow_echo_tool(0.0)

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(text="Final answer preserved"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        await ctx.send_message("Run tool and respond")

        # Should NOT be interrupted
        assert not ctx.engine.current.loaded.bindings.state.was_interrupted

        # Final text should be in session
        raw = await ctx.get_session_messages()
        has_final_text = any(
            msg.get("role") == "assistant"
            and any(
                isinstance(c, dict) and c.get("type") == "text" and "Final answer preserved" in (c.get("text") or "")
                for c in msg.get("contents", [])
            )
            for msg in raw
        )
        assert has_final_text, "Final text should be preserved on normal completion"

    async def test_resume_after_interrupt_during_final_response(self, make_pipeline_ctx):
        """Interrupt during final LLM call, then resume: session ends with new response."""
        slow_echo = slow_echo_tool(0.0)

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            # Final response — interrupt fires during this LLM call
            MockResponse(text="Leaked answer"),
            # Resume responses
            MockResponse(text="Correct resumed answer"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[slow_echo])

        restore_client = interrupt_on_nth(ctx, 2)
        await ctx.send_message("Run tool then respond")

        assert ctx.engine.current.loaded.bindings.state.was_interrupted

        # Resume
        restore_client()
        ctx.events.clear()
        await ctx.send_retry()

        # Verify the resumed response is in session
        raw = await ctx.get_session_messages()
        has_resumed = any(
            msg.get("role") == "assistant"
            and any(
                isinstance(c, dict) and c.get("type") == "text" and "Correct resumed answer" in (c.get("text") or "")
                for c in msg.get("contents", [])
            )
            for msg in raw
        )
        assert has_resumed, "Resumed answer should be in session"

        # Leaked answer should NOT be present
        has_leaked = any(
            msg.get("role") == "assistant"
            and any(
                isinstance(c, dict) and c.get("type") == "text" and "Leaked answer" in (c.get("text") or "")
                for c in msg.get("contents", [])
            )
            for msg in raw
        )
        assert not has_leaked, "Leaked answer from interrupted run should not be in session"


# ---------------------------------------------------------------------------
# S6e: Resume appends to existing state (never drops content)
# ---------------------------------------------------------------------------


class TestResumeAppendsToState:
    """On resume, completed work is preserved and the LLM continues.

    Session history is append-only: only chrys markers (interrupted,
    turn_marker) are removed.  Completed tool calls, user messages, and
    assistant messages are never deleted.
    """

    async def test_resume_preserves_completed_tools(self, make_pipeline_ctx):
        """Interrupt during final LLM call: completed tool preserved on resume."""
        fast_tool = slow_echo_tool(0.0, name="fast_tool", description="Fast tool", result_prefix="result")

        responses = [
            MockResponse(tool_calls=[("fast_tool", "c1", {"message": "hello"})]),
            # Final response — interrupt fires here
            MockResponse(text="Interrupted final"),
            # Resume: LLM sees completed tool result, generates new response
            MockResponse(text="Resumed with context"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[fast_tool])

        restore_client = interrupt_on_nth(ctx, 2)
        await ctx.send_message("Do something")

        assert ctx.engine.current.loaded.bindings.state.was_interrupted

        # Completed tool should be in session
        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)
        assert "fast_tool" in tool_names, "Completed tool should be preserved"

        # Resume — LLM continues from completed work
        restore_client()
        ctx.events.clear()
        await ctx.send_retry()

        finals = extract_final_messages(ctx.events)
        assert any("Resumed" in t for t in finals)

        # Session still has the completed tool AND the new response
        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)
        assert "fast_tool" in tool_names, "Completed tool preserved after resume"

    async def test_resume_text_only_resends_user_message(self, make_pipeline_ctx):
        """Interrupt during text-only response: resume re-sends user message."""
        responses = [
            MockResponse(text="Will be interrupted"),
            MockResponse(text="Fresh response"),
        ]
        ctx = await make_pipeline_ctx(responses)

        restore_client = interrupt_on_nth(ctx, 1)
        await ctx.send_message("Say something")

        assert ctx.engine.current.loaded.bindings.state.was_interrupted

        restore_client()
        ctx.events.clear()
        await ctx.send_retry()

        finals = extract_final_messages(ctx.events)
        assert any("Fresh" in t for t in finals), "Should get fresh response on resume"

    async def test_new_message_after_interrupt_keeps_context(self, make_pipeline_ctx):
        """New message after interrupt preserves all previous context."""
        echo_tool = slow_echo_tool(0.0, name="echo", description="Echo")

        responses = [
            MockResponse(tool_calls=[("echo", "c1", {"message": "first"})]),
            # Interrupt fires during this response
            MockResponse(text="Interrupted"),
            # New message gets this response
            MockResponse(text="Got your follow-up"),
        ]
        ctx = await make_pipeline_ctx(responses, tools=[echo_tool])

        restore_client = interrupt_on_nth(ctx, 2)
        await ctx.send_message("Do something")

        assert ctx.engine.current.loaded.bindings.state.was_interrupted

        # Send a NEW message (not resume)
        restore_client()
        ctx.events.clear()
        await ctx.send_message("Continue please")

        # Session should have: original user msg + completed tool + new user msg + response
        raw = await ctx.get_session_messages()
        user_msgs = [
            m
            for m in raw
            if m.get("role") == "user" and not m.get("additional_properties", {}).get(HistoryMarkerKind.KEY)
        ]
        assert len(user_msgs) >= 2, f"Both user messages should be preserved, got {len(user_msgs)}"

        # Completed tool from first run should still be there
        tool_names = extract_session_tool_names(raw)
        assert "echo" in tool_names, "Completed tool from interrupted run preserved"
