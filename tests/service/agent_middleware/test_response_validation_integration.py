# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration tests for :class:`ResponseValidationMiddleware` in the full pipeline.

The unit tests in ``test_response_validation.py`` exercise the middleware
in isolation with a hand-rolled ``_FakeCallNext``.  These integration
tests drive the **real** pipeline — :class:`AgentEngine` →
:class:`TurnBindings` → ``Agent.run()`` → :class:`MockChatClient` — to
confirm nothing else in the stack is perturbed by the middleware's
consume-validate-replay dance.

What these tests guard against:

1. ``RetryAttempt`` events are actually published on the ``EventBus``
   when malformed responses are received.
2. The final ``AgentMessage(is_final=True)`` text is the GOOD response's
   text, never the bad one's (in both streaming and non-streaming modes).
3. Streaming chunks (``is_final=False`` events) come **only** from the
   successful final response, never from dropped bad attempts.
4. The tool loop still executes downstream tool calls correctly when a
   bad final response is replaced by a good tool-calling + final
   response on retry.
5. Exhausted retries complete the engine run naturally (``IDLE`` FSM
   state, session saved) without raising.  The exhausted response's
   text is preserved but its ``function_call`` contents are stripped
   so the Chrys tool loop exits instead of executing the bad
   call and re-entering another validation cycle.
6. Session persistence contains only the final response's text, so
   downstream turns see a clean history.
7. Sub-agent retries publish :class:`InvocationRetryAttempt` events keyed
   to the right ``invocation_id``.

The mock client's intermediate-text callbacks are wired exactly as the
engine wires real ones, so this also guards the ``batch_id`` / hook
timing regression the unit tests protect at the middleware layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import chrys.orchestration.engine.build.builder as builder_module
import chrys.orchestration.sub_agents.tools as sub_agent_module
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    Error,
    Event,
    InvocationMessage,
    InvocationRetryAttempt,
    InvocationStarted,
    InvocationToolCallResult,
    InvocationToolCallStart,
    SessionReady,
    ToolCompacted,
    UsageUpdate,
    UserMessage,
    Warning,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.workspace import Workspace
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state.machine import EngineState
from chrys.service.agent_middleware.response_validation import (
    MAX_RETRIES,
    ResponseValidationMiddleware,
)
from chrys.service.agent_middleware.validators import NO_VISIBLE_OUTPUT_REASON, OUTPUT_TRUNCATED_REASON
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    ModelConfig,
    SubAgentRef,
    SubAgentsConfig,
    ToolsConfig,
)
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.state.store import JsonFileStateStore
from tests.support.pipeline_helpers import (
    create_test_engine,
    extract_final_messages,
    extract_intermediate_messages,
    extract_tool_results,
    extract_tool_starts,
    wait_for_event,
)
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Varied-invalid responses — needed wherever a test must reach MAX_RETRIES
# exhaustion.  Two consecutive responses with the same validator reason
# would trigger the middleware's fail-fast short-circuit (only 2 LLM
# calls instead of MAX_RETRIES + 1), so the leading attempts cycle
# through three distinct invalid shapes that map to three distinct
# ``ValidationResult.reason`` strings:
#
#   * ``MockResponse(text="")``                      → "empty contents"
#   * ``MockResponse(text="\n\n\n")``                → "empty or whitespace-only text response"
#   * ``MockResponse(text="minimax:tool_call ...")`` → "leaked tool-call marker..."
# ---------------------------------------------------------------------------


_VARIED_INVALID_TEXT_FACTORIES: tuple[Any, ...] = (
    lambda: MockResponse(text=""),
    lambda: MockResponse(text="\n\n\n"),
    lambda: MockResponse(text="minimax:tool_call {} </minimax:tool_call>"),
)


def _varied_invalid_mocks(n: int, *, last: MockResponse | None = None) -> list[MockResponse]:
    """Build ``n`` invalid MockResponses cycling through distinct reasons.

    Each consecutive pair has a different ``ValidationResult.reason`` so
    the fail-fast short-circuit does not fire.  When ``last`` is given,
    it overrides the final response — useful for tests that need the
    give-up cleanup to operate on a specific shape (e.g. a function-call
    response that must be stripped on exhaustion).
    """
    out = [_VARIED_INVALID_TEXT_FACTORIES[i % len(_VARIED_INVALID_TEXT_FACTORIES)]() for i in range(n)]
    if last is not None and out:
        out[-1] = last
    return out


# ---------------------------------------------------------------------------
# Fast backoff fixture — keeps integration tests snappy.
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_validation_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out the validation middleware's backoff schedule.

    Without this, a single empty response adds 0.5 s of sleep — six empty
    responses would add ~15 s.  Tests monkeypatch the class-level
    default so every instance built by ``build_agent`` picks it up.
    """
    monkeypatch.setattr(
        ResponseValidationMiddleware,
        "_BACKOFF_SCHEDULE",
        (0.0, 0.0, 0.0, 0.0, 0.0),
    )


# ===========================================================================
# Main-agent integration — non-streaming
# ===========================================================================


class TestMainAgentNonStreaming:
    @pytest.mark.asyncio
    async def test_empty_response_retried_good_text_delivered(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Bad (empty) response → RetryAttempt event → good response → final message."""
        ctx = await create_test_engine(
            [
                MockResponse(text=""),  # empty contents — invalid
                MockResponse(text="recovered answer"),
            ],
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            # The mock client was called twice (1 bad + 1 good).
            assert ctx.mock_client.call_count == 2

            # A RetryAttempt event was published for the bad response.
            assert len(retries) == 1
            assert "Invalid response" in retries[0].message
            assert "empty" in retries[0].message.lower()
            assert retries[0].attempt == 1
            assert retries[0].max_attempts == MAX_RETRIES
            assert retries[0].session_id == ctx.session_id

            # The final AgentMessage is the GOOD response's text.
            finals = extract_final_messages(ctx.events)
            assert finals == ["recovered answer"]
        finally:
            await ctx.cleanup()

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_truncated_empty_response_fails_without_blank_agent_message_and_persists_error_marker(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        stream: bool,
    ) -> None:
        """finish_reason=length + no output is a terminal turn error.

        It must render through the normal Error path, not the Warning/notify
        path, and it must persist the same error marker that replay uses after
        session reload.  The executor must not publish an empty final
        AgentMessage, which would create a blank Code Agent block in the TUI.
        """
        ctx = await create_test_engine(
            [
                MockResponse(
                    text="",
                    finish_reason="length",
                    usage_details={"input_token_count": 123, "output_token_count": 0, "total_token_count": 123},
                )
            ],
            tmp_path,
            stream=stream,
        )
        errors: list[Error] = []
        warnings: list[Warning] = []
        retries: list[InvocationRetryAttempt] = []
        usage_updates: list[UsageUpdate] = []

        async def _on_error(ev: Error) -> None:
            errors.append(ev)

        async def _on_warning(ev: Warning) -> None:
            warnings.append(ev)

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        async def _on_usage(ev: UsageUpdate) -> None:
            usage_updates.append(ev)

        await ctx.bus.subscribe(Error, _on_error)
        await ctx.bus.subscribe(Warning, _on_warning)
        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)
        await ctx.bus.subscribe(UsageUpdate, _on_usage)

        try:
            await ctx.send_message("go")
            await wait_for(
                lambda: len(usage_updates) == 1,
                timeout=ENGINE_TURN_TIMEOUT,
                description="single usage update after invalid response",
            )

            assert ctx.mock_client.call_count == 1
            assert retries == []
            assert warnings == []
            assert len(usage_updates) == 1
            assert usage_updates[0].input_tokens == 123
            assert usage_updates[0].output_tokens == 0
            assert usage_updates[0].total_tokens == 123
            assert len(errors) == 1
            assert errors[0].code == "executor_error"
            assert errors[0].message == OUTPUT_TRUNCATED_REASON
            assert extract_final_messages(ctx.events) == []
            assert ctx.engine.state is EngineState.FAILED

            raw_messages = await ctx.get_session_messages()
            assert [m.get("role") for m in raw_messages] == ["user", "assistant", "assistant"]
            marker = raw_messages[1]
            assert marker.get("additional_properties", {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
            assert marker.get("additional_properties", {}).get("_interrupted_by") == "error"
            assert marker.get("contents", [{}])[0].get("text") == OUTPUT_TRUNCATED_REASON
            assert raw_messages[2].get("additional_properties", {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
        finally:
            await ctx.cleanup()

    @pytest.mark.parametrize("text", ["", "   "], ids=["empty", "whitespace"])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_a_reply_cut_off_before_any_output_makes_the_retry_compact_first(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        stream: bool,
        text: str,
    ) -> None:
        """The input left no room to answer: the turn still fails, and its Retry compacts before resending."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("echo", "call-1", {"message": "x" * 4000})]),
                MockResponse(text="first answer"),
                MockResponse(text=text, finish_reason="length"),
                MockResponse(text="recovered answer"),
            ],
            tmp_path,
            stream=stream,
            compaction=CompactionConfig(enabled=True),
        )
        errors: list[Error] = []
        compacted: list[ToolCompacted] = []

        async def _on_error(ev: Error) -> None:
            errors.append(ev)

        async def _on_compacted(ev: ToolCompacted) -> None:
            compacted.append(ev)

        await ctx.bus.subscribe(Error, _on_error)
        await ctx.bus.subscribe(ToolCompacted, _on_compacted)

        try:
            await ctx.send_message("first")
            await ctx.send_message("second")

            assert ctx.mock_client.call_count == 3
            assert [error.message for error in errors] == [OUTPUT_TRUNCATED_REASON]
            assert compacted == []

            await ctx.send_retry()

            assert ctx.mock_client.call_count == 4
            assert [event.phase for event in compacted] == ["phase1"]
            assert extract_final_messages(ctx.events) == ["first answer", "recovered answer"]
        finally:
            await ctx.cleanup()

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_an_empty_reply_after_hidden_reasoning_does_not_make_the_retry_compact(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        stream: bool,
    ) -> None:
        """Usage counts reasoning the API did not return: the output limit ran out, not the window."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("echo", "call-1", {"message": "x" * 4000})]),
                MockResponse(text="first answer"),
                MockResponse(
                    finish_reason="length",
                    usage_details={"output_token_count": 4096, "reasoning_output_token_count": 4096},
                ),
                MockResponse(text="recovered answer"),
            ],
            tmp_path,
            stream=stream,
            compaction=CompactionConfig(enabled=True),
        )
        errors: list[Error] = []
        compacted: list[ToolCompacted] = []

        async def _on_error(ev: Error) -> None:
            errors.append(ev)

        async def _on_compacted(ev: ToolCompacted) -> None:
            compacted.append(ev)

        await ctx.bus.subscribe(Error, _on_error)
        await ctx.bus.subscribe(ToolCompacted, _on_compacted)

        try:
            await ctx.send_message("first")
            await ctx.send_message("second")
            assert [error.message for error in errors] == [OUTPUT_TRUNCATED_REASON]

            await ctx.send_retry()

            assert ctx.mock_client.call_count == 4
            assert extract_final_messages(ctx.events) == ["first answer", "recovered answer"]
            assert compacted == []
        finally:
            await ctx.cleanup()

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_a_reply_cut_off_after_reasoning_does_not_make_the_next_turn_compact(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        stream: bool,
    ) -> None:
        """Reasoning was generated, so the window was not full: the validation retry is the whole response."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("echo", "call-1", {"message": "x" * 4000})]),
                MockResponse(text="first answer"),
                MockResponse(reasoning_text="thinking", finish_reason="length"),
                MockResponse(text="second answer"),
                MockResponse(text="third answer"),
            ],
            tmp_path,
            stream=stream,
            compaction=CompactionConfig(enabled=True),
        )
        compacted: list[ToolCompacted] = []

        async def _on_compacted(ev: ToolCompacted) -> None:
            compacted.append(ev)

        await ctx.bus.subscribe(ToolCompacted, _on_compacted)

        try:
            await ctx.send_message("first")
            await ctx.send_message("second")
            await ctx.send_message("third")

            assert ctx.mock_client.call_count == 5
            assert extract_final_messages(ctx.events) == ["first answer", "second answer", "third answer"]
            assert compacted == []
        finally:
            await ctx.cleanup()

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_reasoning_only_response_retried_good_text_delivered(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        stream: bool,
    ) -> None:
        """Reasoning-only final response → RetryAttempt → good response wins."""
        ctx = await create_test_engine(
            [
                MockResponse(reasoning_text="internal reasoning only"),
                MockResponse(text="recovered answer"),
            ],
            tmp_path,
            stream=stream,
        )
        retries: list[InvocationRetryAttempt] = []
        errors: list[Error] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        async def _on_error(ev: Error) -> None:
            errors.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)
        await ctx.bus.subscribe(Error, _on_error)

        try:
            await ctx.send_message("go")

            assert ctx.mock_client.call_count == 2
            assert errors == []
            assert len(retries) == 1
            assert "no visible answer" in retries[0].message
            assert retries[0].session_id == ctx.session_id
            assert extract_final_messages(ctx.events) == ["recovered answer"]
        finally:
            await ctx.cleanup()

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_reasoning_only_repeat_failure_errors_with_session_and_no_blank_message(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        stream: bool,
    ) -> None:
        """Persistent reasoning-only responses retry once, then fail terminally.

        At least one retry must be observable in the event stream; when
        retries still fail, the run ends through the normal Error path
        (session_id populated, error marker persisted for trajectory export)
        and never publishes a blank final AgentMessage.
        """
        ctx = await create_test_engine(
            [
                MockResponse(reasoning_text="internal reasoning only"),
                MockResponse(reasoning_text="still just reasoning"),
            ],
            tmp_path,
            stream=stream,
        )
        retries: list[InvocationRetryAttempt] = []
        errors: list[Error] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        async def _on_error(ev: Error) -> None:
            errors.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)
        await ctx.bus.subscribe(Error, _on_error)

        try:
            await ctx.send_message("go")

            # One initial call + one retry; the identical-reason short-circuit
            # then gives up terminally instead of burning the remaining budget.
            assert ctx.mock_client.call_count == 2
            assert len(retries) == 1
            assert len(errors) == 1
            assert errors[0].code == "executor_error"
            assert errors[0].message == NO_VISIBLE_OUTPUT_REASON
            assert errors[0].session_id == ctx.session_id
            assert extract_final_messages(ctx.events) == []
            assert ctx.engine.state is EngineState.FAILED

            raw_messages = await ctx.get_session_messages()
            marker = raw_messages[1]
            assert marker.get("additional_properties", {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
            assert marker.get("additional_properties", {}).get("_interrupted_by") == "error"
            assert marker.get("contents", [{}])[0].get("text") == NO_VISIBLE_OUTPUT_REASON
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_whitespace_response_retried(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Whitespace-only text is retried; good response becomes final."""
        ctx = await create_test_engine(
            [
                MockResponse(text="\n\n\n"),  # whitespace-only — invalid
                MockResponse(text="real answer"),
            ],
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            assert ctx.mock_client.call_count == 2
            assert len(retries) == 1
            assert "whitespace" in retries[0].message.lower()
            assert extract_final_messages(ctx.events) == ["real answer"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_leaked_tool_call_marker_retried(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Leaked ``minimax:tool_call`` marker → retry → clean final."""
        leaked_text = 'Let me check.\nminimax:tool_call {"name":"x"} </minimax:tool_call>'
        ctx = await create_test_engine(
            [
                MockResponse(text=leaked_text),
                MockResponse(text="clean answer"),
            ],
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            assert ctx.mock_client.call_count == 2
            assert len(retries) == 1
            assert "leaked tool-call" in retries[0].message
            finals = extract_final_messages(ctx.events)
            assert finals == ["clean answer"]
            # Crucially: the leaked text must NOT appear in any AgentMessage.
            assert not any(
                "minimax:tool_call" in e.text
                for e in ctx.events
                if (isinstance(e, InvocationMessage) and e.origin.kind == "turn")
            )
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_retry_then_tool_loop_still_runs(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Bad final response → retry → tool-calling response → tool runs → clean final.

        Guards against the validation middleware disrupting the tool loop:
        the retry should deliver a tool-calling response, the tool must
        execute exactly once, and the downstream final response should
        land as the agent message.
        """
        ctx = await create_test_engine(
            [
                # Attempt 1: bad (empty) final response.
                MockResponse(text=""),
                # Attempt 2 (retry): intermediate text + tool call.
                MockResponse(
                    text="working on it...",
                    tool_calls=[("echo", "call_1", {"message": "hi"})],
                ),
                # After tool result: clean final answer.
                MockResponse(text="done"),
            ],
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            # 3 mock calls: bad + tool-call + final.
            assert ctx.mock_client.call_count == 3
            assert len(retries) == 1

            # Tool executed exactly once — no double-invocation from retries.
            starts = extract_tool_starts(ctx.events)
            assert [s["tool_name"] for s in starts] == ["echo"]
            results = extract_tool_results(ctx.events)
            assert len(results) == 1
            assert "echo: hi" in results[0]["result"]

            # Final message is the post-tool answer.
            assert extract_final_messages(ctx.events) == ["done"]

            # Intermediate text from the GOOD retry still flows through —
            # "working on it..." came from the (post-retry) tool-call
            # response and should appear as an intermediate AgentMessage.
            intermediates = extract_intermediate_messages(ctx.events)
            assert any("working on it" in t for t in intermediates)
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_exhaustion_completes_run_without_raising(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """All retries invalid → run finishes IDLE with the last (still-empty) response as final.

        Uses ``_varied_invalid_mocks`` (distinct reason per attempt) so
        the loop reaches MAX_RETRIES exhaustion instead of taking the
        fail-fast short-circuit; ``last=MockResponse(text="")`` pins the
        final attempt to empty so the give-up cleanup runs on it.
        """
        # One initial + MAX_RETRIES retries = MAX_RETRIES + 1 total attempts.
        ctx = await create_test_engine(
            _varied_invalid_mocks(MAX_RETRIES + 1, last=MockResponse(text="")),
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            assert ctx.mock_client.call_count == MAX_RETRIES + 1
            # Retry events fire only for the retries themselves
            # (MAX_RETRIES), not for the final give-up attempt.
            assert len(retries) == MAX_RETRIES
            # Engine ends cleanly.
            assert ctx.engine.state is EngineState.IDLE

            # Attempt numbers are 1-based and ascending.
            assert [r.attempt for r in retries] == list(range(1, MAX_RETRIES + 1))
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_exhaustion_does_not_persist_empty_assistant_message(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """All retries return empty content → exhaustion → saved session
        must NOT contain a stray ``content: []`` assistant message right
        before the engine's turn marker.

        Regression: prior to the give-up scrub, CompressibleHistoryProvider.after_run
        appended the exhausted (empty) message to history; ``_post_run``
        then appended the turn marker.  ``session.json`` ended up with two
        consecutive assistant messages — first ``content: []`` from the
        bad model output, then the turn marker
        from the engine — which broke downstream replay assumptions.
        """
        ctx = await create_test_engine(
            [MockResponse(text="")] * (MAX_RETRIES + 1),
            tmp_path,
            stream=False,
        )

        try:
            await ctx.send_message("go")

            raw = await ctx.get_session_messages()
            # Find every assistant message in the saved history.
            assistant_msgs = [m for m in raw if m.get("role") == "assistant"]

            # Exactly one assistant message — the turn marker.  Without
            # the fix there would be two: the empty-contents bad message
            # plus the turn marker.
            assert len(assistant_msgs) == 1, (
                f"Expected exactly 1 assistant message (the turn marker), got {len(assistant_msgs)}: {assistant_msgs}"
            )
            assert (
                assistant_msgs[0].get("additional_properties", {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
            )
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_exhaustion_does_not_persist_whitespace_assistant_message(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Whitespace-only exhaustion is the same shape as empty-contents
        from the persistence side — the bad message must not survive."""
        ctx = await create_test_engine(
            [MockResponse(text="\n\n\n")] * (MAX_RETRIES + 1),
            tmp_path,
            stream=False,
        )

        try:
            await ctx.send_message("go")

            raw = await ctx.get_session_messages()
            assistant_msgs = [m for m in raw if m.get("role") == "assistant"]
            assert len(assistant_msgs) == 1
            assert (
                assistant_msgs[0].get("additional_properties", {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
            )
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_exhaustion_with_leaked_marker_persists_text(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """A leaked-marker exhaustion preserves the (still-bad) text in
        history so the user can see what the model actually emitted —
        only structurally-empty messages are dropped."""
        leaked = "answer: minimax:tool_call {} </minimax:tool_call>"
        ctx = await create_test_engine(
            [MockResponse(text=leaked)] * (MAX_RETRIES + 1),
            tmp_path,
            stream=False,
        )

        try:
            await ctx.send_message("go")

            raw = await ctx.get_session_messages()
            # Exactly one non-marker assistant message + the turn marker.
            assistant_msgs = [m for m in raw if m.get("role") == "assistant"]
            non_markers = [
                m
                for m in assistant_msgs
                if m.get("additional_properties", {}).get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
            ]
            assert len(non_markers) == 1
            text = "".join(
                c.get("text", "") or ""
                for c in non_markers[0].get("contents", [])
                if isinstance(c, dict) and c.get("type") == "text"
            )
            # Bad text preserved verbatim — only function_call would be stripped.
            assert "minimax" in text
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_exhaustion_with_function_call_does_not_loop(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Exhausted retries on an invalid+tool-call response must NOT re-loop.

        Regression: previously, an invalid response carrying real
        ``function_call`` items would be returned as-is on the give-up
        path; the Chrys tool loop would then execute the call,
        iterate, and re-trigger the full validation cycle every
        iteration — never terminating.  The fix strips function_call
        contents on exhaustion so ``_extract_function_calls`` returns
        ``[]`` and the loop exits.
        """
        leaked_text = "pre\n<function_call>garbage</function_call>\npost"
        bad_with_fc = MockResponse(
            text=leaked_text,
            tool_calls=[("echo", "call_final", {"message": "x"})],
        )
        # Use varied leading reasons + leaked-marker+function_call as the
        # final attempt so the loop reaches MAX_RETRIES exhaustion (not
        # the fail-fast short-circuit) and the give-up cleanup must
        # strip the function_call from the *exhausted* response.  Supply
        # exactly MAX_RETRIES + 1 responses: any regression that makes
        # extra calls would hit the mock's fallback ("[No more scripted
        # responses]" — valid + no tool_calls) and bump call_count past
        # MAX_RETRIES + 1, failing the count assertion below.
        ctx = await create_test_engine(
            _varied_invalid_mocks(MAX_RETRIES + 1, last=bad_with_fc),
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            # Exactly one validation cycle: 1 initial + MAX_RETRIES retries.
            # If the bug regressed the count would be a multiple of this
            # (e.g. 2x or 3x as the tool loop re-iterates).
            assert ctx.mock_client.call_count == MAX_RETRIES + 1, (
                f"Expected exactly {MAX_RETRIES + 1} LLM calls "
                f"(one validation cycle), got {ctx.mock_client.call_count} — "
                "the tool loop is iterating and re-triggering validation."
            )
            assert len(retries) == MAX_RETRIES
            # Engine ended cleanly, not stuck in a loop.
            assert ctx.engine.state is EngineState.IDLE

            # The function call from the exhausted response must not
            # have executed — stripping made it invisible to the loop.
            starts = extract_tool_starts(ctx.events)
            assert starts == [], f"Function calls must not execute on exhaustion, got {starts}"
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_session_persistence_only_has_good_response(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Saved session on disk must contain only the GOOD final response,
        never the bad one(s).  This is how subsequent turns see history —
        if bad attempts leaked, the next turn would read nonsense context.
        """
        leaked = "before\n<function_call>bad</function_call>\nafter"
        ctx = await create_test_engine(
            [
                MockResponse(text=""),  # bad 1
                MockResponse(text=leaked),  # bad 2 (different reason)
                MockResponse(text="the real answer"),
            ],
            tmp_path,
            stream=False,
        )

        try:
            await ctx.send_message("go")
            raw = await ctx.get_session_messages()
            # Concatenate all text across the saved session.
            all_text = ""
            for msg in raw:
                for c in msg.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        all_text += c.get("text", "") or ""

            assert "the real answer" in all_text
            # Bad content must NOT be present anywhere in the saved state.
            assert "function_call" not in all_text
            assert "bad" not in all_text
        finally:
            await ctx.cleanup()


# ===========================================================================
# Main-agent integration — streaming
# ===========================================================================


class TestMainAgentStreaming:
    @pytest.mark.asyncio
    async def test_streaming_bad_then_good_only_good_chunks_visible(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """In streaming mode a bad response must NOT emit any is_final=False
        chunks.  Chunks only appear once the (post-retry) good stream
        replays through the TUI."""
        ctx = await create_test_engine(
            [
                MockResponse(text=""),  # empty — invalid
                MockResponse(text="line1\nline2\n"),  # valid
            ],
            tmp_path,
            stream=True,
        )

        try:
            await ctx.send_message("go")

            # Streaming chunks are is_final=False, is_intermediate=False.
            stream_chunks = [
                e.text
                for e in ctx.events
                if (isinstance(e, InvocationMessage) and e.origin.kind == "turn")
                and not e.is_final
                and not e.is_intermediate
            ]
            # The one streamed snapshot is the good response's text.
            assert stream_chunks == ["line1\nline2\n"]
            # Final event is the good full response.
            assert extract_final_messages(ctx.events) == ["line1\nline2\n"]
            assert ctx.mock_client.call_count == 2
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_leaked_tool_call_retried(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Streaming path with a leaked marker on attempt 1; no chunks from that attempt."""
        leaked = "<tool_use>bad</tool_use>"
        ctx = await create_test_engine(
            [
                MockResponse(text=leaked),
                MockResponse(text="clean text"),
            ],
            tmp_path,
            stream=True,
        )

        try:
            await ctx.send_message("go")

            # No streamed chunk contains the leaked marker.
            stream_chunks = [
                e.text
                for e in ctx.events
                if (isinstance(e, InvocationMessage) and e.origin.kind == "turn")
                and not e.is_final
                and not e.is_intermediate
            ]
            assert stream_chunks == ["clean text"]
            assert not any("tool_use" in c for c in stream_chunks)
            assert extract_final_messages(ctx.events) == ["clean text"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_exhaustion_still_idle(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Streaming exhaustion → run completes (no raise), FSM IDLE.

        Distinct reasons each attempt so the loop reaches MAX_RETRIES
        exhaustion instead of the fail-fast short-circuit.
        """
        ctx = await create_test_engine(
            _varied_invalid_mocks(MAX_RETRIES + 1, last=MockResponse(text="")),
            tmp_path,
            stream=True,
        )

        try:
            await ctx.send_message("go")
            assert ctx.mock_client.call_count == MAX_RETRIES + 1
            assert ctx.engine.state is EngineState.IDLE
            assert extract_final_messages(ctx.events) == [""]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_exhaustion_does_not_persist_empty_message(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Streaming counterpart of the non-streaming persistence test —
        the lazy validation proxy scrubs both buffered updates and the final
        response, so saved history must stay free of empty assistant messages."""
        ctx = await create_test_engine(
            [MockResponse(text="")] * (MAX_RETRIES + 1),
            tmp_path,
            stream=True,
        )

        try:
            await ctx.send_message("go")

            raw = await ctx.get_session_messages()
            assistant_msgs = [m for m in raw if m.get("role") == "assistant"]
            assert len(assistant_msgs) == 1
            assert (
                assistant_msgs[0].get("additional_properties", {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
            )
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_exhaustion_with_function_call_does_not_loop(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Streaming counterpart of the non-streaming exhaustion+function_call test.

        The streaming give-up path replays buffered ``updates`` and returns the
        validated ``final``; both must be stripped of function_call contents so
        the Chrys tool loop exits.
        """
        leaked_text = "<tool_use>garbage</tool_use>"
        bad_with_fc = MockResponse(
            text=leaked_text,
            tool_calls=[("echo", "call_final", {"message": "x"})],
        )
        # Same varied-leading + function_call-on-final pattern as the
        # non-streaming counterpart; see comment there for the rationale.
        ctx = await create_test_engine(
            _varied_invalid_mocks(MAX_RETRIES + 1, last=bad_with_fc),
            tmp_path,
            stream=True,
        )

        try:
            await ctx.send_message("go")

            assert ctx.mock_client.call_count == MAX_RETRIES + 1, (
                f"Streaming path should also exit after one validation cycle, "
                f"got {ctx.mock_client.call_count} calls — regression."
            )
            assert ctx.engine.state is EngineState.IDLE
            # No tool execution from the exhausted streaming response.
            assert extract_tool_starts(ctx.events) == []
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_intermediate_text_fires_once_not_per_retry(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """The mock client's on_intermediate_text_sync (wired via result_hook)
        must fire only for the FINAL accepted attempt, never for dropped
        bad attempts.

        Bad final responses are text-only with no tool calls → they don't
        fire the mock's intermediate-text callback anyway, so we exercise
        the trickier case: a response with BOTH text+tool_call (which
        DOES trigger the sync callback).  The validator's leaked-marker
        rule flags the first of these, we retry with a clean
        text+tool_call, and the mock's sync_cb must fire exactly once —
        for the good attempt — bumping ``batch_id`` appropriately.
        """
        leaked = 'plan: \nminimax:tool_call {"x":1} </minimax:tool_call>'
        ctx = await create_test_engine(
            [
                # Bad: intermediate with leaked markers in text.
                MockResponse(
                    text=leaked,
                    tool_calls=[("echo", "call_1", {"message": "a"})],
                ),
                # Good retry: intermediate with clean text.
                MockResponse(
                    text="real plan: ",
                    tool_calls=[("echo", "call_2", {"message": "b"})],
                ),
                # Final.
                MockResponse(text="done"),
            ],
            tmp_path,
            stream=True,
        )

        try:
            await ctx.send_message("go")

            # Tool should fire TWICE — once per tool_call the LLM emitted
            # (the bad one had a tool call too).  The engine loop submits
            # both attempts' tool calls because validation happens at the
            # final-response level, not per-turn.  Wait — actually, with
            # streaming the bad response WILL be replaced by the good
            # one's stream before the agent loop sees it, so the loop
            # only sees ONE tool_call per turn.
            #
            # So: tool called once, with message="b" (from the good retry).
            starts = extract_tool_starts(ctx.events)
            assert [s["tool_name"] for s in starts] == ["echo"]
            results = extract_tool_results(ctx.events)
            assert len(results) == 1
            assert "echo: b" in results[0]["result"]

            # Final is "done".
            assert extract_final_messages(ctx.events) == ["done"]

            # Intermediate text must come from the GOOD retry, not the leaked one.
            intermediates = extract_intermediate_messages(ctx.events)
            joined_inter = " ".join(intermediates)
            assert "real plan" in joined_inter
            assert "minimax:tool_call" not in joined_inter
        finally:
            await ctx.cleanup()


# ===========================================================================
# Retry event metadata — outer observability invariant
# ===========================================================================


class TestRetryEventPublication:
    @pytest.mark.asyncio
    async def test_retry_attempt_event_carries_session_and_attempt_numbers(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        # Distinct reasons on the two bad attempts so the fail-fast
        # short-circuit does not fire before we observe two retry events.
        ctx = await create_test_engine(
            [
                MockResponse(text=""),  # reason: empty contents
                MockResponse(text="\n\n\n"),  # reason: whitespace text
                MockResponse(text="ok"),
            ],
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            assert len(retries) == 2
            # attempts are 1-based
            assert [r.attempt for r in retries] == [1, 2]
            assert all(r.max_attempts == MAX_RETRIES for r in retries)
            assert all(r.session_id == ctx.session_id for r in retries)
            # Delay seconds reflect the monkeypatched zero schedule.
            assert all(r.delay_seconds == 0 for r in retries)
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_no_retry_event_on_valid_first_try(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """Happy path: first response is valid, no RetryAttempt events fire."""
        ctx = await create_test_engine(
            [MockResponse(text="hello")],
            tmp_path,
            stream=False,
        )
        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        try:
            await ctx.send_message("go")

            assert ctx.mock_client.call_count == 1
            assert retries == []
            assert extract_final_messages(ctx.events) == ["hello"]
        finally:
            await ctx.cleanup()


# ===========================================================================
# Sub-agent integration — InvocationRetryAttempt events
# ===========================================================================
#
# This mirrors the existing ``test_sub_agent_integration`` harness but
# drives a VALIDATION retry (malformed response) rather than a transport
# error.  Validation retries are handled inside the middleware, so the
# sub-agent's outer StreamRetryLoop never sees them — they must surface
# via the middleware's ``publish_retry`` callback as InvocationRetryAttempt
# events keyed to the right invocation_id.


class _SubAgentMiniCtx:
    def __init__(
        self,
        *,
        engine: AgentEngine,
        bus: EventBus,
        events: list[Event],
        main_client: MockChatClient,
        sub_client: MockChatClient,
        session_id: str,
        create_client_patcher: pytest.MonkeyPatch,
    ) -> None:
        self.engine = engine
        self.bus = bus
        self.events = events
        self.main_client = main_client
        self.sub_client = sub_client
        self.session_id = session_id
        self._create_client_patcher = create_client_patcher

    async def cleanup(self) -> None:
        self._create_client_patcher.undo()

    async def wait_for_idle(self, *, timeout: float = ENGINE_TURN_TIMEOUT) -> None:
        """Wait for the complete parent run-task chain, including its save."""
        await await_run_task_chain(
            self.engine, turn_state=self.engine.turns.turn_state, timeout=timeout, expect_installed=True
        )


async def _make_sub_agent_ctx(
    tmp_path: Path,
    *,
    main_responses: list[MockResponse],
    sub_responses: list[MockResponse],
    agent_engine,
) -> _SubAgentMiniCtx:
    """Build an engine with a parent + one sub-agent, each backed by its own
    ``MockChatClient`` routed via ``ModelProfile.id``."""
    events: list[Event] = []

    async def _collect(ev: Event) -> None:
        events.append(ev)

    bus = EventBus()
    for cls in (
        SessionReady,
        InvocationMessage,
        InvocationToolCallStart,
        InvocationToolCallResult,
        InvocationStarted,
        InvocationRetryAttempt,
    ):
        await bus.subscribe(cls, _collect)

    model_registry = ModelProfileRegistry()
    model_registry.register(
        ModelProfile(
            id="main-mock",
            name="main",
            provider="mock",
            model_id="mock",
            stream=False,
            max_context_tokens=100_000,
        )
    )
    model_registry.register(
        ModelProfile(
            id="sub-mock",
            name="sub",
            provider="mock",
            model_id="mock",
            stream=False,
            max_context_tokens=100_000,
        )
    )
    # Validation retries must not spend their turn budget on advisory disk scans.
    settings = Settings(model_profile="main-mock", workspace_change_notice=False)
    store = JsonFileStateStore(tmp_path / "sessions")

    parent_profile = AgentProfile(
        name="Parent",
        instructions="Use Explore.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
        sub_agents=SubAgentsConfig(
            max_total_concurrency=2,
            agents=[SubAgentRef(profile="Explore", tool_name="Explore")],
        ),
    )
    sub_profile = AgentProfile(
        name="Explore",
        instructions="Explore and summarise.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
        model=ModelConfig(profile_id="sub-mock"),
    )
    agent_registry = AgentProfileRegistry()
    agent_registry.register(parent_profile)
    agent_registry.register(sub_profile)

    main_client = MockChatClient(responses=main_responses)
    sub_client = MockChatClient(responses=sub_responses)

    async def _patched_create(p: Any = None, **kw: Any) -> MockChatClient:
        profile_id = getattr(p, "id", "") or ""
        client = sub_client if profile_id == "sub-mock" else main_client
        client._on_intermediate_text_async = kw.get("on_intermediate_text_async")
        client._on_intermediate_text_sync = kw.get("on_intermediate_text_sync")
        return client

    create_client_patcher = pytest.MonkeyPatch()
    create_client_patcher.setattr(builder_module, "create_client", _patched_create)
    sub_create_client_patcher = pytest.MonkeyPatch()
    sub_create_client_patcher.setattr(sub_agent_module, "create_client", _patched_create)

    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=agent_registry,
        model_registry=model_registry,
        state_store=store,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    try:
        await engine.start(parent_profile)
    except Exception:
        create_client_patcher.undo()
        sub_create_client_patcher.undo()
        raise

    # Restore sub_agent_module so any further sub-agent builds during the
    # test (there shouldn't be any) use the real path; the builder_module
    # patch stays live until cleanup.
    sub_create_client_patcher.undo()

    return _SubAgentMiniCtx(
        engine=engine,
        bus=bus,
        events=events,
        main_client=main_client,
        sub_client=sub_client,
        session_id=engine.session.session_id or "",
        create_client_patcher=create_client_patcher,
    )


class TestSubAgentValidationRetry:
    @pytest.mark.asyncio
    async def test_sub_agent_bad_response_emits_sub_agent_retry_attempt(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        agent_engine,
    ) -> None:
        """Sub-agent's first response is malformed; middleware retries
        inside the sub-agent's run and publishes ``InvocationRetryAttempt``
        events — NOT a main-agent ``RetryAttempt`` event, since the
        parent never sees the bad response."""
        ctx = await _make_sub_agent_ctx(
            tmp_path,
            main_responses=[
                MockResponse(tool_calls=[("Explore", "call_1", {"prompt": "explore"})]),
                MockResponse(text="parent final"),
            ],
            sub_responses=[
                MockResponse(text=""),  # invalid — empty
                MockResponse(text="sub-agent final summary"),
            ],
            agent_engine=agent_engine,
        )

        try:
            await ctx.bus.publish(UserMessage(text="go"))
            await ctx.wait_for_idle()

            # Poll the asserted conditions (sub-agent call count + retry event
            # presence) rather than relying on a fixed settle — both are
            # published from async run/sub-agent tasks.
            await wait_for(
                lambda: (
                    ctx.sub_client.call_count == 2
                    and any(
                        (isinstance(e, InvocationRetryAttempt) and e.origin.kind == "sub_agent") for e in ctx.events
                    )
                ),
                timeout=ENGINE_TURN_TIMEOUT,
                description="sub-agent validation retry and second model call",
            )

            # Sub-agent was called twice (bad + good).
            assert ctx.sub_client.call_count == 2

            # A InvocationRetryAttempt event fired, with invocation_id + agent_name.
            sub_retries = [
                e for e in ctx.events if (isinstance(e, InvocationRetryAttempt) and e.origin.kind == "sub_agent")
            ]
            assert len(sub_retries) == 1
            assert sub_retries[0].origin.invocation_id  # non-empty
            assert sub_retries[0].agent_name == "Explore"
            assert "Invalid response" in sub_retries[0].message
            assert sub_retries[0].session_id == ctx.session_id

            # NO main-agent RetryAttempt event — parent never saw the bad.
            main_retries = [
                e for e in ctx.events if (isinstance(e, InvocationRetryAttempt) and e.origin.kind == "turn")
            ]
            assert main_retries == []

            # Sub-agent's tool_result carries the GOOD summary.
            tool_results = [
                e
                for e in ctx.events
                if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
            ]
            assert tool_results and "sub-agent final summary" in tool_results[-1].result

            # Parent produced its final response normally.
            finals = [
                e for e in ctx.events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final
            ]
            assert any("parent final" in f.text for f in finals)

            # Engine ended IDLE.
            assert ctx.engine.state is EngineState.IDLE
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_sub_agent_retry_event_invocation_id_matches_invocation_start(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
        agent_engine,
    ) -> None:
        """The ``invocation_id`` on the retry event must match the one the
        TUI uses to render the sub-agent card — i.e. the same one the
        controller carries."""
        ctx = await _make_sub_agent_ctx(
            tmp_path,
            main_responses=[
                MockResponse(tool_calls=[("Explore", "call_1", {"prompt": "x"})]),
                MockResponse(text="done"),
            ],
            sub_responses=[
                MockResponse(text=""),
                MockResponse(text="ok"),
            ],
            agent_engine=agent_engine,
        )

        try:
            await ctx.bus.publish(UserMessage(text="go"))
            await ctx.wait_for_idle()

            # Poll for the retry event (published from an async sub-agent task)
            # instead of leaning on a fixed settle.
            await wait_for(
                lambda: any(
                    (isinstance(e, InvocationRetryAttempt) and e.origin.kind == "sub_agent") for e in ctx.events
                ),
                timeout=ENGINE_TURN_TIMEOUT,
                description="sub-agent retry event",
            )

            sub_retries = [
                e for e in ctx.events if (isinstance(e, InvocationRetryAttempt) and e.origin.kind == "sub_agent")
            ]
            assert len(sub_retries) == 1
            retry_inv = sub_retries[0].origin.invocation_id

            # The retry event must carry the SAME invocation_id the
            # ``InvocationStarted`` bound to the widget — that id is
            # what routes retry banners to the right sub-agent card.
            starts = [
                e
                for e in ctx.events
                if (isinstance(e, InvocationStarted) and e.origin.kind == "sub_agent") and e.tool_name == "Explore"
            ]
            assert len(starts) == 1
            assert retry_inv == starts[0].origin.invocation_id
        finally:
            await ctx.cleanup()


# ===========================================================================
# Outer-middleware invariance — the dog that didn't bark
# ===========================================================================


class TestOuterMiddlewareNotPerturbed:
    @pytest.mark.asyncio
    async def test_session_ready_fires_once_despite_retries(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """SessionReady fires at engine startup — independently of the LLM
        retry loop — so retries must not duplicate or delay it."""
        ctx = await create_test_engine(
            [
                MockResponse(text=""),
                MockResponse(text=""),
                MockResponse(text="ok"),
            ],
            tmp_path,
            stream=False,
        )

        try:
            # SessionReady was already published by create_test_engine.
            ready = await wait_for_event(ctx.events, SessionReady, timeout=2.0)
            assert len(ready) == 1

            await ctx.send_message("go")
            # Still exactly one SessionReady after the retry storm.
            ready_after = [e for e in ctx.events if isinstance(e, SessionReady)]
            assert len(ready_after) == 1
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_mock_call_count_matches_middleware_attempts(
        self,
        tmp_path: Path,
        fast_validation_backoff: None,
    ) -> None:
        """The LLM is called exactly once per middleware attempt — outer
        middleware doesn't amplify (or swallow) the retry loop.

        Distinct reason on each bad attempt so the fail-fast short-circuit
        does not fire and we exercise three real retries before recovery.
        """
        ctx = await create_test_engine(
            [
                MockResponse(text=""),  # reason: empty contents
                MockResponse(text="\n\n\n"),  # reason: whitespace text
                MockResponse(text="minimax:tool_call {} </minimax:tool_call>"),  # leaked marker
                MockResponse(text="done"),
            ],
            tmp_path,
            stream=False,
        )

        try:
            await ctx.send_message("go")
            # 3 bad + 1 good = 4 mock calls.
            assert ctx.mock_client.call_count == 4
            assert extract_final_messages(ctx.events) == ["done"]
        finally:
            await ctx.cleanup()
