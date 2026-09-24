# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for line-by-line streaming of the final agent response.

In streaming mode (``CHRYS_STREAM=true``) the executor emits per-line
``AgentMessage(is_final=False)`` events as the final LLM response's text
arrives, so the TUI progressively reveals the answer.  Intermediate text
(text emitted alongside ``function_call`` content) is NOT streamed — it
flows through the existing non-streamed intermediate-text handler.

Covers these edge cases:
- Multi-line final text with trailing newline
- Multi-line final text *without* trailing newline (last line unterminated)
- Single-line final text (no newline at all)
- Empty final text
- Intermediate text before a tool call is not streamed
- Multiple intermediate responses then a final are all handled
- Mock streaming order (function_call chunk emitted before text chunk in
  the same LLM response) still discards the intermediate text
"""

from __future__ import annotations

import pytest

from chrys.foundation.events.types import InvocationMessage
from chrys.service.llm.mock import MockResponse
from tests.support.pipeline_helpers import create_test_engine


def _streaming_chunks(events: list) -> list[str]:
    """Extract ``is_final=False`` AgentMessage texts (the progressive reveal)."""
    return [
        e.text
        for e in events
        if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and not e.is_final and not e.is_intermediate
    ]


def _finals(events: list) -> list[str]:
    return [
        e.text
        for e in events
        if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final and not e.is_intermediate
    ]


def _intermediates(events: list) -> list[str]:
    return [
        e.text for e in events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_intermediate
    ]


class TestStreamingLineByLine:
    """Final-response text is revealed line by line via is_final=False events."""

    @pytest.mark.asyncio
    async def test_multiline_with_trailing_newline(self, tmp_path):
        ctx = await create_test_engine(
            [MockResponse(text="line1\nline2\nline3\n")],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            # Each complete line emits a cumulative snapshot.
            assert chunks == [
                "line1\n",
                "line1\nline2\n",
                "line1\nline2\nline3\n",
            ]
            assert _finals(ctx.events) == ["line1\nline2\nline3\n"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multiline_without_trailing_newline(self, tmp_path):
        """The unterminated final line must also be emitted as is_final=False."""
        ctx = await create_test_engine(
            [MockResponse(text="line1\nline2\nline3")],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            # Two \n-terminated lines, plus the unterminated tail.
            assert chunks == [
                "line1\n",
                "line1\nline2\n",
                "line1\nline2\nline3",
            ]
            assert _finals(ctx.events) == ["line1\nline2\nline3"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_single_line_no_newline(self, tmp_path):
        """A single unterminated line still emits one streaming event."""
        ctx = await create_test_engine(
            [MockResponse(text="Hello, world!")],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            assert _streaming_chunks(ctx.events) == ["Hello, world!"]
            assert _finals(ctx.events) == ["Hello, world!"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_crlf_line_endings(self, tmp_path):
        """Windows-style CRLF line endings should each emit one streaming event."""
        ctx = await create_test_engine(
            [MockResponse(text="line1\r\nline2\r\nline3\r\n")],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            assert chunks == [
                "line1\r\n",
                "line1\r\nline2\r\n",
                "line1\r\nline2\r\nline3\r\n",
            ]
            assert _finals(ctx.events) == ["line1\r\nline2\r\nline3\r\n"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_crlf_without_trailing_newline(self, tmp_path):
        """CRLF separators with an unterminated tail."""
        ctx = await create_test_engine(
            [MockResponse(text="line1\r\nline2\r\nline3")],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            assert chunks == [
                "line1\r\n",
                "line1\r\nline2\r\n",
                "line1\r\nline2\r\nline3",
            ]
            assert _finals(ctx.events) == ["line1\r\nline2\r\nline3"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_cr_line_endings(self, tmp_path):
        """Classic Mac-style CR-only separators should each emit one streaming event."""
        ctx = await create_test_engine(
            [MockResponse(text="line1\rline2\rline3\r")],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            assert chunks == [
                "line1\r",
                "line1\rline2\r",
                "line1\rline2\rline3\r",
            ]
            assert _finals(ctx.events) == ["line1\rline2\rline3\r"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_cr_without_trailing_newline(self, tmp_path):
        """CR separators with an unterminated tail — the critical regression case."""
        ctx = await create_test_engine(
            [MockResponse(text="line1\rline2\rline3")],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            assert chunks == [
                "line1\r",
                "line1\rline2\r",
                "line1\rline2\rline3",
            ]
            assert _finals(ctx.events) == ["line1\rline2\rline3"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_empty_final_text(self, tmp_path, monkeypatch):
        """Empty final response produces no streaming events.

        ``ResponseValidationMiddleware`` flags empty text as invalid and
        retries up to ``MAX_RETRIES`` times before returning the last
        (still-empty) response as-is.  Supply enough empty responses to
        cover every attempt so the mock doesn't fall back to its
        "[No more scripted responses]" canned text — the final
        (exhausted) empty reply should still produce zero streamed chunks.
        """
        from chrys.service.agent_middleware.response_validation import (
            MAX_RETRIES,
            ResponseValidationMiddleware,
        )

        # Flatten validation backoff so the test stays fast.
        monkeypatch.setattr(ResponseValidationMiddleware, "_BACKOFF_SCHEDULE", (0.0,))

        ctx = await create_test_engine(
            [MockResponse(text="")] * (MAX_RETRIES + 1),
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            assert _streaming_chunks(ctx.events) == []
            assert _finals(ctx.events) == [""]
        finally:
            await ctx.cleanup()


class TestStreamingSkipsIntermediate:
    """Intermediate text (alongside function_calls) is not streamed."""

    @pytest.mark.asyncio
    async def test_intermediate_text_is_not_streamed(self, tmp_path):
        """Text emitted alongside tool calls goes through the intermediate
        handler, not the streaming path."""
        ctx = await create_test_engine(
            [
                # LLM response 1: text + tool call — intermediate
                MockResponse(
                    text="Let me check first\n",
                    tool_calls=[("echo", "c1", {"message": "x"})],
                ),
                # LLM response 2: pure text — the final answer
                MockResponse(text="All done\n"),
            ],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            # Only the FINAL response streams.  The intermediate "Let me
            # check first\n" must NOT appear in streaming chunks.
            assert "Let me check first\n" not in chunks
            assert chunks == ["All done\n"]

            # Intermediate text still reaches the TUI via the non-streamed
            # intermediate handler.
            assert "Let me check first\n" in _intermediates(ctx.events)
            assert _finals(ctx.events) == ["All done\n"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_intermediates_then_final(self, tmp_path):
        """Two intermediate responses in a row, then a final.  Only the
        final streams; both intermediates are discarded from the buffer."""
        ctx = await create_test_engine(
            [
                MockResponse(
                    text="First thought\n",
                    tool_calls=[("echo", "c1", {"message": "a"})],
                ),
                MockResponse(
                    text="Second thought\n",
                    tool_calls=[("concat", "c2", {"a": "x", "b": "y"})],
                ),
                MockResponse(text="Answer line 1\nAnswer line 2"),
            ],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            chunks = _streaming_chunks(ctx.events)
            # Only the final-response lines stream.
            assert chunks == [
                "Answer line 1\n",
                "Answer line 1\nAnswer line 2",
            ]
            # Neither intermediate text leaks into streaming chunks.
            assert not any("thought" in c for c in chunks)
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_tool_only_response_then_final(self, tmp_path):
        """A pure-tool-call response (no text at all) followed by a final."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("echo", "c1", {"message": "x"})]),
                MockResponse(text="All done"),
            ],
            tmp_path,
            stream=True,
        )
        try:
            await ctx.send_message("go")
            assert _streaming_chunks(ctx.events) == ["All done"]
            assert _finals(ctx.events) == ["All done"]
        finally:
            await ctx.cleanup()


class TestStreamingDisabled:
    """With CHRYS_STREAM=false the executor never emits is_final=False text."""

    @pytest.mark.asyncio
    async def test_no_streaming_events_when_disabled(self, tmp_path):
        ctx = await create_test_engine(
            [MockResponse(text="line1\nline2\nline3")],
            tmp_path,
            stream=False,
        )
        try:
            await ctx.send_message("go")
            assert _streaming_chunks(ctx.events) == []
            assert _finals(ctx.events) == ["line1\nline2\nline3"]
        finally:
            await ctx.cleanup()
