# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Scrubbing done by the response-validation middleware.

Two scrub paths: the give-up path (retries exhausted) strips ``function_call`` contents and
empty assistant messages so the tool loop terminates; the valid path drops leading empty
assistant messages while keeping the accepted response intact.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from chrys.service.agent_middleware.response_validation import (
    MAX_RETRIES,
    ResponseValidationMiddleware,
)
from chrys.service.agent_middleware.validators import (
    DefaultResponseValidator,
)
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _FakeCallNext,
    _final_response,
    _make_context,
    _semantic_updates,
    _varied_bads,
)

# ---------------------------------------------------------------------------
# Give-up path: strip function_call contents and empty assistant messages
# ---------------------------------------------------------------------------


class TestGiveUpScrubbing:
    """The give-up path must strip ``function_call`` contents from the
    exhausted response so the Chrys tool loop sees no calls to
    execute and exits — without this the loop would iterate, re-trigger
    the full validation cycle on the next call, and never terminate
    when the model is stuck in a leaked-marker + function_call state.
    """

    async def test_non_stream_exhaustion_strips_function_call(self) -> None:
        # Use varied reasons across the leading attempts so the fail-fast
        # short-circuit does not fire — we want to actually exhaust
        # MAX_RETRIES.  The final attempt is the leaked-marker +
        # function_call shape we want to verify gets stripped.
        bad_with_fc = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext(_varied_bads(MAX_RETRIES + 1, last=bad_with_fc), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == MAX_RETRIES + 1
        assert isinstance(ctx.result, ChatResponse)
        # function_call stripped, text preserved.
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" not in types
        assert "text" in types

    async def test_exhaustion_strips_actionable_call_but_preserves_informational_transcript(self) -> None:
        hosted_call = Content.from_function_call(
            call_id="hosted-1",
            name="web_search",
            arguments={"query": "chrys"},
            informational_only=True,
        )
        bad = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="local-1", name="echo", arguments={"message": "x"}),
                hosted_call,
            ]
        )
        fake = _FakeCallNext(_varied_bads(MAX_RETRIES + 1, last=bad), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        await ResponseValidationMiddleware(backoff_schedule=[0.0]).process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        calls = [content for content in ctx.result.messages[0].contents if content.type == "function_call"]
        assert calls == [hosted_call]

    async def test_stream_exhaustion_strips_function_call(self) -> None:
        bad_with_fc = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext(_varied_bads(MAX_RETRIES + 1, last=bad_with_fc), stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)

        # Drain the replay — function_call must NOT appear in any update.
        updates = [u async for u in ctx.result]
        assert fake.call_count == MAX_RETRIES + 1
        for u in updates:
            for c in u.contents or []:
                assert c.type != "function_call", "Replay must not yield function_call updates"

        # Finalised response must also be stripped.
        final = await ctx.result.get_final_response()
        for msg in final.messages:
            for c in msg.contents:
                assert c.type != "function_call", "Final response must not carry function_call"

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_exhaustion_drops_whitespace_only_message(self, stream: bool) -> None:
        """Whitespace-only text exhaustion drops the message — same root
        cause as ``content: []``: a bad message would otherwise land in
        history next to the turn marker as ``content: [{text: ""}]``.
        """
        bad = _assistant([Content.from_text("\n\n  \t\n")])
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=stream)
        ctx = _make_context(stream=stream)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        final = await _final_response(ctx, stream=stream)
        assert final.messages == []

    async def test_exhaustion_text_with_leaked_marker_keeps_message(self) -> None:
        """A leaked-marker exhaustion preserves the (still-bad) text so
        the user can see what the model emitted.  Only function_call
        contents are stripped; the text message stays in history."""
        bad = _assistant(
            [
                Content.from_text("Sure, calling it: minimax:tool_call {} </minimax:tool_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        # Message preserved, function_call gone, text intact.
        assert len(ctx.result.messages) == 1
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" not in types
        assert "text" in types
        assert "minimax" in (ctx.result.messages[0].contents[0].text or "")

    async def test_exhaustion_drops_only_empty_message_in_multi_message(self) -> None:
        """Multi-message response: empty assistant message dropped,
        valid one preserved.  Catches the rare provider/streaming case
        where ``_process_update`` creates a placeholder
        ``Message("assistant", [])`` that never gets populated."""

        empty = Message(role="assistant", contents=[])
        valid = Message(role="assistant", contents=[Content.from_text("real")])
        # Build a response where validation flags the LAST message
        # (empty contents wins) — so the give-up path runs.
        bad = ChatResponse(messages=[valid, empty], finish_reason="stop")
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        # Empty message dropped, valid message preserved.
        assert len(ctx.result.messages) == 1
        assert ctx.result.messages[0].contents[0].text == "real"

    async def test_exhaustion_preserves_non_assistant_messages(self) -> None:
        """Tool / user messages embedded in a malformed response stay put —
        only assistant outputs are scrubbed."""

        tool_msg = Message(
            role="tool",
            contents=[Content.from_function_result(call_id="c1", result="result text")],
        )
        bad_assistant = Message(role="assistant", contents=[])
        bad = ChatResponse(messages=[tool_msg, bad_assistant], finish_reason="stop")
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        roles = [m.role for m in ctx.result.messages]
        assert roles == ["tool"]


# ---------------------------------------------------------------------------
# Valid path: drop leading empty assistant messages, keep the accepted response
# ---------------------------------------------------------------------------


class TestValidPathScrubbing:
    """Successful responses are scrubbed too: leading empty assistant messages are dropped
    without touching the accepted final message."""

    async def test_valid_path_drops_leading_empty_assistant_non_stream(self) -> None:
        """[empty, valid] response shape — validation passes (only the
        last assistant message is checked) BUT the leading empty message
        must still be scrubbed before persistence.

        Without scrubbing on the valid path, HistoryProvider.after_run
        would persist BOTH messages: a stray ``content: []`` followed by
        the real reply.
        """
        empty = Message(role="assistant", contents=[])
        valid = Message(role="assistant", contents=[Content.from_text("real answer")])
        good = ChatResponse(messages=[empty, valid], finish_reason="stop")
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        # Validation passed first try — no retries.
        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        # Empty dropped, valid preserved.
        assert len(ctx.result.messages) == 1
        assert ctx.result.messages[0].contents[0].text == "real answer"

    async def test_valid_path_drops_leading_whitespace_assistant_non_stream(self) -> None:
        """Same shape but with whitespace-only text instead of empty contents."""
        ws = Message(role="assistant", contents=[Content.from_text("\n\n")])
        valid = Message(role="assistant", contents=[Content.from_text("done")])
        good = ChatResponse(messages=[ws, valid], finish_reason="stop")
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        assert len(ctx.result.messages) == 1
        assert ctx.result.messages[0].contents[0].text == "done"

    async def test_valid_path_drops_leading_empty_assistant_stream(self) -> None:
        """Streaming counterpart of the [empty, valid] non-stream case.

        A consumer can rebuild a response from replayed updates with
        ``ChatResponse.from_updates(updates)``. Without filtering
        updates on the valid path, the rebuild would re-create the
        empty leading assistant message even though we cleaned
        ``final.messages``.
        """
        # Build a stream whose updates produce [empty assistant, valid assistant].
        # Two distinct message_ids force ``_process_update`` to create
        # two messages.
        empty_upd = ChatResponseUpdate(role="assistant", message_id="m1", contents=[])
        valid_upd = ChatResponseUpdate(role="assistant", message_id="m2", contents=[Content.from_text("real")])

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield empty_upd
            yield valid_upd

        # Finaliser mirrors what ``ChatResponse.from_updates`` would produce.
        final_response = ChatResponse(
            messages=[
                Message(role="assistant", message_id="m1", contents=[]),
                Message(role="assistant", message_id="m2", contents=[Content.from_text("real")]),
            ],
            finish_reason="stop",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        # Replayed updates: empty filtered out, only valid yielded so a
        # ChatResponse.from_updates rebuild stays clean.
        replayed = [u async for u in ctx.result]
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 1
        assert semantic[0].contents[0].text == "real"
        # Final (the inner replay's finaliser path) is also clean.
        rebuilt = await ctx.result.get_final_response()
        assert len(rebuilt.messages) == 1
        assert rebuilt.messages[0].contents[0].text == "real"

    async def test_valid_path_keeps_metadata_terminal_update_in_filled_group(self) -> None:
        """Real providers emit a terminal "stop" update: empty contents +
        ``finish_reason="stop"``.  When the same ``(role, message_id)``
        group already received text, that group is non-empty and ALL
        its updates — including the terminal one — are kept so the
        rebuilt response retains ``finish_reason`` / ``response_id``.
        """
        text_upd = ChatResponseUpdate(
            role="assistant",
            message_id="m1",
            contents=[Content.from_text("hello")],
        )
        terminal_upd = ChatResponseUpdate(
            role="assistant",
            message_id="m1",
            contents=[],
            finish_reason="stop",
            response_id="resp-123",
        )

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield text_upd
            yield terminal_upd

        final_response = ChatResponse(
            messages=[Message(role="assistant", message_id="m1", contents=[Content.from_text("hello")])],
            finish_reason="stop",
            response_id="resp-123",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        replayed = [u async for u in ctx.result]
        # Both updates kept — the terminal one preserves response-level
        # metadata that ``_process_update`` propagates onto the rebuild.
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 2
        assert semantic[1].finish_reason == "stop"
        assert semantic[1].response_id == "resp-123"

    async def test_valid_path_drops_roleless_empty_then_tool_role_starts_new_group(self) -> None:
        """Regression: kernel default role for a placeholder message is
        ``"assistant"``.  A role-less empty leading update therefore
        creates an empty assistant message; a following ``role="tool"``
        update crosses a role boundary and starts a new tool message.

        The grouping logic must initialise its tracked role to
        ``"assistant"`` to mirror this — otherwise the empty leading
        update gets bucketed with the tool group, the assistant group is
        never identified as empty, and ChatResponse.from_updates reproduces
        the stray ``Message("assistant", [])`` we tried to drop.
        """
        # Build the failing shape: empty role-less update + tool update.
        empty_upd = ChatResponseUpdate(contents=[])
        tool_upd = ChatResponseUpdate(
            role="tool",
            contents=[Content.from_function_result(call_id="c1", result="result text")],
        )

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield empty_upd
            yield tool_upd

        # Use a no-op validator so the test isolates the scrub path.
        # ``DefaultResponseValidator`` would actually flag this shape
        # (``_final_assistant_message`` walks back from the end and
        # returns the empty assistant message, then Rule 1 fires for
        # ``empty contents``), forcing the give-up branch.  That branch
        # also runs the scrub, but we want this test to fail loudly if
        # the *valid* branch's scrub regresses — hence the bypass.

        no_op = DefaultResponseValidator(
            disable_empty_contents=True, disable_whitespace_text=True, disable_leaked_tool_call=True
        )

        final_response = ChatResponse(
            messages=[
                Message(role="assistant", contents=[]),
                Message(
                    role="tool",
                    contents=[Content.from_function_result(call_id="c1", result="result text")],
                ),
            ],
            finish_reason="stop",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware(validator=no_op)
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        replayed = [u async for u in ctx.result]
        # Empty assistant update dropped, tool update kept — confirms the
        # role boundary is recognised even when the leading update has
        # no explicit role.
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 1
        assert semantic[0].role == "tool"
        # Final response also has the empty assistant message dropped.
        rebuilt = await ctx.result.get_final_response()
        roles = [m.role for m in rebuilt.messages]
        assert roles == ["tool"]

    async def test_valid_path_drops_empty_assistant_after_tool_via_message_id_boundary(self) -> None:
        """Regression: a new group caused by a ``message_id`` change must
        treat the new placeholder as ``Message("assistant", [])``, matching
        ``_process_update``; inheriting the previous group's role is incorrect.

        Without the reset, a sequence like::

            Update(role="tool", message_id="tool-1", contents=[fr])
            Update(message_id="assistant-1", contents=[])

        would be bucketed as ``[tool, tool]`` (role inherited), so the
        empty trailing group is treated as non-assistant and kept —
        letting ChatResponse.from_updates reproduce the empty assistant
        placeholder we tried to drop.

        Expected: tool update kept, role-less empty new-message_id
        update dropped.
        """

        no_op = DefaultResponseValidator(
            disable_empty_contents=True, disable_whitespace_text=True, disable_leaked_tool_call=True
        )

        tool_upd = ChatResponseUpdate(
            role="tool",
            message_id="tool-1",
            contents=[Content.from_function_result(call_id="c1", result="ok")],
        )
        empty_assistant_upd = ChatResponseUpdate(
            message_id="assistant-1",
            contents=[],
        )

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield tool_upd
            yield empty_assistant_upd

        # Mirror what ``ChatResponse.from_updates`` would build: tool
        # message + empty assistant placeholder (role defaults to
        # "assistant" for the new message_id boundary).
        final_response = ChatResponse(
            messages=[
                Message(
                    role="tool",
                    message_id="tool-1",
                    contents=[Content.from_function_result(call_id="c1", result="ok")],
                ),
                Message(role="assistant", message_id="assistant-1", contents=[]),
            ],
            finish_reason="stop",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware(validator=no_op)
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        replayed = [u async for u in ctx.result]
        # Tool update kept, empty assistant placeholder dropped.
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 1
        assert semantic[0].role == "tool"
        rebuilt = await ctx.result.get_final_response()
        roles = [m.role for m in rebuilt.messages]
        assert roles == ["tool"]

    async def test_valid_path_does_not_drop_message_with_real_text(self) -> None:
        """Sanity check: the valid-path scrub MUST NOT drop messages
        carrying real text or non-text payloads."""
        msg1 = Message(role="assistant", contents=[Content.from_text("first part")])
        msg2 = Message(role="assistant", contents=[Content.from_text("second part")])
        good = ChatResponse(messages=[msg1, msg2], finish_reason="stop")
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        # Both real-text messages preserved.
        assert len(ctx.result.messages) == 2
        assert ctx.result.messages[0].contents[0].text == "first part"
        assert ctx.result.messages[1].contents[0].text == "second part"

    async def test_valid_response_with_function_call_not_stripped(self) -> None:
        """Stripping must only happen on the give-up path, never on success.

        A first-try valid response with function_call must pass through
        unchanged — otherwise the tool loop would never execute any calls.
        """
        good_with_tool = _assistant(
            [
                Content.from_text("calling tool"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext([good_with_tool], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" in types  # still present
