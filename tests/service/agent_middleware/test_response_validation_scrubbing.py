# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Scrubbing done by the response-validation middleware.

An accepted response keeps only its non-empty assistant messages, so a stray
``content: []`` message never lands in persisted history; everything else is
kept as the provider sent it. A stream is never filtered: its updates reach
the consumer unchanged, and only the final response is scrubbed. A rejected
response is raised, never scrubbed into an accepted one.
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
    ResponseValidationMiddleware,
    TerminalResponseValidationError,
)
from chrys.service.agent_middleware.validators import DefaultResponseValidator
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _FakeCallNext,
    _final_response,
    _make_context,
)

_MODES = pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])

# Accepts everything, so a test reaches the accepted-response scrub with a
# shape the default rules would reject.
_NO_OP = DefaultResponseValidator(
    disable_empty_contents=True, disable_whitespace_text=True, disable_unnamed_tool_call=True
)


async def _accept(
    response: ChatResponse, *, stream: bool, validator: DefaultResponseValidator | None = None
) -> ChatResponse:
    fake = _FakeCallNext([response], stream=stream)
    context = _make_context(stream=stream)
    fake.bind(context)
    await ResponseValidationMiddleware(validator=validator).process(context, fake)
    final = await _final_response(context, stream=stream)
    assert fake.call_count == 1
    return final


class TestAcceptedResponseScrubbing:
    @pytest.mark.parametrize(
        "leading",
        [
            pytest.param(Message(role="assistant", contents=[]), id="empty"),
            pytest.param(Message(role="assistant", contents=[Content.from_text("\n\n")]), id="whitespace"),
        ],
    )
    @_MODES
    async def test_a_leading_blank_assistant_message_is_dropped(self, stream: bool, leading: Message) -> None:
        """``[blank, valid]`` passes (only the last assistant message is judged), but the blank one must not persist."""
        valid = Message(role="assistant", contents=[Content.from_text("real answer")])

        final = await _accept(ChatResponse(messages=[leading, valid], finish_reason="stop"), stream=stream)

        assert final.messages == [valid]

    async def test_streamed_updates_pass_through_unfiltered_while_the_final_response_is_scrubbed(self) -> None:
        empty_update = ChatResponseUpdate(role="assistant", message_id="m1", contents=[])
        valid_update = ChatResponseUpdate(role="assistant", message_id="m2", contents=[Content.from_text("real")])

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield empty_update
            yield valid_update

        valid = Message(role="assistant", message_id="m2", contents=[Content.from_text("real")])
        final_response = ChatResponse(
            messages=[Message(role="assistant", message_id="m1", contents=[]), valid],
            finish_reason="stop",
            response_id="resp-123",
        )
        context = _make_context(stream=True)

        async def _call_next() -> None:
            context.result = ResponseStream(_gen(), finalizer=lambda _updates: final_response)

        await ResponseValidationMiddleware().process(context, _call_next)

        assert isinstance(context.result, ResponseStream)
        streamed = [update async for update in context.result]
        assert len(streamed) == 2
        assert streamed[0] is empty_update
        assert streamed[1] is valid_update
        final = await context.result.get_final_response()
        assert final.messages == [valid]
        assert (final.finish_reason, final.response_id) == ("stop", "resp-123")

    @pytest.mark.parametrize("empty_first", [True, False], ids=["empty_then_tool", "tool_then_empty"])
    @_MODES
    async def test_only_assistant_messages_are_scrubbed(self, stream: bool, empty_first: bool) -> None:
        tool = Message(role="tool", contents=[Content.from_function_result(call_id="c1", result="result text")])
        empty = Message(role="assistant", contents=[])
        messages = [empty, tool] if empty_first else [tool, empty]

        final = await _accept(ChatResponse(messages=messages, finish_reason="stop"), stream=stream, validator=_NO_OP)

        assert final.messages == [tool]

    async def test_messages_with_real_text_are_kept(self) -> None:
        first = Message(role="assistant", contents=[Content.from_text("first part")])
        second = Message(role="assistant", contents=[Content.from_text("second part")])

        final = await _accept(ChatResponse(messages=[first, second], finish_reason="stop"), stream=False)

        assert final.messages == [first, second]

    @_MODES
    async def test_a_tool_call_is_kept_whatever_the_text_beside_it_says(self, stream: bool) -> None:
        """Nothing is stripped from an accepted response: the tool loop runs its calls."""
        call = Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"})
        response = _assistant([Content.from_text("<function_call>garbage</function_call>"), call])

        final = await _accept(response, stream=stream)

        [message] = final.messages
        assert [content.type for content in message.contents] == ["text", "function_call"]
        assert message.contents[1] is call

    async def test_a_message_with_non_text_payload_and_blank_text_is_kept(self) -> None:
        call = Content.from_function_call(call_id="c1", name="echo", arguments={})
        message = Message(role="assistant", contents=[Content.from_text(""), call])

        final = await _accept(ChatResponse(messages=[message], finish_reason="stop"), stream=False)

        assert final.messages == [message]


class TestRejectedResponsesAreNotSalvaged:
    @_MODES
    async def test_a_trailing_empty_message_fails_the_response_even_after_real_text(self, stream: bool) -> None:
        """The final assistant message is the reply: an earlier one is not promoted in its place."""
        valid = Message(role="assistant", contents=[Content.from_text("real")])
        empty = Message(role="assistant", contents=[])
        fake = _FakeCallNext([ChatResponse(messages=[valid, empty], finish_reason="stop")], stream=stream)
        context = _make_context(stream=stream)
        fake.bind(context)

        with pytest.raises(TerminalResponseValidationError, match="empty contents"):
            await ResponseValidationMiddleware().process(context, fake)
            await _final_response(context, stream=stream)

        assert fake.call_count == 1
