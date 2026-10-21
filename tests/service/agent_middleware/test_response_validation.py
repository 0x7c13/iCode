# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the response-validation middleware: terminal verdicts on an unbuffered stream.

A response that ends without an answer fails the turn with its reason, once,
without a second provider call; one that showed nothing is marked for the
kernel wire caller's one resend. A stream passes through unchanged and is
judged only when it finishes; a background response still running is not
judged at all. Accepted-response scrubbing lives in
``test_response_validation_scrubbing.py``, hosted-commit probes in
``test_response_validation_service_storage.py``, filtered and failed provider
responses in ``test_response_validation_failed_responses.py``, and the pure
validator rules in ``test_response_validators.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from chrys.foundation.errors.formatting import clean_error_message
from chrys.foundation.trajectory.context import ExchangeTrace
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from chrys.kernel.middleware import ChatContext
from chrys.service.agent_middleware.response_validation import (
    ResponseValidationMiddleware,
    TerminalResponseValidationError,
)
from chrys.service.agent_middleware.validators import (
    HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
    NO_VISIBLE_OUTPUT_REASON,
    OUTPUT_TRUNCATED_REASON,
    REASONING_EXHAUSTED_OUTPUT_REASON,
    UNNAMED_TOOL_CALL_REASON,
    DefaultResponseValidator,
    ValidationResult,
)
from chrys.service.context.middleware.usage import UsageTrackingMiddleware
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _assistant_truncated,
    _bad_empty,
    _bad_whitespace,
    _FakeCallNext,
    _final_response,
    _make_context,
    _ObservationHook,
    _search_without_final_text,
    _traced_context,
)
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.waiting import wait_for

_MODES = pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])


def _exchange(sink: FakeSink) -> ExchangeTrace:
    return ExchangeTrace(make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id()))


async def _judge(
    middleware: ResponseValidationMiddleware, context: ChatContext, call_next: Callable[[], Awaitable[None]]
) -> ChatResponse:
    """One pass through *middleware*; a streamed result is drained to its final response."""
    await middleware.process(context, call_next)
    return await _final_response(context, stream=context.stream)


async def _judge_over_usage(
    validation: ResponseValidationMiddleware,
    usage: UsageTrackingMiddleware,
    context: ChatContext,
    provider: Callable[[], Awaitable[None]],
) -> ChatResponse:
    """The production order: usage sits below validation, closest to the provider."""

    async def _usage_next() -> None:
        await usage.process(context, provider)

    return await _judge(validation, context, _usage_next)


class _UsageCalls:
    """``on_usage`` double recording each counted (input, output) pair."""

    def __init__(self) -> None:
        self.counted: list[tuple[int, int]] = []

    def __call__(self, total: int, input_tokens: int, output_tokens: int, *_calibration: Any) -> None:
        self.counted.append((input_tokens, output_tokens))


def _stream_of(
    updates: list[ChatResponseUpdate], final: ChatResponse | None = None
) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
    """A provider stream yielding *updates*, finalized to *final* or to what they assemble."""

    async def _gen() -> AsyncIterator[ChatResponseUpdate]:
        for update in updates:
            yield update

    return ResponseStream(_gen(), finalizer=ChatResponse.from_updates if final is None else lambda _updates: final)


# ---------------------------------------------------------------------------
# Terminal verdicts — each rejection fails the turn with its reason, once
# ---------------------------------------------------------------------------

_REJECTED = [
    pytest.param(_bad_empty, "empty contents", id="empty_contents"),
    pytest.param(_bad_whitespace, "empty or whitespace-only text response", id="whitespace"),
    pytest.param(
        lambda: _assistant([Content.from_text_reasoning(text="private thought")]),
        NO_VISIBLE_OUTPUT_REASON,
        id="reasoning_only",
    ),
    pytest.param(
        lambda: _assistant_truncated([Content.from_text_reasoning(text="thinking harder")]),
        REASONING_EXHAUSTED_OUTPUT_REASON,
        id="reasoning_exhausted",
    ),
    pytest.param(_search_without_final_text, HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON, id="hosted_evidence_only"),
    pytest.param(lambda: _assistant_truncated([]), OUTPUT_TRUNCATED_REASON, id="output_truncated"),
    pytest.param(
        lambda: _assistant([Content.from_function_call("call_1", "", arguments="{}")]),
        UNNAMED_TOOL_CALL_REASON,
        id="unnamed_tool_call",
    ),
    pytest.param(
        lambda: ChatResponse(messages=[], finish_reason="stop"), "no assistant message in response", id="no_message"
    ),
]
# The verdicts on a response that showed nothing: no answer text, no call.
_BLANK_REASONS = {
    "empty contents",
    "empty or whitespace-only text response",
    NO_VISIBLE_OUTPUT_REASON,
    REASONING_EXHAUSTED_OUTPUT_REASON,
    "no assistant message in response",
}


class TestTerminalVerdicts:
    @pytest.mark.parametrize(("build", "reason"), _REJECTED)
    @_MODES
    async def test_a_response_without_an_answer_fails_once_without_resampling(
        self, stream: bool, build: Callable[[], ChatResponse], reason: str
    ) -> None:
        sink = FakeSink()
        fake = _FakeCallNext([build(), _assistant([Content.from_text("must stay unused")])], stream=stream)
        context = _traced_context(stream, _exchange(sink))
        fake.bind(context)
        hook = _ObservationHook()

        with pytest.raises(TerminalResponseValidationError) as raised:
            await _judge(ResponseValidationMiddleware(observation_hook=hook), context, fake)

        assert fake.call_count == 1
        assert str(raised.value) == reason
        assert raised.value.blank_response is (reason in _BLANK_REASONS)
        # The reason is the error the turn shows: nothing is chained before it.
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert clean_error_message(raised.value) == reason
        assert [event for event in hook.events if event[0] == "rejected"] == [("rejected", reason)]
        [finished] = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
        assert finished.payload["outcome"] == "rejected"
        # Only the verdict: a resend is the retry owner's decision, recorded by its own event.
        assert "retryable" not in finished.payload
        assert "gave_up" not in finished.payload

    @_MODES
    async def test_a_valid_response_passes_with_one_call(self, stream: bool) -> None:
        sink = FakeSink()
        fake = _FakeCallNext([_assistant([Content.from_text("all good")])], stream=stream)
        context = _traced_context(stream, _exchange(sink))
        fake.bind(context)

        final = await _judge(ResponseValidationMiddleware(), context, fake)

        assert fake.call_count == 1
        assert final.text == "all good"
        [finished] = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
        assert finished.payload["outcome"] == "accepted"
        assert "gave_up" not in finished.payload

    @pytest.mark.parametrize(
        "text",
        [
            'Here goes:\nminimax:tool_call {"x":1} </minimax:tool_call>\nend',
            "<tool_use>bad</tool_use>",
            "<function_call>garbage</function_call>",
        ],
        ids=["minimax_marker", "tool_use_marker", "function_call_marker"],
    )
    @_MODES
    async def test_answer_text_is_never_judged_by_what_it_says(self, stream: bool, text: str) -> None:
        fake = _FakeCallNext([_assistant([Content.from_text(text)])], stream=stream)
        context = _make_context(stream=stream)
        fake.bind(context)

        final = await _judge(ResponseValidationMiddleware(), context, fake)

        assert fake.call_count == 1
        assert final.text == text

    @_MODES
    async def test_a_background_response_still_running_is_accepted_without_a_verdict(self, stream: bool) -> None:
        sink = FakeSink()
        running = ChatResponse(
            messages=[Message(role="assistant", contents=[])], continuation_token={"response_id": "resp_1"}
        )
        fake = _FakeCallNext([running], stream=stream)
        context = _traced_context(stream, _exchange(sink))
        fake.bind(context)
        hook = _ObservationHook()

        final = await _judge(ResponseValidationMiddleware(observation_hook=hook), context, fake)

        assert fake.call_count == 1
        assert final.continuation_token == {"response_id": "resp_1"}
        assert sink.of_type(EventType.MODEL_VALIDATION_FINISHED) == []
        assert hook.events[-1] == ("accepted", ((),))
        assert not any(event[0] == "rejected" for event in hook.events)

    @_MODES
    async def test_the_poll_that_finishes_a_background_response_is_judged(self, stream: bool) -> None:
        fake = _FakeCallNext([_bad_empty()], stream=stream)
        context = ChatContext(
            client=None,  # type: ignore[arg-type]
            messages=[Message("user", ["hi"])],
            options={"continuation_token": {"response_id": "resp_1"}},
            stream=stream,
        )
        fake.bind(context)
        hook = _ObservationHook()

        with pytest.raises(TerminalResponseValidationError, match="empty contents"):
            await _judge(ResponseValidationMiddleware(observation_hook=hook), context, fake)

        # A poll resumes the response an earlier request began: no new response starts.
        assert hook.events[0] == ("started", True)
        assert not any(event[0] == "response" for event in hook.events)


# ---------------------------------------------------------------------------
# Streaming — updates pass through as they arrive; the finished stream is judged
# ---------------------------------------------------------------------------


class TestStreamingPassthrough:
    async def test_each_update_reaches_the_consumer_before_the_stream_ends(self) -> None:
        first = ChatResponseUpdate(contents=[Content.from_text("first ")], role="assistant")
        last = ChatResponseUpdate(contents=[Content.from_text("second")], role="assistant")
        release = asyncio.Event()

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield first
            await release.wait()
            yield last

        provider_stream = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)
        context = _make_context(stream=True)

        async def _call_next() -> None:
            context.result = provider_stream

        await ResponseValidationMiddleware().process(context, _call_next)
        assert context.result is provider_stream
        seen: list[ChatResponseUpdate] = []

        async def _consume() -> ChatResponse:
            async for update in provider_stream:
                seen.append(update)
            return await provider_stream.get_final_response()

        consumer = asyncio.create_task(_consume())
        try:
            await wait_for(lambda: seen or consumer.done(), description="first streamed update")
            if consumer.done():
                await consumer  # surfaces an early failure
            assert len(seen) == 1
            assert seen[0] is first
            assert not consumer.done()
        finally:
            release.set()
            final = await consumer

        assert len(seen) == 2
        assert seen[1] is last
        assert final.text == "first second"

    async def test_a_rejected_stream_shows_what_it_streamed_then_fails_at_its_end(self) -> None:
        raw = {"generated": "   "}
        blank = ChatResponseUpdate(contents=[Content.from_text("   ")], role="assistant", raw_representation=raw)
        context = _make_context(stream=True)
        calls = 0

        async def _call_next() -> None:
            nonlocal calls
            calls += 1
            context.result = _stream_of([blank])

        await ResponseValidationMiddleware().process(context, _call_next)
        assert isinstance(context.result, ResponseStream)
        seen: list[ChatResponseUpdate] = []
        with pytest.raises(TerminalResponseValidationError, match="whitespace-only") as raised:
            async for update in context.result:
                seen.append(update)

        assert calls == 1
        assert len(seen) == 1
        assert seen[0] is blank
        assert seen[0].raw_representation is raw
        assert raised.value.__context__ is None

    async def test_the_provider_result_hook_fires_once_when_the_caller_finalizes(self) -> None:
        good = _assistant([Content.from_text("once")])
        fired = 0

        def _provider_hook(response: ChatResponse) -> ChatResponse:
            nonlocal fired
            fired += 1
            return response

        provider_stream = _stream_of([ChatResponseUpdate(contents=good.messages[0].contents, role="assistant")])
        provider_stream.with_result_hook(_provider_hook)
        context = _make_context(stream=True)

        async def _call_next() -> None:
            context.result = provider_stream

        await ResponseValidationMiddleware().process(context, _call_next)
        assert fired == 0

        assert isinstance(context.result, ResponseStream)
        _ = [update async for update in context.result]
        await context.result.get_final_response()

        assert fired == 1

    async def test_provider_and_usage_finalize_before_validation_judges(self) -> None:
        order: list[str] = []

        class _OrderRecordingValidator:
            def validate(self, response: ChatResponse) -> ValidationResult:
                order.append("validation")
                return DefaultResponseValidator().validate(response)

        def _provider_hook(response: ChatResponse) -> ChatResponse:
            order.append("provider")
            response.usage_details = {"input_token_count": 11, "output_token_count": 3}
            return response

        good = _assistant([Content.from_text("once")])
        provider_stream = _stream_of([ChatResponseUpdate(contents=good.messages[0].contents, role="assistant")], good)
        provider_stream.with_result_hook(_provider_hook)
        context = _make_context(stream=True)

        async def _provider() -> None:
            context.result = provider_stream

        usage_calls = _UsageCalls()

        def _on_usage(total: int, input_tokens: int, output_tokens: int, *calibration: Any) -> None:
            order.append("usage")
            usage_calls(total, input_tokens, output_tokens, *calibration)

        await _judge_over_usage(
            ResponseValidationMiddleware(validator=_OrderRecordingValidator()),
            UsageTrackingMiddleware(on_usage=_on_usage),
            context,
            _provider,
        )

        assert order == ["provider", "usage", "validation"]
        assert usage_calls.counted == [(11, 3)]


# ---------------------------------------------------------------------------
# Usage — the middleware below validation counts every landed response once
# ---------------------------------------------------------------------------


class TestRejectedResponseUsage:
    @_MODES
    async def test_a_rejected_response_is_counted_once(self, stream: bool) -> None:
        truncated = _assistant_truncated([])
        truncated.usage_details = {"input_token_count": 11, "output_token_count": 0, "total_token_count": 11}
        fake = _FakeCallNext([truncated], stream=stream)
        context = _make_context(stream=stream)
        fake.bind(context)
        usage_calls = _UsageCalls()

        with pytest.raises(TerminalResponseValidationError, match="output token limit"):
            await _judge_over_usage(
                ResponseValidationMiddleware(), UsageTrackingMiddleware(on_usage=usage_calls), context, fake
            )

        assert fake.call_count == 1
        assert usage_calls.counted == [(11, 0)]

    async def test_streamed_usage_snapshots_collapse_into_one_count_before_the_rejection(self) -> None:
        updates = [
            ChatResponseUpdate(
                contents=[Content.from_usage(usage_details={"input_token_count": 100, "output_token_count": 1})],
                role="assistant",
            ),
            ChatResponseUpdate(
                contents=[Content.from_usage(usage_details={"output_token_count": 7})], role="assistant"
            ),
            ChatResponseUpdate(contents=[], role="assistant", finish_reason="length"),
        ]
        context = _make_context(stream=True)

        async def _provider() -> None:
            context.result = _stream_of(updates)

        usage_calls = _UsageCalls()
        with pytest.raises(TerminalResponseValidationError, match="output token limit"):
            await _judge_over_usage(
                ResponseValidationMiddleware(), UsageTrackingMiddleware(on_usage=usage_calls), context, _provider
            )

        assert usage_calls.counted == [(100, 7)]


# ---------------------------------------------------------------------------
# Output truncation — the next request compacts first when the input left no room
# ---------------------------------------------------------------------------


class _OverflowNotes:
    """``ContextOverflowSink`` double counting the notes it is handed."""

    def __init__(self) -> None:
        self.notes = 0

    def note_context_overflow(self, exc: BaseException | None = None) -> bool:
        self.notes += 1
        return True


class TestOutputTruncatedNote:
    @pytest.mark.parametrize(
        ("build", "usage", "notes"),
        [
            pytest.param(lambda: _assistant_truncated([]), None, 1, id="nothing_generated"),
            pytest.param(
                lambda: _assistant_truncated([]),
                {"output_token_count": 4096, "reasoning_output_token_count": 4096},
                0,
                id="hidden_reasoning_counted",
            ),
            pytest.param(
                lambda: _assistant_truncated([Content.from_text_reasoning(text="thinking")]),
                None,
                0,
                id="reasoning_shown",
            ),
            pytest.param(_bad_empty, None, 0, id="stopped_empty"),
        ],
    )
    async def test_only_a_length_stop_with_nothing_generated_notes_an_overflow(
        self, build: Callable[[], ChatResponse], usage: dict[str, int] | None, notes: int
    ) -> None:
        response = build()
        response.usage_details = usage
        strategy = _OverflowNotes()
        fake = _FakeCallNext([response], stream=False)
        context = ChatContext(
            client=None,  # type: ignore[arg-type]
            messages=[Message("user", ["hi"])],
            options=None,
            stream=False,
            kwargs={"compaction_strategy": strategy},
        )
        fake.bind(context)

        with pytest.raises(TerminalResponseValidationError):
            await ResponseValidationMiddleware().process(context, fake)

        assert strategy.notes == notes


# ---------------------------------------------------------------------------
# Observation hook — raw contents and the attempt's lifecycle, in order
# ---------------------------------------------------------------------------


class TestValidationObservationHook:
    async def test_blocking_accepted_attempt_callbacks(self) -> None:
        fake = _FakeCallNext([_assistant([Content.from_text("accepted")])], stream=False)
        context = _make_context(stream=False)
        fake.bind(context)
        hook = _ObservationHook()

        await ResponseValidationMiddleware(observation_hook=hook).process(context, fake)

        assert hook.events == [
            ("response", None),
            ("started", False),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]

    async def test_blocking_rejected_attempt_callbacks(self) -> None:
        fake = _FakeCallNext([_bad_empty(), _assistant([Content.from_text("must stay unused")])], stream=False)
        context = _make_context(stream=False)
        fake.bind(context)
        hook = _ObservationHook()

        with pytest.raises(TerminalResponseValidationError):
            await ResponseValidationMiddleware(observation_hook=hook).process(context, fake)

        assert hook.events == [
            ("response", None),
            ("started", False),
            ("contents", True, ()),
            ("rejected", "empty contents"),
        ]

    async def test_streaming_observes_updates_and_acceptance(self) -> None:
        fake = _FakeCallNext([_assistant([Content.from_text("accepted")])], stream=True)
        context = _make_context(stream=True)
        fake.bind(context)
        hook = _ObservationHook()

        await _judge(ResponseValidationMiddleware(observation_hook=hook), context, fake)

        assert hook.events == [
            ("response", None),
            ("started", False),
            ("contents", False, ("text",)),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]
