# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the response-validation middleware retry loop.

Covers the three malformed-response shapes the user explicitly called out:

1. ``contents == []`` — provider returned a message with no content items.
2. Whitespace-only text — ``text=""`` or ``text="\\n\\n\\n"``.
3. Leaked ``minimax:tool_call ... </minimax:tool_call>`` markers in text
   (with text before, after, or surrounding the marker).

Each scenario is exercised in **both** streaming and non-streaming modes,
plus the fail-fast short-circuit on identical reasons and the observation
hook contract.  The retry-exhaustion scrub lives in
``test_response_validation_scrubbing.py``, the service-storage mode in
``test_response_validation_service_storage.py``, and the pure validator
rules in ``test_response_validators.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from chrys.foundation.retry import RetryAttemptInfo
from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    ResponseStream,
)
from chrys.service.agent_middleware.response_validation import (
    MAX_RETRIES,
    ResponseValidationMiddleware,
    RetryableResponseValidationError,
    TerminalResponseValidationError,
    ValidationRetryExemption,
)
from chrys.service.agent_middleware.validators import (
    HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
    NO_VISIBLE_OUTPUT_REASON,
    REASONING_EXHAUSTED_OUTPUT_REASON,
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
    _semantic_updates,
    _service_context,
    _varied_bads,
)

# ---------------------------------------------------------------------------
# Middleware — non-streaming retries
# ---------------------------------------------------------------------------


class TestMiddlewareNonStreaming:
    async def test_valid_first_try_no_retry(self) -> None:
        good = _assistant([Content.from_text("all good")])
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[0].contents[0].text == "all good"

    async def test_search_without_final_text_retried_then_valid(self) -> None:
        bad = _search_without_final_text()
        good = _assistant([Content.from_text("recovered answer")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[0].contents[0].text == "recovered answer"
        assert [retry.reason for retry in retries] == [HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON]

    async def test_truncated_empty_response_raises_terminal_without_retry(self) -> None:
        # finish_reason="length" + empty output is terminal: give up on the first
        # attempt (no retry, no backoff) by raising into the executor error path.
        # This avoids the retry channel and avoids returning an empty assistant
        # response that the UI would render as a blank Code Agent block.
        truncated = _assistant_truncated([])
        truncated.usage_details = {"input_token_count": 11, "total_token_count": 11}
        fake = _FakeCallNext([truncated], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(
            publish_retry=on_retry,
            backoff_schedule=[0.0],
        )
        with pytest.raises(TerminalResponseValidationError, match="output token limit") as exc_info:
            await mw.process(ctx, fake)

        assert fake.call_count == 1, "terminal failure must not retry"
        assert retries == [], "terminal failure must not use the retry channel"
        assert exc_info.value.usage_details == {"input_token_count": 11, "total_token_count": 11}

    async def test_reasoning_only_exhaustion_raises_terminal(self) -> None:
        # Alternating stop/length reasoning-only responses carry distinct
        # reasons, so the short-circuit never fires and the loop runs the full
        # MAX_RETRIES retries before the terminal give-up raise.
        stop_flavor = _assistant([Content.from_text_reasoning(text="thinking")])
        length_flavor = _assistant_truncated([Content.from_text_reasoning(text="thinking harder")])
        responses = [stop_flavor if i % 2 == 0 else length_flavor for i in range(MAX_RETRIES + 1)]
        fake = _FakeCallNext(responses, stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        with pytest.raises(TerminalResponseValidationError):
            await mw.process(ctx, fake)

        assert fake.call_count == MAX_RETRIES + 1
        assert len(retries) == MAX_RETRIES
        assert retries[0].reason == NO_VISIBLE_OUTPUT_REASON
        assert retries[1].reason == REASONING_EXHAUSTED_OUTPUT_REASON

    async def test_exhaustion_drops_empty_assistant_message(self) -> None:
        """After MAX_RETRIES retries of (varied) bad responses, the bad
        assistant message is dropped on the give-up path so it does not
        land in persisted history next to the engine's turn marker.

        Uses ``_varied_bads`` so consecutive attempts have distinct
        validator reasons — fail-fast does not short-circuit and the
        loop reaches MAX_RETRIES exhaustion as intended.  ``last``
        forces the final attempt's response to be empty so the give-up
        cleanup demonstrably drops it.
        """
        responses = _varied_bads(MAX_RETRIES + 1, last=_bad_empty())
        fake = _FakeCallNext(responses, stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        # Must NOT raise — per spec, on exhaustion we return without re-raising.
        await mw.process(ctx, fake)

        # One initial + MAX_RETRIES retries = MAX_RETRIES + 1 total call_next invocations.
        assert fake.call_count == MAX_RETRIES + 1
        # Retry events fire only on the retries themselves, not on the final attempt.
        assert len(retries) == MAX_RETRIES
        # The give-up path scrubs the empty assistant message so the
        # framework's after_run never appends it to history.
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages == []

    async def test_publish_retry_receives_correct_metadata(self) -> None:
        # Distinct reasons on each bad attempt so fail-fast does not
        # short-circuit the loop before we observe two retry events.
        good = _assistant([Content.from_text("ok")])
        fake = _FakeCallNext([_bad_empty(), _bad_whitespace(), good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        recorded: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            recorded.append(info)

        mw = ResponseValidationMiddleware(
            publish_retry=on_retry,
            max_retries=3,
            backoff_schedule=[0.0, 0.0, 0.0],
        )
        await mw.process(ctx, fake)

        assert [r.attempt for r in recorded] == [1, 2]  # attempt numbers are 1-based
        assert all(r.max_attempts == 3 for r in recorded)

    async def test_publish_retry_exception_does_not_break_loop(self) -> None:
        """A raising publish_retry must not prevent retry from happening."""
        bad = _assistant([])
        good = _assistant([Content.from_text("ok")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        async def bad_publish(_info: RetryAttemptInfo) -> None:
            raise RuntimeError("telemetry broke")

        mw = ResponseValidationMiddleware(publish_retry=bad_publish, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert ctx.result.messages[0].contents[0].text == "ok"


# ---------------------------------------------------------------------------
# Middleware — streaming retries (consume, replay, retry)
# ---------------------------------------------------------------------------


class TestMiddlewareStreaming:
    async def test_valid_first_try_replay_stream_works(self) -> None:
        good = _assistant([Content.from_text("streaming good")])
        fake = _FakeCallNext([good], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        # Caller sees a ResponseStream — must be iterable and finalizable.
        assert isinstance(ctx.result, ResponseStream)
        replayed_updates = [u async for u in ctx.result]
        assert len(_semantic_updates(replayed_updates)) == 1
        final = await ctx.result.get_final_response()
        assert final.messages[0].contents[0].text == "streaming good"

    async def test_rejected_provider_raw_payload_never_leaks_through_heartbeats(self) -> None:
        """Only opaque markers cross validation before an attempt is accepted."""
        rejected_raw = {"generated": "rejected secret"}
        accepted_raw = {"event": "accepted provider metadata"}
        attempt_updates = [
            [
                ChatResponseUpdate(
                    contents=[Content.from_text("   ")],
                    role="assistant",
                    raw_representation=rejected_raw,
                )
            ],
            [
                ChatResponseUpdate(contents=[], raw_representation=accepted_raw),
                ChatResponseUpdate(contents=[Content.from_text("accepted")], role="assistant"),
            ],
        ]
        attempt = 0
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            nonlocal attempt
            updates = attempt_updates[attempt]
            attempt += 1

            async def _gen() -> AsyncIterator[ChatResponseUpdate]:
                for update in updates:
                    yield update

            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        await ResponseValidationMiddleware(backoff_schedule=[0.0]).process(ctx, _call_next)
        assert isinstance(ctx.result, ResponseStream)
        visible = [update async for update in ctx.result]

        assert attempt == 2
        assert visible[0].is_transport_heartbeat
        assert all(update.raw_representation is not rejected_raw for update in visible)
        accepted_raw_index = next(
            index for index, update in enumerate(visible) if update.raw_representation is accepted_raw
        )
        assert all(update.is_transport_heartbeat for update in visible[:accepted_raw_index])

    async def test_truncated_empty_stream_normalizes_usage_before_terminal_error(self) -> None:
        usage_start = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"input_token_count": 100, "output_token_count": 1}),
            ],
            role="assistant",
        )
        usage_final = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"output_token_count": 7}),
            ],
            role="assistant",
        )
        terminal = ChatResponseUpdate(contents=[], role="assistant", finish_reason="length")

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield usage_start
            yield usage_final
            yield terminal

        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, _call_next)
        assert isinstance(ctx.result, ResponseStream)
        stream = ctx.result
        with pytest.raises(TerminalResponseValidationError, match="output token limit") as exc_info:
            await stream.get_final_response()

        assert exc_info.value.usage_details == {"input_token_count": 100, "output_token_count": 7}

    async def test_usage_cleanup_failure_does_not_mask_terminal_stream_error(self) -> None:
        usage_update = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"input_token_count": 100, "output_token_count": 1}),
            ],
            role="assistant",
        )
        terminal = ChatResponseUpdate(contents=[], role="assistant", finish_reason="length")

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield usage_update
            yield terminal

        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        validation = ResponseValidationMiddleware(backoff_schedule=[0.0])

        async def _validation_next() -> None:
            await validation.process(ctx, _call_next)

        usage_calls: list[tuple[Any, ...]] = []

        def _on_usage(*args: Any) -> None:
            usage_calls.append(args)
            raise RuntimeError("usage callback failed")

        await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _validation_next)

        assert isinstance(ctx.result, ResponseStream)
        with pytest.raises(TerminalResponseValidationError, match="output token limit") as exc_info:
            await ctx.result.get_final_response()

        assert exc_info.value.usage_details == {"input_token_count": 100, "output_token_count": 1}
        assert usage_calls
        assert usage_calls[0][1] == 100
        assert usage_calls[0][2] == 1

    async def test_service_retry_usage_counted_before_reraise(self) -> None:
        """A service-side retryable failure bills its rejected attempt like a terminal one."""
        err = RetryableResponseValidationError(
            "stored response invalid",
            exemption=ValidationRetryExemption(attempt=1, max_attempts=3, delay_seconds=1),
            usage_details={"input_token_count": 17, "output_token_count": 4},
        )
        ctx = _make_context(stream=False)

        async def _failing_next() -> None:
            raise err

        usage_calls: list[tuple[Any, ...]] = []

        def _on_usage(*args: Any) -> None:
            usage_calls.append(args)

        with pytest.raises(RetryableResponseValidationError) as exc_info:
            await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _failing_next)

        assert exc_info.value is err
        assert usage_calls
        assert usage_calls[0][1] == 17
        assert usage_calls[0][2] == 4

    async def test_service_retry_stream_usage_counted_during_cleanup(self) -> None:
        """Streaming service-side failures bill the rejected attempt from the stream error."""
        usage_update = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"input_token_count": 17, "output_token_count": 4}),
            ],
            role="assistant",
        )
        empty = ChatResponseUpdate(contents=[], role="assistant")

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield usage_update
            yield empty

        ctx = _service_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        validation = ResponseValidationMiddleware(backoff_schedule=[0.0])

        async def _validation_next() -> None:
            await validation.process(ctx, _call_next)

        usage_calls: list[tuple[Any, ...]] = []

        def _on_usage(*args: Any) -> None:
            usage_calls.append(args)

        await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _validation_next)
        assert isinstance(ctx.result, ResponseStream)
        with pytest.raises(RetryableResponseValidationError) as exc_info:
            await ctx.result.get_final_response()

        assert exc_info.value.usage_details == {"input_token_count": 17, "output_token_count": 4}
        assert usage_calls
        assert usage_calls[0][1] == 17
        assert usage_calls[0][2] == 4

    async def test_stream_exhaustion_drops_empty_assistant_message(self) -> None:
        """Streaming exhaustion: replay stream still iterable, but both
        the finalised response AND the scrubbed updates are empty so
        consumers using ``ChatResponse.from_updates`` cannot
        rebuild the bad message.

        Without filtering updates, the streaming wrapper would replay
        the original empty-content update and then ``from_updates``
        would re-create the same ``content: []`` assistant message we
        just dropped from ``response.messages``.

        Uses ``_varied_bads`` to walk distinct validator reasons so the
        fail-fast short-circuit does not fire and the loop reaches
        MAX_RETRIES exhaustion.  ``last`` pins the final attempt to an
        empty response so the give-up cleanup demonstrably drops it.
        """
        responses = _varied_bads(MAX_RETRIES + 1, last=_bad_empty())
        fake = _FakeCallNext(responses, stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)  # must NOT raise

        assert isinstance(ctx.result, ResponseStream)
        # Empty semantic updates were filtered out — only transport heartbeats
        # remain. A consumer rebuilding with ChatResponse.from_updates
        # would re-create the bad message from any leftover empty update,
        # even if the replay stream's final response was already scrubbed.
        updates = [u async for u in ctx.result]
        assert fake.call_count == MAX_RETRIES + 1
        assert _semantic_updates(updates) == []
        # And the finalised response is empty too.
        final = await ctx.result.get_final_response()
        assert final.messages == []

    async def test_inner_result_hook_fires_only_on_caller_finalize(self) -> None:
        """The inner stream's result_hook MUST NOT fire while the middleware
        is draining + validating.  It must fire only when the caller finalises
        the replay — matching the original (no-middleware) ordering.

        Regression guard: without this, provider-installed hooks (e.g. the
        mock client's intermediate-text sync callback, which bumps
        ``IntermediateTextBuffer.batch_id``) fire too early and break
        chrys's streaming intermediate-text detection.
        """
        # Build a ChatResponse we expect to survive validation.
        good = _assistant([Content.from_text("once")])

        # Track exactly when the inner's result_hook fires.
        hook_fired_count = 0

        def _inner_hook(response: ChatResponse) -> ChatResponse:
            nonlocal hook_fired_count
            hook_fired_count += 1
            return response

        # Build an inner stream with a result_hook attached, just like
        # real providers / the MockChatClient do.
        updates = [ChatResponseUpdate(contents=good.messages[0].contents, role="assistant")]

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            for u in updates:
                yield u

        inner_stream = ResponseStream(_gen(), finalizer=lambda _u: good)
        inner_stream.with_result_hook(_inner_hook)

        # Custom call_next that returns this specific inner stream.
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = inner_stream

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, _call_next)

        # Hook must NOT have fired yet — validation uses a preview finalize
        # that bypasses hooks.
        assert hook_fired_count == 0

        # Iterate + finalise the replay.  NOW the inner's hook should fire,
        # since the replay inherits it.
        assert isinstance(ctx.result, ResponseStream)
        _ = [u async for u in ctx.result]
        await ctx.result.get_final_response()

        assert hook_fired_count == 1

    async def test_inner_result_hook_runs_before_outer_result_hooks(self) -> None:
        """Provider finalization must populate usage before UsageTracking runs."""
        good = _assistant([Content.from_text("once")])
        updates = [ChatResponseUpdate(contents=good.messages[0].contents, role="assistant")]
        hook_order: list[str] = []
        outer_usage: list[tuple[int, int]] = []

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            for update in updates:
                yield update

        def _provider_hook(response: ChatResponse) -> ChatResponse:
            hook_order.append("provider")
            response.usage_details = {"input_token_count": 11, "output_token_count": 3}
            return response

        inner_stream = ResponseStream(_gen(), finalizer=lambda _updates: good)
        inner_stream.with_result_hook(_provider_hook)
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = inner_stream

        validation = ResponseValidationMiddleware()

        async def _validation_next() -> None:
            await validation.process(ctx, _call_next)

        def _on_usage(_total: int, input_tokens: int, output_tokens: int, *_args: Any) -> None:
            hook_order.append("usage")
            outer_usage.append((input_tokens, output_tokens))

        await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _validation_next)
        assert isinstance(ctx.result, ResponseStream)
        proxy = ctx.result
        await proxy.get_final_response()

        assert hook_order == ["provider", "usage"]
        assert outer_usage == [(11, 3)]

    async def test_inner_hook_does_not_fire_for_dropped_bad_attempts(self) -> None:
        """A failed validation attempt must NOT run the inner's result_hooks.

        Running them would double-fire telemetry / intermediate-text
        capture for responses the caller never sees.
        """
        bad = _assistant([])  # empty contents — invalid
        good = _assistant([Content.from_text("recovered")])

        hook_fired = 0

        def _inner_hook(response: ChatResponse) -> ChatResponse:
            nonlocal hook_fired
            hook_fired += 1
            return response

        responses = [bad, good]
        stream_idx = 0

        async def _call_next() -> None:
            nonlocal stream_idx
            resp = responses[min(stream_idx, len(responses) - 1)]
            stream_idx += 1
            updates = [ChatResponseUpdate(contents=resp.messages[0].contents, role="assistant")]

            async def _gen() -> AsyncIterator[ChatResponseUpdate]:
                for u in updates:
                    yield u

            inner = ResponseStream(_gen(), finalizer=lambda _u, r=resp: r)
            inner.with_result_hook(_inner_hook)
            ctx.result = inner

        ctx = _make_context(stream=True)
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, _call_next)

        # Before caller finalises the replay, no inner hook has fired.
        assert hook_fired == 0

        _ = [u async for u in ctx.result]  # type: ignore[union-attr]
        await ctx.result.get_final_response()  # type: ignore[union-attr]

        # Only the SUCCESSFUL (final, good) attempt's hook fires — not
        # the dropped bad attempt's.
        assert hook_fired == 1


# ---------------------------------------------------------------------------
# Malformed shapes retried in both modes
# ---------------------------------------------------------------------------


class TestMalformedShapeRetries:
    """Retry-then-recover twins that behave the same way in streaming and
    non-streaming mode, parametrized on ``stream``."""

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_empty_contents_retried_then_valid(self, stream: bool) -> None:
        bad = _assistant([])
        good = _assistant([Content.from_text("recovered")])
        fake = _FakeCallNext([bad, good], stream=stream)
        ctx = _make_context(stream=stream)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        final = await _final_response(ctx, stream=stream)
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "recovered"
        assert len(retries) == 1
        assert "empty contents" in retries[0].reason

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_reasoning_only_response_retries_then_succeeds(self, stream: bool) -> None:
        # A reasoning-only final response is a transient over-thinking
        # failure — retry it like the other malformed shapes instead of
        # terminating the run on first sight.  In streaming mode the retry
        # happens inside the lazy validating proxy instead of failing the
        # stream.
        reasoning_only = _assistant([Content.from_text_reasoning(text="private thought")])
        good = _assistant([Content.from_text("recovered answer")])
        fake = _FakeCallNext([reasoning_only, good], stream=stream)
        ctx = _make_context(stream=stream)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        final = await _final_response(ctx, stream=stream)
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "recovered answer"
        assert [r.reason for r in retries] == [NO_VISIBLE_OUTPUT_REASON]

    @pytest.mark.parametrize(
        ("stream", "build", "match"),
        [
            pytest.param(False, _assistant, "no visible answer", id="non_stream"),
            pytest.param(True, _assistant_truncated, "output token limit while reasoning", id="stream"),
        ],
    )
    async def test_reasoning_only_repeat_failure_raises_terminal_after_retry(
        self, stream: bool, build: Callable[[list[Any]], ChatResponse], match: str
    ) -> None:
        # Two identical reasoning-only failures trip the deterministic-stuck
        # short-circuit; the give-up must RAISE (terminal_on_giveup) instead of
        # returning a blank response the caller would treat as a success.  The
        # streaming flavor raises out of the proxy's finalizer so the executor
        # records an error instead of a blank final message.
        reasoning_only = build([Content.from_text_reasoning(text="private thought")])
        fake = _FakeCallNext([reasoning_only, reasoning_only], stream=stream)
        ctx = _make_context(stream=stream)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        if stream:
            await mw.process(ctx, fake)
            assert isinstance(ctx.result, ResponseStream)
            with pytest.raises(TerminalResponseValidationError, match=match):
                await ctx.result.get_final_response()
        else:
            with pytest.raises(TerminalResponseValidationError, match=match):
                await mw.process(ctx, fake)

        assert fake.call_count == 2, "reasoning-only must retry at least once before giving up"
        assert len(retries) == 1

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_whitespace_text_retried(self, stream: bool) -> None:
        bad = _assistant([Content.from_text("\n\n\n")])
        good = _assistant([Content.from_text("OK")])
        fake = _FakeCallNext([bad, good], stream=stream)
        ctx = _make_context(stream=stream)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        final = await _final_response(ctx, stream=stream)
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "OK"

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_leaked_minimax_tool_call_retried(self, stream: bool) -> None:
        bad = _assistant([Content.from_text('Here goes:\nminimax:tool_call {"x":1} </minimax:tool_call>\nend')])
        good = _assistant([Content.from_text("clean reply")])
        fake = _FakeCallNext([bad, good], stream=stream)
        ctx = _make_context(stream=stream)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        final = await _final_response(ctx, stream=stream)
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "clean reply"


# ---------------------------------------------------------------------------
# Fail-fast — identical-reason short-circuit
# ---------------------------------------------------------------------------


class TestFailFastOnIdenticalReason:
    """Two consecutive attempts with the same ``ValidationResult.reason``
    almost always indicate a deterministic-stuck producer (KV-cache /
    chat-template / fine-tuning quirk).  Re-rolling with identical input
    will produce identical output, so the middleware short-circuits to
    the give-up path instead of paying the full MAX_RETRIES latency tax.
    """

    async def test_two_identical_empty_responses_short_circuit_non_stream(self) -> None:
        """Same reason on attempts 0 and 1 → give up at attempt 1 without
        consuming the remaining MAX_RETRIES - 1 attempts."""
        # Three available bads; we expect only 2 to be consumed before fail-fast.
        fake = _FakeCallNext([_bad_empty(), _bad_empty(), _bad_empty()], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        # Initial + one retry = 2 attempts.  Remaining retries are skipped.
        assert fake.call_count == 2
        # publish_retry fires only on the first (real) retry, not on the
        # fail-fast short-circuit.
        assert len(retries) == 1
        # Give-up cleanup still runs: empty assistant message dropped.
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages == []

    async def test_two_identical_responses_short_circuit_stream(self) -> None:
        fake = _FakeCallNext([_bad_empty(), _bad_empty(), _bad_empty()], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        # Replay must still drop empty updates so ChatResponse.from_updates
        # cannot rebuild the bad message after the fail-fast short circuit.
        updates = [u async for u in ctx.result]
        assert fake.call_count == 2
        assert _semantic_updates(updates) == []
        final = await ctx.result.get_final_response()
        assert final.messages == []

    async def test_fail_fast_strips_function_call_on_repeat(self) -> None:
        """Give-up cleanup must run even when triggered by fail-fast,
        not only on MAX_RETRIES exhaustion.  Otherwise a stuck producer
        would leak ``function_call`` contents into the tool loop after
        the early give-up and re-trigger the validation cycle every
        iteration — defeating the whole point of give-up scrubbing.
        """
        bad_with_fc = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        # Two identical bads → fail-fast at attempt 1 (well before MAX_RETRIES).
        fake = _FakeCallNext([bad_with_fc, bad_with_fc, bad_with_fc, bad_with_fc], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2  # short-circuit, not MAX_RETRIES + 1
        assert isinstance(ctx.result, ChatResponse)
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" not in types  # stripped on the early give-up
        assert "text" in types

    async def test_first_attempt_invalid_always_retries_at_least_once(self) -> None:
        """Fail-fast needs a *previous* reason to compare against, so the
        very first invalid attempt must always be retried — even if the
        producer ends up deterministic.  Without this, a single transient
        hiccup would skip recovery entirely.
        """
        good = _assistant([Content.from_text("recovered")])
        fake = _FakeCallNext([_bad_empty(), good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert len(retries) == 1  # the first invalid attempt did get a retry
        assert ctx.result.messages[0].contents[0].text == "recovered"

    async def test_alternating_reasons_do_not_trigger_fail_fast(self) -> None:
        """When consecutive attempts have *different* reasons, the producer
        is plausibly stochastic — keep retrying until either a valid
        response arrives or MAX_RETRIES is reached.  Verifies fail-fast
        does not misfire on a sequence like empty → whitespace → empty.
        """
        good = _assistant([Content.from_text("ok")])
        # empty / whitespace / empty / good — three distinct reason
        # transitions, none repeating consecutively.
        fake = _FakeCallNext(
            [_bad_empty(), _bad_whitespace(), _bad_empty(), good],
            stream=False,
        )
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        # All 3 bads consumed + 1 good = 4 calls.  Three retry events,
        # one for each invalid attempt that preceded the next call.
        assert fake.call_count == 4
        assert len(retries) == 3
        assert ctx.result.messages[0].contents[0].text == "ok"

    async def test_final_attempt_takes_priority_over_fail_fast(self) -> None:
        """When the final attempt's reason matches the previous, the
        regular exhaustion give-up path runs (not the early fail-fast
        log).  Both lead to the same cleanup, but the log message
        differs — verify cleanup correctness here regardless.
        """
        # MAX_RETRIES + 1 attempts: varied for the leading ones, then
        # the same reason on the final two.
        responses = [_bad_empty(), _bad_whitespace(), _bad_empty(), _bad_empty()]
        assert len(responses) == MAX_RETRIES + 1
        fake = _FakeCallNext(responses, stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        # All MAX_RETRIES + 1 attempts consumed because the repeat only
        # appears on the final attempt — there were no earlier
        # consecutive matches to short-circuit on.
        assert fake.call_count == MAX_RETRIES + 1
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages == []  # give-up cleanup ran


class TestValidationObservationHook:
    async def test_blocking_accepted_attempt_callbacks(self) -> None:
        good = _assistant([Content.from_text("accepted")])
        fake = _FakeCallNext([good], stream=False)
        context = _make_context(stream=False)
        fake.bind(context)
        hook = _ObservationHook()

        await ResponseValidationMiddleware(observation_hook=hook).process(context, fake)

        assert hook.events == [
            ("response", None, None),
            ("started", False),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]

    async def test_blocking_rejected_then_accepted_callbacks(self) -> None:
        bad = _assistant([])
        good = _assistant([Content.from_text("accepted")])
        fake = _FakeCallNext([bad, good], stream=False)
        context = _make_context(stream=False)
        fake.bind(context)
        hook = _ObservationHook()

        await ResponseValidationMiddleware(
            observation_hook=hook,
            backoff_schedule=[0.0],
        ).process(context, fake)

        assert hook.events == [
            ("response", None, None),
            ("started", False),
            ("contents", True, ()),
            ("rejected", "empty contents"),
            ("started", False),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]

    async def test_streaming_observes_updates_and_acceptance(self) -> None:
        good = _assistant([Content.from_text("accepted")])
        fake = _FakeCallNext([good], stream=True)
        context = _make_context(stream=True)
        fake.bind(context)
        hook = _ObservationHook()
        middleware = ResponseValidationMiddleware(observation_hook=hook)

        await middleware.process(context, fake)
        assert isinstance(context.result, ResponseStream)
        await context.result.get_final_response()

        assert hook.events == [
            ("response", None, None),
            ("started", False),
            ("contents", False, ("text",)),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]
