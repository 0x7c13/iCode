# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the provider-agnostic instrumented LLM client helpers and ``_IntermediateTextMixin``."""

from __future__ import annotations

import asyncio
from copy import copy
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from chrys.foundation.trajectory.context import (
    TRAJECTORY_EXCHANGE_KWARG,
    ExchangeTrace,
    side_call_scope,
    trajectory_scope,
)
from chrys.foundation.trajectory.envelope import ActorRole
from chrys.foundation.trajectory.event_types import EventType, ExchangeOutcome
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory_timing import TRAJECTORY_TIMING_KEY, build_trajectory_timing
from chrys.foundation.util.chrys_headers import PARENT_SESSION_ID_HEADER, SESSION_ID_HEADER
from chrys.kernel import (
    AgentResponse,
    AgentSession,
    BaseChatClient,
    ChatClientException,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
    SessionContext,
    internal_side_call_scope,
)
from chrys.kernel.exceptions import ChatClientContentFilterException, ChatClientInvalidRequestException
from chrys.service.context.providers.history import CompressibleHistoryProvider
from chrys.service.llm.instrumented import (
    _compose_client_stack,
    _count_function_calls,
    _ensure_openai_response_has_choices,
    _extract_intermediate_text,
    _IntermediateTextMixin,
    create_instrumented_anthropic_client,
    create_instrumented_openai_client,
    create_instrumented_openai_responses_client,
)
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY, stamp_message_response_timing
from tests.service.trajectory._fakes import FakeSink, make_context


def _make_response(*messages: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(messages=list(messages))


def _make_msg(*contents: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(contents=list(contents))


def _text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t, provider_hosted=False)


def _fn_call(name: str = "tool", *, informational_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        informational_only=informational_only,
        provider_hosted=False,
    )


def _fn_result() -> SimpleNamespace:
    return SimpleNamespace(type="function_result", provider_hosted=False)


# ──────────────── _extract_intermediate_text ────────────────────────────


def test_extract_text_with_function_call() -> None:
    """Text alongside function_call should be extracted."""
    resp = _make_response(_make_msg(_text("Let me check"), _fn_call()))
    assert _extract_intermediate_text(resp) == "Let me check"


def test_extract_multiple_text_parts() -> None:
    """Multiple text parts should be concatenated."""
    resp = _make_response(_make_msg(_text("A"), _text("B"), _fn_call()))
    assert _extract_intermediate_text(resp) == "AB"


def test_extract_no_function_call_returns_none() -> None:
    """Text-only response (no function_call) should return None."""
    resp = _make_response(_make_msg(_text("Hello")))
    assert _extract_intermediate_text(resp) is None


def test_extract_function_call_only_returns_none() -> None:
    """Function call without text should return None."""
    resp = _make_response(_make_msg(_fn_call()))
    assert _extract_intermediate_text(resp) is None


def test_extract_empty_text_ignored() -> None:
    """Empty text parts should not count as intermediate text."""
    resp = _make_response(_make_msg(_text(""), _fn_call()))
    assert _extract_intermediate_text(resp) is None


def test_extract_across_multiple_messages() -> None:
    """Text in one message, function_call in another."""
    resp = _make_response(
        _make_msg(_text("thinking")),
        _make_msg(_fn_call()),
    )
    assert _extract_intermediate_text(resp) == "thinking"


def test_extract_ignores_informational_function_call() -> None:
    resp = _make_response(_make_msg(_text("visible"), _fn_call(informational_only=True)))

    assert _extract_intermediate_text(resp) is None


def test_extract_defers_hosted_response_text_to_presentation_bridge() -> None:
    hosted = SimpleNamespace(type="search_tool_call", provider_hosted=True)
    resp = _make_response(_make_msg(_text("checking"), hosted, _text("answer")))

    assert _extract_intermediate_text(resp) is None


# ──────────────── _count_function_calls ─────────────────────────────────


def test_count_function_calls_single() -> None:
    resp = _make_response(_make_msg(_fn_call()))
    assert _count_function_calls(resp) == 1


def test_count_function_calls_multiple() -> None:
    resp = _make_response(_make_msg(_fn_call("a"), _fn_call("b"), _fn_call("c")))
    assert _count_function_calls(resp) == 3


def test_count_function_calls_ignores_informational_calls() -> None:
    resp = _make_response(_make_msg(_fn_call("hosted", informational_only=True)))

    assert _count_function_calls(resp) == 0


def test_count_function_calls_text_only() -> None:
    resp = _make_response(_make_msg(_text("hello")))
    assert _count_function_calls(resp) == 0


def test_count_function_calls_empty() -> None:
    resp = _make_response(_make_msg())
    assert _count_function_calls(resp) == 0


def test_count_function_calls_ignores_function_result() -> None:
    """function_result is NOT a function_call."""
    resp = _make_response(_make_msg(_fn_result()))
    assert _count_function_calls(resp) == 0


# ──────────────── _ensure_openai_response_has_choices ───────────────────


def test_ensure_choices_passes_with_populated_choices() -> None:
    resp = SimpleNamespace(choices=[SimpleNamespace()])
    _ensure_openai_response_has_choices(resp)


def test_ensure_choices_passes_with_empty_list() -> None:
    """Empty choices is a valid (no-completion) response and must not raise."""
    resp = SimpleNamespace(choices=[])
    _ensure_openai_response_has_choices(resp)


def test_ensure_choices_raises_when_none_and_includes_body() -> None:
    body = '{"error": "rate limit exceeded"}'
    resp = SimpleNamespace(choices=None, model_dump_json=lambda: body)
    with pytest.raises(ChatClientException) as exc_info:
        _ensure_openai_response_has_choices(resp)
    msg = str(exc_info.value)
    assert "missing the required 'choices' array" in msg
    assert "rate limit exceeded" in msg


def test_ensure_choices_raises_when_not_a_list() -> None:
    """Non-strict SDK construction can leave a str/dict in ``choices``; reject it too."""
    body = '{"choices": "oops"}'
    resp = SimpleNamespace(choices="oops", model_dump_json=lambda: body)
    with pytest.raises(ChatClientException) as exc_info:
        _ensure_openai_response_has_choices(resp)
    msg = str(exc_info.value)
    assert "'choices' is str; expected an array" in msg
    assert body in msg


def test_ensure_choices_truncates_long_body() -> None:
    big = '{"error": "' + ("x" * 5000) + '"}'
    resp = SimpleNamespace(choices=None, model_dump_json=lambda: big)
    with pytest.raises(ChatClientException) as exc_info:
        _ensure_openai_response_has_choices(resp)
    assert "[truncated]" in str(exc_info.value)


def test_ensure_choices_falls_back_to_repr_on_dump_failure() -> None:
    def _bad_dump() -> str:
        raise RuntimeError("pydantic broken")

    resp = SimpleNamespace(choices=None, model_dump_json=_bad_dump)
    with pytest.raises(ChatClientException):
        _ensure_openai_response_has_choices(resp)


async def test_instrumented_responses_factory_preserves_deepseek_subclass_callbacks_and_headers() -> None:
    from chrys.service.llm.deepseek import DeepSeekResponsesClient

    async_calls: list[str] = []
    sync_calls: list[str] = []

    async def _async_callback(text: str) -> None:
        async_calls.append(text)

    def _sync_callback(text: str) -> None:
        sync_calls.append(text)

    from openai import AsyncOpenAI

    from chrys.service.llm.instrumented import create_instrumented_openai_responses_client

    client = create_instrumented_openai_responses_client(
        model_id="deepseek-test",
        session_id="session-1",
        parent_session_id="parent-1",
        client=AsyncOpenAI(api_key="sk-fake"),
        chat_client_cls=DeepSeekResponsesClient,
        on_intermediate_text_async=_async_callback,
        on_intermediate_text_sync=_sync_callback,
    )
    raw = client.inner.inner

    prepared = await raw._prepare_options([Message("user", ["hi"])], {})

    assert DeepSeekResponsesClient in type(raw).__mro__
    assert raw._on_intermediate_text_async is _async_callback
    assert raw._on_intermediate_text_sync is _sync_callback
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "session-1"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "parent-1"
    assert async_calls == []
    assert sync_calls == []


def test_raw_clients_require_preconfigured_sdk_clients() -> None:
    from chrys.service.llm.anthropic_chat import RawAnthropicClient
    from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient
    from chrys.service.llm.openai_responses import RawOpenAIChatClient

    with pytest.raises(ValueError, match="pre-configured async_client"):
        RawOpenAIChatCompletionClient(model="gpt-test")
    with pytest.raises(ValueError, match="pre-configured async_client"):
        RawOpenAIChatClient(model="gpt-test")
    with pytest.raises(ValueError, match="pre-configured anthropic_client"):
        RawAnthropicClient(model="claude-test")


def test_instrumented_factories_require_preconfigured_sdk_clients() -> None:
    with pytest.raises(TypeError, match="client"):
        create_instrumented_openai_client(model_id="gpt-test")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="client"):
        create_instrumented_openai_responses_client(model_id="gpt-test")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="anthropic_client"):
        create_instrumented_anthropic_client(model_id="claude-test")  # type: ignore[call-arg]


class _FailingWireClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def _inner_get_response(
        self,
        *,
        messages: Any,
        options: Any,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        del messages, options, kwargs
        if stream:

            async def _updates() -> Any:
                raise self.exc
                yield ChatResponseUpdate(contents=[Content.from_text("unused")])

            return ResponseStream(
                _updates(),
                finalizer=ChatResponse.from_updates,
            )

        async def _response() -> Any:
            raise self.exc

        return _response()


class _InstrumentedFailingWireClient(_IntermediateTextMixin, _FailingWireClient):
    pass


def _make_instrumented_wire_client(
    *,
    response: Any = None,
    stream_text: str = "streamed",
    delay: float = 0,
) -> Any:
    class _RawWireClient(BaseChatClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        def _inner_get_response(
            self,
            *,
            messages: Any,
            options: Any,
            stream: bool = False,
            **kwargs: Any,
        ) -> Any:
            self.calls.append(
                {"messages": list(messages), "options": dict(options), "stream": stream, "kwargs": kwargs}
            )
            if stream:

                async def _updates() -> Any:
                    if delay > 0:
                        await asyncio.sleep(delay)
                    yield ChatResponseUpdate(
                        contents=[Content.from_text(stream_text)],
                        role="assistant",
                    )

                return ResponseStream(
                    _updates(),
                    finalizer=lambda updates: ChatResponse.from_updates(
                        updates,
                        output_format_type=options.get("response_format"),
                    ),
                )

            async def _response() -> Any:
                if delay > 0:
                    await asyncio.sleep(delay)
                if response is not None:
                    return response
                return ChatResponse(
                    messages=[
                        Message(
                            "assistant",
                            [Content.from_text(stream_text)],
                        )
                    ],
                    response_format=options.get("response_format"),
                )

            return _response()

    class _InstrumentedWireClient(_IntermediateTextMixin, _RawWireClient):
        pass

    return _InstrumentedWireClient()


class _NoopCompaction:
    async def __call__(self, messages: list[Any], context: Any = None) -> bool:
        self.messages = messages
        self.context = context
        return False


async def test_chrys_chat_client_exception_propagates_non_streaming() -> None:
    inner = ValueError("root cause")
    chrys_exc = ChatClientInvalidRequestException(
        "provider rejected request",
        inner_exception=inner,
        log_level=None,
    )
    client = _InstrumentedFailingWireClient(chrys_exc)

    with pytest.raises(ChatClientInvalidRequestException) as exc_info:
        await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    assert exc_info.value is chrys_exc
    assert exc_info.value.args == ("provider rejected request", inner)


async def test_chrys_chat_client_exception_propagates_streaming() -> None:
    inner = RuntimeError("filter details")
    chrys_exc = ChatClientContentFilterException(
        "provider content filter",
        inner_exception=inner,
        log_level=None,
    )
    client = _InstrumentedFailingWireClient(chrys_exc)

    stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)

    assert isinstance(stream, ResponseStream)
    with pytest.raises(ChatClientContentFilterException) as exc_info:
        async for _update in stream:
            pass
    assert exc_info.value is chrys_exc
    assert exc_info.value.args == ("provider content filter", inner)


async def test_streaming_get_response_with_compaction_returns_chrys_stream() -> None:
    client = _make_instrumented_wire_client(stream_text="compacted stream")
    compaction = _NoopCompaction()

    stream = client.get_response(
        [Message("user", ["hi"])],
        stream=True,
        options={},
        compaction_strategy=compaction,
    )

    assert isinstance(stream, ResponseStream)
    updates = [update async for update in stream]
    assert [update.text for update in updates] == ["compacted stream"]
    final = await stream.get_final_response()
    assert final.text == "compacted stream"
    assert client.calls[0]["stream"] is True
    assert compaction.messages


async def test_native_response_preserves_lazy_value_parse() -> None:
    class StructuredPayload(BaseModel):
        answer: int

    native_response = ChatResponse(
        messages=[Message("assistant", [Content.from_text("not-json")])],
        response_format=StructuredPayload,
    )
    client = _make_instrumented_wire_client(response=native_response)

    response = await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    assert response._value_parsed is False
    with pytest.raises(ValidationError):
        _ = response.value


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_instrumented_response_persists_measured_trajectory_timing(stream: bool) -> None:
    client = _make_instrumented_wire_client(stream_text="timed")

    response_or_stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={},
        stream=stream,
    )
    if stream:
        async for _update in response_or_stream:
            pass
        response = await response_or_stream.get_final_response()
    else:
        response = await response_or_stream

    message = response.messages[-1]
    timing = message.additional_properties[TRAJECTORY_TIMING_KEY]
    assert timing["started_at"] <= timing["finished_at"]
    assert timing["finished_at"] == message.additional_properties[MESSAGE_CREATED_AT_KEY]
    assert isinstance(timing["duration_ms"], int)
    assert timing["duration_ms"] >= 0


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_measured_timing_survives_tool_loop_and_history_persistence(stream: bool) -> None:
    """The production loop and history provider preserve the wire-stamped message."""
    wire_client = _make_instrumented_wire_client(stream_text="persisted", delay=0.01)
    client = _compose_client_stack(
        wire_client,
        max_iterations=None,
        max_consecutive_errors=None,
    )
    user = Message("user", [Content.from_text("hi")])
    provider = CompressibleHistoryProvider()
    session = AgentSession(session_id="timing-survival")
    context = SessionContext(session_id="timing-survival", input_messages=[user])
    state: dict[str, Any] = {"messages": [], "compressed_msgs": []}

    await provider.before_run(agent=object(), session=session, context=context, state=state)
    wire_messages = context.get_messages(include_input=True)
    if stream:
        response_stream = client.get_response(wire_messages, stream=True, options={})
        assert isinstance(response_stream, ResponseStream)
        _ = [update async for update in response_stream]
        response = await response_stream.get_final_response()
    else:
        pending_response = client.get_response(wire_messages, stream=False, options={})
        assert not isinstance(pending_response, ResponseStream)
        response = await pending_response
    context._response = AgentResponse(messages=response.messages)
    await provider.after_run(agent=object(), session=session, context=context, state=state)

    persisted = state["messages"][-1]
    assert persisted is response.messages[-1]
    timing = persisted.additional_properties[TRAJECTORY_TIMING_KEY]
    assert timing["finished_at"] == persisted.additional_properties[MESSAGE_CREATED_AT_KEY]
    assert timing["duration_ms"] >= 1


async def test_instrumented_response_stamps_provider_hosted_tool_contents() -> None:
    hosted_call = Content.from_search_tool_call(
        "search-1",
        tool_name="web_search",
        arguments={"query": "timing"},
    )
    native_response = ChatResponse(messages=[Message("assistant", [hosted_call])])
    client = _make_instrumented_wire_client(response=native_response)

    response = await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    message_timing = response.messages[-1].additional_properties[TRAJECTORY_TIMING_KEY]
    assert hosted_call.additional_properties[TRAJECTORY_TIMING_KEY] == message_timing


async def test_instrumented_response_timing_does_not_mutate_echoed_request_objects() -> None:
    old_started_at = "2000-01-01T01:02:03+00:00"
    old_finished_at = "2000-01-01T01:02:04+00:00"
    hosted_call = Content.from_search_tool_call(
        "search-old",
        tool_name="web_search",
        arguments={"query": "old"},
    )
    old_timing = build_trajectory_timing(
        started_at=old_started_at,
        finished_at=old_finished_at,
        duration_ms=1_000,
    )
    hosted_call.additional_properties[TRAJECTORY_TIMING_KEY] = dict(old_timing)
    echoed = Message("assistant", [hosted_call])
    stamp_message_response_timing(
        echoed,
        started_at=old_started_at,
        finished_at=old_finished_at,
        duration_ms=1_000,
    )
    shallow_hosted_echo = copy(hosted_call)
    assert shallow_hosted_echo.additional_properties is hosted_call.additional_properties
    shallow_echo = Message("assistant", [shallow_hosted_echo])
    fresh = Message("assistant", [Content.from_text("fresh")])
    client = _make_instrumented_wire_client(response=ChatResponse(messages=[echoed, shallow_echo, fresh]))

    response = await client._inner_get_response(messages=[echoed], options={})

    assert echoed.additional_properties[TRAJECTORY_TIMING_KEY] == old_timing
    assert echoed.additional_properties[MESSAGE_CREATED_AT_KEY] == old_timing["finished_at"]
    assert hosted_call.additional_properties[TRAJECTORY_TIMING_KEY] == old_timing
    assert shallow_hosted_echo.additional_properties[TRAJECTORY_TIMING_KEY] == old_timing
    assert response.messages[-1].additional_properties[TRAJECTORY_TIMING_KEY] != old_timing


async def test_native_stream_final_response_preserves_response_format() -> None:
    class StructuredPayload(BaseModel):
        answer: str

    client = _make_instrumented_wire_client(stream_text='{"answer":"ok"}')

    stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={"response_format": StructuredPayload},
        stream=True,
    )
    updates = [update async for update in stream]
    final = await stream.get_final_response()

    assert [update.text for update in updates] == ['{"answer":"ok"}']
    assert final.value == StructuredPayload(answer="ok")


# ──────────────── internal side-call suppression ─────────────────────────
#
# LAST_WORDS side calls go through ``_inner_get_response`` inside
# ``internal_side_call_scope()``.  If the model ignores the no-tools
# instruction and returns text alongside a function_call, the mixin must NOT
# publish that text (or a batch-boundary signal) — the throwaway side-call
# response never joins the conversation.


def _make_tool_call_wire_client() -> Any:
    """Instrumented wire client whose responses carry text + function_call."""

    def _contents() -> list[Content]:
        return [Content.from_text("Let me check"), Content.from_function_call("call-1", "tool")]

    class _RawToolCallWireClient(BaseChatClient):
        def _inner_get_response(
            self,
            *,
            messages: Any,
            options: Any,
            stream: bool = False,
            **kwargs: Any,
        ) -> Any:
            del messages, options, kwargs
            if stream:

                async def _updates() -> Any:
                    yield ChatResponseUpdate(contents=_contents(), role="assistant")

                return ResponseStream(_updates(), finalizer=ChatResponse.from_updates)

            async def _response() -> Any:
                return ChatResponse(messages=[Message("assistant", _contents())])

            return _response()

    class _InstrumentedToolCallWireClient(_IntermediateTextMixin, _RawToolCallWireClient):
        pass

    return _InstrumentedToolCallWireClient()


async def test_intermediate_text_suppressed_in_internal_side_call_non_streaming() -> None:
    client = _make_tool_call_wire_client()
    fired: list[str] = []

    async def _cb(text: str) -> None:
        fired.append(text)

    client._on_intermediate_text_async = _cb

    with internal_side_call_scope():
        response = await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    assert response.text == "Let me check"
    assert fired == []

    # Control: the same response outside the scope does fire the callback.
    await client._inner_get_response(messages=[Message("user", ["hi"])], options={})
    assert fired == ["Let me check"]


async def test_intermediate_text_suppressed_in_internal_side_call_streaming() -> None:
    client = _make_tool_call_wire_client()
    fired: list[str] = []
    client._on_intermediate_text_sync = fired.append

    with internal_side_call_scope():
        stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
        final = await stream.get_final_response()

    assert final.text == "Let me check"
    assert fired == []

    # Control: outside the scope the result hook publishes on finalization.
    stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
    await stream.get_final_response()
    assert fired == ["Let me check"]


# ──────────────── side-call exchange closure ─────────────────────────────
#
# A side call below the kernel opens its own exchange trace; nothing above it
# holds the handle, so a stream that never reaches a final response has to
# report its own end or the acquisition reads as one still in flight.


def _make_stream_wire_client(fail_with: type[BaseException] | None) -> Any:
    """Instrumented wire client whose stream ends in *fail_with* (or normally)."""

    class _RawStreamWireClient(BaseChatClient):
        def _inner_get_response(self, *, messages: Any, options: Any, stream: bool = False, **kwargs: Any) -> Any:
            del messages, options, stream, kwargs

            async def _updates() -> Any:
                yield ChatResponseUpdate(contents=[Content.from_text("partial")], role="assistant")
                if fail_with is not None:
                    raise fail_with()

            return ResponseStream(_updates(), finalizer=ChatResponse.from_updates)

    class _InstrumentedStreamWireClient(_IntermediateTextMixin, _RawStreamWireClient):
        pass

    return _InstrumentedStreamWireClient()


async def _drain_side_call_stream(client: Any, sink: FakeSink) -> None:
    with trajectory_scope(make_context(sink)), internal_side_call_scope(), side_call_scope(ActorRole.COMPLETER):
        stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
        await stream.get_final_response()


async def test_a_side_call_stream_that_errors_closes_its_own_exchange() -> None:
    sink = FakeSink()
    with pytest.raises(RuntimeError):
        await _drain_side_call_stream(_make_stream_wire_client(RuntimeError), sink)

    finished = sink.only(EventType.MODEL_EXCHANGE_FINISHED)
    assert finished.payload["outcome"] == ExchangeOutcome.ERROR
    assert finished.payload["error_code"] == "RuntimeError"
    assert finished.operation_id == sink.only(EventType.MODEL_EXCHANGE_STARTED).operation_id


async def test_a_side_call_stream_dropped_mid_flight_closes_its_own_exchange() -> None:
    sink = FakeSink()
    with pytest.raises(asyncio.CancelledError):
        await _drain_side_call_stream(_make_stream_wire_client(asyncio.CancelledError), sink)

    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["outcome"] == ExchangeOutcome.ABANDONED


async def test_a_side_call_stream_that_finishes_reports_success_once() -> None:
    sink = FakeSink()
    await _drain_side_call_stream(_make_stream_wire_client(None), sink)

    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["outcome"] == ExchangeOutcome.SUCCESS


async def test_a_forwarded_exchange_is_left_to_the_loop_that_owns_it() -> None:
    """The loop closes its own exchanges with the outcome it knows (stalled,
    interrupted), so a failing stream must not close them first."""
    sink = FakeSink()
    context = make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id())
    client = _make_stream_wire_client(RuntimeError)
    with trajectory_scope(context), pytest.raises(RuntimeError):
        stream = client._inner_get_response(
            messages=[Message("user", ["hi"])],
            options={},
            stream=True,
            **{TRAJECTORY_EXCHANGE_KWARG: ExchangeTrace(context)},
        )
        await stream.get_final_response()

    assert sink.of_type(EventType.MODEL_EXCHANGE_STARTED)
    assert not sink.of_type(EventType.MODEL_EXCHANGE_FINISHED)


async def test_a_per_request_model_override_cannot_grow_past_the_line_budget() -> None:
    """A profile's chat options are unrestricted, and the per-request model
    override is the one request fact only the start marker carries: one long
    enough to make that line unwritable would leave the terminal closing a
    start that became a gap."""
    sink = FakeSink()
    client = _make_stream_wire_client(None)
    with trajectory_scope(make_context(sink)), internal_side_call_scope(), side_call_scope(ActorRole.COMPLETER):
        stream = client._inner_get_response(
            messages=[Message("user", ["hi"])], options={"model": "m" * 200_000}, stream=True
        )
        await stream.get_final_response()

    # The sink applies the writer's own checks, so an unbounded override fails
    # here as the over-budget line it would have been.
    assert sink.only(EventType.MODEL_EXCHANGE_STARTED).payload["request_model"] == "m" * 256
    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["outcome"] == ExchangeOutcome.SUCCESS
