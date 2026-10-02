# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""How a Responses reply ends: failures, streams cut short, running responses, and refusals that ask for calls."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from chrys.foundation.errors import ErrorKind, ProviderResponseError, classify_error, invalidates_continuation_token
from chrys.kernel import Message, ResponseStream
from chrys.kernel.loop import StallExhaustedAction
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.orchestration.invoker.attempts import WireRetryPolicyAdapter
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from tests.service.llm._responses_wire import (
    RESPONSE_ID,
    Script,
    blocking,
    call_item,
    mcp_item,
    paths,
    refusal_item,
    respond,
    responses_client,
    snapshot,
    tool_runs,
)
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import WEATHER_REPORT_JSON, WeatherReport, resp_message

_STREAM_LOGGER = "chrys.service.llm.openai_responses.stream"
_NO_TERMINAL_WARNING = "ended without a terminal event"
_TOKEN = {"response_id": RESPONSE_ID}
# The three ways a stream is read: a new request, one parsed into a model, and
# a running response resumed by its token.
_STREAMS = ["create", "parsed", "retrieve"]


def _stream_options(mode: str) -> dict[str, Any]:
    match mode:
        case "parsed":
            return {"response_format": WeatherReport}
        case "retrieve":
            return {"continuation_token": dict(_TOKEN)}
        case _:
            return {}


def _answer(index: int = 0) -> Script:
    """A running response that has started its answer."""
    return Script().started().text(index, "msg_1", WEATHER_REPORT_JSON)


def _assert_failure(error: ProviderResponseError, code: str, *, retryable: bool) -> None:
    assert (error.code, error.retryable, error.invalidates_continuation_token) == (code, retryable, True)
    assert classify_error(error).retryable is retryable
    assert invalidates_continuation_token(error) is True


# ---------------------------------------------------------------------------
# Failures the response reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _STREAMS)
@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("server_error", True),
        ("rate_limit_exceeded", True),
        ("context_length_exceeded", False),
        ("invalid_prompt", False),
    ],
)
async def test_a_failed_stream_raises_the_error_it_reports(mode: str, code: str, retryable: bool) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await respond(
            _answer().failed(code=code, message="It broke.").reply(), stream=True, options=_stream_options(mode)
        )

    _assert_failure(raised.value, code, retryable=retryable)
    assert raised.value.provider_message == "It broke."


@pytest.mark.parametrize("mode", _STREAMS)
@pytest.mark.parametrize(
    ("code", "read", "retryable"),
    [
        ("rate_limit_exceeded", "rate_limit_exceeded", True),
        ("server_error", "server_error", True),
        (None, "server_error", True),
        ("context_length_exceeded", "context_length_exceeded", False),
    ],
)
async def test_an_error_event_ends_the_stream_with_its_error(
    mode: str, code: str | None, read: str, retryable: bool
) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await respond(_answer().error(code).reply(), stream=True, options=_stream_options(mode))

    _assert_failure(raised.value, read, retryable=retryable)


@pytest.mark.parametrize("poll", [False, True], ids=["create", "poll"])
@pytest.mark.parametrize(
    ("fields", "code", "retryable"),
    [
        ({"status": "failed", "error": {"code": "server_error", "message": "It broke."}}, "server_error", True),
        ({"status": "failed", "error": None}, "server_error", True),
        (
            {"status": "failed", "error": {"code": "insufficient_quota", "message": "Pay up."}},
            "insufficient_quota",
            False,
        ),
        ({"status": "cancelled"}, "cancelled", False),
    ],
    ids=["failed", "failed_without_error", "quota", "cancelled"],
)
async def test_a_blocking_response_that_failed_raises(
    poll: bool, fields: Mapping[str, Any], code: str, retryable: bool
) -> None:
    options = {"continuation_token": dict(_TOKEN)} if poll else {}

    with pytest.raises(ProviderResponseError) as raised:
        await respond(blocking(resp_message("msg_1", "Partial"), **fields), stream=False, options=options)

    _assert_failure(raised.value, code, retryable=retryable)


# ---------------------------------------------------------------------------
# Streams that end before their terminal event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _STREAMS)
async def test_a_stream_cut_off_before_its_terminal_event_is_a_resumable_truncation(mode: str) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await respond(_answer().reply(), stream=True, options=_stream_options(mode))

    error = raised.value
    assert (error.code, error.retryable, error.invalidates_continuation_token) == ("stream_truncated", True, False)
    assert classify_error(error).retryable is True
    assert invalidates_continuation_token(error) is False


async def test_a_stream_without_lifecycle_events_is_kept_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=_STREAM_LOGGER)
    script = Script().text(0, "msg_1", "Sunny.").call_added(1, "fc_1", "call_1").call_deltas(1, "fc_1")

    response, _ = await respond(script.reply(), stream=True)

    assert response.text == "Sunny."
    [call] = [content for content in response.messages[0].contents if content.type == "function_call"]
    assert (call.call_id, call.name, call.arguments) == ("call_1", "lookup", '{"city": "Paris"}')
    assert _NO_TERMINAL_WARNING in caplog.text


async def test_a_finished_stream_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=_STREAM_LOGGER)

    response, _ = await respond(_answer().finished(resp_message("msg_1", WEATHER_REPORT_JSON)).reply(), stream=True)

    assert response.text == WEATHER_REPORT_JSON
    assert caplog.text == ""


# ---------------------------------------------------------------------------
# Continuation tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "minted"),
    [("in_progress", True), ("queued", True), ("completed", False), ("failed", False), ("incomplete", False)],
)
async def test_only_a_running_response_can_be_resumed(status: str, minted: bool) -> None:
    script = Script().started()
    script.emit("response.in_progress", response={**script.events[0][1]["response"], "status": status})
    script.text(0, "msg_1", "Sunny.").finished(resp_message("msg_1", "Sunny."))

    _, updates = await respond(script.reply(), stream=True)

    created, in_progress = updates[0], updates[2]
    assert created.continuation_token == _TOKEN
    assert in_progress.continuation_token == (_TOKEN if minted else None)


async def _observe_tokens(
    *scripts: Script, validate: bool = False
) -> tuple[list[Any], BaseException | None, list[str]]:
    """Run one stream through the kernel: the tokens it reported, what it raised and the requests sent."""
    observed: list[Any] = []
    middleware = [ResponseValidationMiddleware(backoff_schedule=(0,))] if validate else []
    async with responses_client(*(script.reply() for script in scripts)) as (client, wire):
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=middleware))
        result = layer.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=True,
            client_kwargs={"continuation_token_observer": observed.append},
        )
        assert isinstance(result, ResponseStream)
        try:
            await result.get_final_response()
        except Exception as error:
            return observed, error, paths(wire.requests)
        return observed, None, paths(wire.requests)


async def test_a_failed_stream_drops_its_continuation_token() -> None:
    observed, error, _ = await _observe_tokens(_answer().failed())

    assert isinstance(error, ProviderResponseError)
    assert observed == [_TOKEN, None]


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(Script().started().finished(incomplete="content_filter"), id="nothing"),
        pytest.param(
            Script()
            .started()
            .refusal(0, "msg_1", "I can't help with that.")
            .finished(refusal_item("msg_1", "I can't help with that."), incomplete="content_filter"),
            id="refusal",
        ),
    ],
)
async def test_a_filtered_stream_drops_its_token_and_is_not_resumed(script: Script) -> None:
    observed, _, requests = await _observe_tokens(script, script, validate=True)

    assert observed == [_TOKEN, None]
    assert requests == ["POST /v1/responses"]


_SHOWN_REFUSAL = "I can't help with that."


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(
            Script()
            .started()
            .emit("response.output_item.added", output_index=0, item=refusal_item("msg_1", _SHOWN_REFUSAL))
            .finished(),
            id="added_message",
        ),
        pytest.param(
            Script()
            .started()
            .emit("response.output_item.done", output_index=0, item=refusal_item("msg_1", _SHOWN_REFUSAL))
            .finished(),
            id="done_message",
        ),
        pytest.param(
            Script()
            .started()
            .emit(
                "response.content_part.done",
                item_id="msg_1",
                output_index=0,
                content_index=0,
                part={"type": "refusal", "refusal": _SHOWN_REFUSAL},
            )
            .finished(),
            id="done_part",
        ),
        pytest.param(Script().started().finished(refusal_item("msg_1", _SHOWN_REFUSAL)), id="terminal_response"),
        pytest.param(
            Script().emit("response.output_item.done", output_index=0, item=refusal_item("msg_1", _SHOWN_REFUSAL)),
            id="done_message_without_lifecycle_events",
        ),
    ],
)
async def test_a_refusal_no_text_showed_is_reported_as_filtered(script: Script) -> None:
    """Only a message snapshot shows the refusal: the response is refused, not blank, so it is not sent again."""
    _, error, requests = await _observe_tokens(script, script, validate=True)

    assert error is not None
    assert (classify_error(error).kind, classify_error(error).retryable) == (ErrorKind.CONTENT_FILTERED, False)
    assert requests == ["POST /v1/responses"]


async def test_a_refusal_whose_text_streamed_is_the_answer() -> None:
    script = Script().started().refusal(0, "msg_1", _SHOWN_REFUSAL).finished(refusal_item("msg_1", _SHOWN_REFUSAL))

    response, _ = await respond(script.reply(), stream=True)

    assert response.text == _SHOWN_REFUSAL
    assert response.finish_reason is None


async def test_a_cut_off_stream_keeps_its_token_and_the_next_attempt_resumes_it() -> None:
    observed, error, requests = await _observe_tokens(_answer())

    assert isinstance(error, ProviderResponseError)
    assert error.code == "stream_truncated"
    assert observed == [_TOKEN]
    assert requests == ["POST /v1/responses"]

    resumed = _answer().finished(resp_message("msg_1", WEATHER_REPORT_JSON))
    async with responses_client(resumed.reply()) as (client, wire):
        result = client._inner_get_response(
            messages=[Message("user", ["What is the weather in Paris?"])],
            options={"continuation_token": observed[-1]},
            stream=True,
        )
        assert isinstance(result, ResponseStream)
        response = await result.get_final_response()

    assert response.text == WEATHER_REPORT_JSON
    assert paths(wire.requests) == [f"GET /v1/responses/{RESPONSE_ID}"]


# ---------------------------------------------------------------------------
# Hosted work a failed response showed
# ---------------------------------------------------------------------------


def _call_ids(error: ProviderResponseError) -> list[tuple[str, str | None]]:
    return [(content.type, content.call_id) for content in error.observed_contents]


async def test_a_failed_stream_reports_only_the_hosted_work_it_never_sent() -> None:
    script = Script().started().hosted(0, mcp_item("mcp_sent")).failed(mcp_item("mcp_sent"), mcp_item("mcp_unsent"))

    with pytest.raises(ProviderResponseError) as raised:
        await respond(script.reply(), stream=True)

    assert _call_ids(raised.value) == [("mcp_server_tool_call", "mcp_unsent"), ("mcp_server_tool_result", "mcp_unsent")]


@pytest.mark.parametrize("options", [{}, {"response_format": WeatherReport}], ids=["create", "parsed"])
async def test_a_failed_stream_reports_the_hosted_work_it_still_held(options: dict[str, Any]) -> None:
    script = Script().started().call_added(0, "fc_1", "call_1").call_deltas(0, "fc_1").hosted(1, mcp_item("mcp_held"))
    script.failed(call_item("fc_1", "call_1"), mcp_item("mcp_held"))

    with pytest.raises(ProviderResponseError) as raised:
        await respond(script.reply(), stream=True, options=options)

    assert _call_ids(raised.value) == [("mcp_server_tool_call", "mcp_held"), ("mcp_server_tool_result", "mcp_held")]


async def test_a_failed_blocking_response_reports_its_hosted_work() -> None:
    reply = blocking(mcp_item("mcp_1"), status="failed", error={"code": "server_error", "message": "It broke."})

    with pytest.raises(ProviderResponseError) as raised:
        await respond(reply, stream=False)

    assert _call_ids(raised.value) == [("mcp_server_tool_call", "mcp_1"), ("mcp_server_tool_result", "mcp_1")]


async def test_an_error_event_reports_no_hosted_work() -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await respond(Script().started().hosted(0, mcp_item("mcp_sent")).error("server_error").reply(), stream=True)

    assert raised.value.observed_contents == ()


# ---------------------------------------------------------------------------
# Usage a failed response reported
# ---------------------------------------------------------------------------


def _completed_with_an_error(*output: Mapping[str, Any]) -> dict[str, Any]:
    """A terminal response that reports an error although it says it completed."""
    return snapshot(*output, error={"code": "server_error", "message": "It broke."})


@pytest.mark.parametrize(
    ("reply", "stream"),
    [
        pytest.param(_answer().failed().reply(), True, id="failed_stream"),
        pytest.param(
            _answer().emit("response.completed", response=_completed_with_an_error()).reply(),
            True,
            id="completed_event_with_an_error",
        ),
        pytest.param(
            _answer().emit("response.failed", response=snapshot()).reply(), True, id="failed_event_without_an_error"
        ),
        pytest.param(
            blocking(resp_message("msg_1", "Partial"), status="failed", error={"code": "server_error", "message": "x"}),
            False,
            id="failed_blocking",
        ),
        pytest.param(
            Script()
            .started()
            .call(0, "fc_1", "call_1")
            .finished(call_item("fc_1", "call_1"), incomplete="content_filter")
            .reply(),
            True,
            id="refused_stream",
        ),
        pytest.param(
            blocking(call_item("fc_1", "call_1"), status="incomplete", incomplete="content_filter"),
            False,
            id="refused_blocking",
        ),
    ],
)
async def test_a_failed_response_carries_the_usage_it_reported(reply: Reply, stream: bool) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await respond(reply, stream=stream)

    usage = raised.value.usage_details
    assert usage is not None
    assert (usage["input_token_count"], usage["output_token_count"], usage["total_token_count"]) == (70, 9, 79)


@pytest.mark.parametrize("script", [_answer().error("server_error"), _answer()], ids=["error_event", "cut_off"])
async def test_a_stream_that_ends_without_a_response_carries_no_usage(script: Script) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await respond(script.reply(), stream=True)

    assert raised.value.usage_details is None


def _policy(validation: ResponseValidationMiddleware, retries: list[BaseException]) -> WireRetryPolicyAdapter:
    async def no_sleep(_seconds: int) -> bool:
        return False

    async def publish(_message: str, _attempt: int, _total: int, _delay: int, error: BaseException) -> None:
        retries.append(error)

    return WireRetryPolicyAdapter(
        max_retries=2,
        stall_timeout_seconds=None,
        stall_max_retries=0,
        stall_exhausted_action=StallExhaustedAction.BLOCKING_FALLBACK,
        backoff_schedule=(0,),
        interrupted=lambda: False,
        interruptible_sleep=no_sleep,
        publish_retry=publish,
        hosted_commits_in_flight=validation.hosted_commits_in_flight,
    )


@pytest.mark.parametrize("hosted", [True, False], ids=["hosted_work", "no_hosted_work"])
@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "blocking"])
async def test_hosted_work_on_a_failed_response_stops_the_wire_retry(stream: bool, hosted: bool) -> None:
    work = [mcp_item("mcp_1")] if hosted else []
    if stream:
        failed = Script().started().failed(*work).reply()
        recovered = Script().started().text(0, "msg_1", "Sunny.").finished(resp_message("msg_1", "Sunny.")).reply()
    else:
        failed = blocking(*work, status="failed", error={"code": "server_error", "message": "It broke."})
        recovered = blocking(resp_message("msg_1", "Sunny."))
    validation = ResponseValidationMiddleware(backoff_schedule=(0,))
    retries: list[BaseException] = []

    async with responses_client(failed, recovered) as (client, wire):
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=[validation]))
        result = layer.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=stream,
            # Local storage: the kernel retries the wire call in place.
            options={"store": False},
            client_kwargs={"wire_retry_policy": _policy(validation, retries)},
        )
        if hosted:
            with pytest.raises(ProviderResponseError, match="server_error"):
                await (result.get_final_response() if isinstance(result, ResponseStream) else result)
            assert paths(wire.requests) == ["POST /v1/responses"]
            assert retries == []
        else:
            response = await (result.get_final_response() if isinstance(result, ResponseStream) else result)
            assert response.text == "Sunny."
            assert paths(wire.requests) == ["POST /v1/responses", "POST /v1/responses"]
            assert len(retries) == 1


@pytest.mark.parametrize("hosted", [True, False], ids=["hosted_work", "no_hosted_work"])
async def test_hosted_work_held_behind_an_unfinished_call_stops_the_wire_retry(hosted: bool) -> None:
    cut = Script().started().call_added(0, "fc_1", "call_1").call_deltas(0, "fc_1")
    if hosted:
        # Ran, but waits behind the call, which the cut stream never finishes.
        cut.hosted(1, mcp_item("mcp_1"))
    recovered = Script().started().text(0, "msg_1", "Sunny.").finished(resp_message("msg_1", "Sunny."))
    validation = ResponseValidationMiddleware(backoff_schedule=(0,))
    retries: list[BaseException] = []

    async with responses_client(cut.reply(), recovered.reply()) as (client, wire):
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=[validation]))
        result = layer.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=True,
            options={"store": False},
            client_kwargs={"wire_retry_policy": _policy(validation, retries)},
        )
        assert isinstance(result, ResponseStream)
        if hosted:
            with pytest.raises(ProviderResponseError) as raised:
                await result.get_final_response()
            assert raised.value.code == "stream_truncated"
            assert paths(wire.requests) == ["POST /v1/responses"]
            assert retries == []
        else:
            response = await result.get_final_response()
            assert response.text == "Sunny."
            assert paths(wire.requests) == ["POST /v1/responses", "POST /v1/responses"]
            assert len(retries) == 1


async def test_hosted_work_on_a_failed_service_side_stream_is_recorded_as_committed() -> None:
    validation = ResponseValidationMiddleware(backoff_schedule=(0,))

    async with responses_client(Script().started().failed(mcp_item("mcp_1")).reply()) as (client, _):
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=[validation]))
        result = layer.get_response([Message("user", ["What is the weather in Paris?"])], stream=True)
        assert isinstance(result, ResponseStream)
        with pytest.raises(ProviderResponseError):
            await result.get_final_response()

    assert validation.hosted_commits_observed() != ()


# ---------------------------------------------------------------------------
# Refusals that ask for calls
# ---------------------------------------------------------------------------

_REFUSAL = "I can't help with that."

_REFUSED_STREAMS = [
    pytest.param(
        Script()
        .started()
        .refusal(0, "msg_1", _REFUSAL)
        .call(1, "fc_1", "call_1")
        .finished(refusal_item("msg_1", _REFUSAL), call_item("fc_1", "call_1")),
        id="refusal_first",
    ),
    pytest.param(
        Script()
        .started()
        .refusal(0, "msg_1", _REFUSAL)
        .call(1, "fc_1", "call_1")
        .finished(call_item("fc_1", "call_1")),
        id="refusal_only_in_the_stream",
    ),
    pytest.param(
        Script().refusal(0, "msg_1", _REFUSAL).call(1, "fc_1", "call_1"),
        id="refusal_in_a_stream_without_lifecycle_events",
    ),
    # A refusal only a message snapshot shows, with no terminal response to list it.
    pytest.param(
        Script()
        .emit("response.output_item.added", output_index=0, item=refusal_item("msg_1", _REFUSAL))
        .call(1, "fc_1", "call_1"),
        id="refusal_only_in_an_added_message",
    ),
    pytest.param(
        Script()
        .emit("response.output_item.done", output_index=0, item=refusal_item("msg_1", _REFUSAL))
        .call(1, "fc_1", "call_1"),
        id="refusal_only_in_a_done_message",
    ),
    pytest.param(
        Script()
        .emit(
            "response.content_part.done",
            item_id="msg_1",
            output_index=0,
            content_index=0,
            part={"type": "refusal", "refusal": _REFUSAL},
        )
        .call(1, "fc_1", "call_1"),
        id="refusal_only_in_a_done_part",
    ),
    pytest.param(
        Script()
        .started()
        .call(0, "fc_1", "call_1")
        .refusal(1, "msg_1", _REFUSAL)
        .finished(call_item("fc_1", "call_1"), refusal_item("msg_1", _REFUSAL)),
        id="refusal_after_the_call",
    ),
    pytest.param(
        Script()
        .started()
        .call(0, "fc_1", "call_1")
        .finished(call_item("fc_1", "call_1"), refusal_item("msg_1", _REFUSAL)),
        id="refusal_only_in_the_terminal_response",
    ),
    pytest.param(
        Script().started().call(0, "fc_1", "call_1").finished(call_item("fc_1", "call_1"), incomplete="content_filter"),
        id="filtered",
    ),
    # However the response then ends, the refusal stands: a retry could run the calls.
    pytest.param(
        Script()
        .started()
        .refusal(0, "msg_1", _REFUSAL)
        .call(1, "fc_1", "call_1")
        .failed(refusal_item("msg_1", _REFUSAL), call_item("fc_1", "call_1")),
        id="refusal_then_failed",
    ),
    pytest.param(
        Script().started().refusal(0, "msg_1", _REFUSAL).call(1, "fc_1", "call_1").error("server_error"),
        id="refusal_then_error_event",
    ),
    pytest.param(Script().started().refusal(0, "msg_1", _REFUSAL).call(1, "fc_1", "call_1"), id="refusal_then_cut_off"),
    pytest.param(
        Script().started().refusal(0, "msg_1", _REFUSAL).call(1, "fc_1", "call_1").breaks_off(),
        id="refusal_then_the_connection_breaks",
    ),
    pytest.param(
        Script()
        .started()
        .refusal(0, "msg_1", _REFUSAL)
        .call(1, "fc_1", "call_1")
        .emit("response.completed", response=_completed_with_an_error(call_item("fc_1", "call_1"))),
        id="refusal_then_a_completed_event_with_an_error",
    ),
    # A call the terminal response lists counts even when no event streamed it.
    pytest.param(
        Script()
        .started()
        .refusal(0, "msg_1", _REFUSAL)
        .failed(refusal_item("msg_1", _REFUSAL), call_item("fc_1", "call_1")),
        id="refusal_then_failed_with_the_call_only_in_its_response",
    ),
    pytest.param(
        Script().started().failed(refusal_item("msg_1", _REFUSAL), call_item("fc_1", "call_1")),
        id="refusal_and_call_only_in_a_failed_response",
    ),
    pytest.param(
        Script()
        .started()
        .refusal(0, "msg_1", _REFUSAL)
        .emit(
            "response.completed",
            response=_completed_with_an_error(refusal_item("msg_1", _REFUSAL), call_item("fc_1", "call_1")),
        ),
        id="refusal_then_a_completed_event_with_an_error_listing_the_call",
    ),
    pytest.param(
        Script().started().finished(call_item("fc_1", "call_1"), incomplete="content_filter"),
        id="call_only_in_a_filtered_response",
    ),
]


def _assert_refused(result: Any) -> None:
    assert result.runs == []
    assert len(result.requests) == 1
    assert result.error is not None
    _assert_failure(result.error, "content_filter", retryable=False)


@pytest.mark.parametrize("script", _REFUSED_STREAMS)
async def test_a_refused_stream_runs_none_of_its_calls(script: Script) -> None:
    _assert_refused(await tool_runs(script.reply()))


@pytest.mark.parametrize(
    ("refusal", "call"),
    [(True, True), (True, False), (False, True)],
    ids=["refusal_with_a_call", "refusal_only", "call_only"],
)
async def test_a_stream_that_breaks_off_is_sent_again_unless_it_refused_with_calls(refusal: bool, call: bool) -> None:
    refused = refusal and call
    broken = Script().started()
    if refusal:
        broken.refusal(0, "msg_1", _REFUSAL)
    if call:
        broken.call(1, "fc_1", "call_1")
    broken.breaks_off()
    recovered = Script().started().text(0, "msg_2", "Sunny.").finished(resp_message("msg_2", "Sunny."))
    validation = ResponseValidationMiddleware(backoff_schedule=(0,))
    retries: list[BaseException] = []

    async with responses_client(broken.reply(), recovered.reply()) as (client, wire):
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=[validation]))
        result = layer.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=True,
            # Local storage: the kernel retries the wire call in place.
            options={"store": False},
            client_kwargs={"wire_retry_policy": _policy(validation, retries)},
        )
        assert isinstance(result, ResponseStream)
        if refused:
            with pytest.raises(ProviderResponseError) as raised:
                await result.get_final_response()
            _assert_failure(raised.value, "content_filter", retryable=False)
            assert paths(wire.requests) == ["POST /v1/responses"]
            assert retries == []
        else:
            response = await result.get_final_response()
            assert response.text == "Sunny."
            assert paths(wire.requests) == ["POST /v1/responses", "POST /v1/responses"]
            assert len(retries) == 1


@pytest.mark.parametrize("mode", ["create", "parsed"])
async def test_a_stream_is_not_read_past_its_terminal_event(mode: str) -> None:
    # Sent again, an answer that ran hosted work could not be, and a filtered
    # one would meet the filter again.
    finished = _answer().finished(resp_message("msg_1", WEATHER_REPORT_JSON)).breaks_off()
    again = _answer().finished(resp_message("msg_1", WEATHER_REPORT_JSON))
    validation = ResponseValidationMiddleware(backoff_schedule=(0,))
    retries: list[BaseException] = []

    async with responses_client(finished.reply(), again.reply()) as (client, wire):
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=[validation]))
        result = layer.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=True,
            options={"store": False, **_stream_options(mode)},
            client_kwargs={"wire_retry_policy": _policy(validation, retries)},
        )
        assert isinstance(result, ResponseStream)
        response = await result.get_final_response()

    assert response.text == WEATHER_REPORT_JSON
    assert paths(wire.requests) == ["POST /v1/responses"]
    assert retries == []


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(blocking(refusal_item("msg_1", _REFUSAL), call_item("fc_1", "call_1")), id="refusal"),
        pytest.param(
            blocking(call_item("fc_1", "call_1"), status="incomplete", incomplete="content_filter"), id="filtered"
        ),
        pytest.param(
            blocking(
                refusal_item("msg_1", _REFUSAL),
                call_item("fc_1", "call_1"),
                status="failed",
                error={"code": "server_error", "message": "It broke."},
            ),
            id="refusal_in_a_failed_response",
        ),
    ],
)
async def test_a_refused_blocking_response_runs_none_of_its_calls(reply: Reply) -> None:
    _assert_refused(await tool_runs(reply, stream=False))


@pytest.mark.parametrize(
    ("reply", "stream", "code"),
    [
        pytest.param(
            Script().started().refusal(0, "msg_1", _REFUSAL).failed(refusal_item("msg_1", _REFUSAL)).reply(),
            True,
            "server_error",
            id="failed_stream",
        ),
        pytest.param(
            Script().started().refusal(0, "msg_1", _REFUSAL).reply(), True, "stream_truncated", id="cut_off_stream"
        ),
        pytest.param(
            blocking(
                refusal_item("msg_1", _REFUSAL), status="failed", error={"code": "server_error", "message": "It broke."}
            ),
            False,
            "server_error",
            id="failed_blocking",
        ),
    ],
)
async def test_a_refusal_without_calls_keeps_the_failure_it_ended_with(reply: Reply, stream: bool, code: str) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await respond(reply, stream=stream)

    assert (raised.value.code, raised.value.retryable) == (code, True)


async def test_a_refusal_that_asks_for_calls_reports_the_hosted_work_it_never_sent() -> None:
    script = (
        Script()
        .started()
        .call(0, "fc_1", "call_1")
        .finished(call_item("fc_1", "call_1"), mcp_item("mcp_1"), refusal_item("msg_1", _REFUSAL))
    )

    with pytest.raises(ProviderResponseError) as raised:
        await respond(script.reply(), stream=True)

    assert _call_ids(raised.value) == [("mcp_server_tool_call", "mcp_1"), ("mcp_server_tool_result", "mcp_1")]


@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "blocking"])
async def test_a_refusal_without_calls_is_an_ordinary_answer(stream: bool) -> None:
    reply = (
        Script().started().refusal(0, "msg_1", _REFUSAL).finished(refusal_item("msg_1", _REFUSAL)).reply()
        if stream
        else blocking(refusal_item("msg_1", _REFUSAL))
    )

    response, _ = await respond(reply, stream=stream)

    assert response.text == _REFUSAL


@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "blocking"])
async def test_calls_of_an_unrefused_response_run(stream: bool) -> None:
    if stream:
        calls = Script().started().call(0, "fc_1", "call_1").finished(call_item("fc_1", "call_1")).reply()
        answer = Script().started().text(0, "msg_2", "Sunny.").finished(resp_message("msg_2", "Sunny.")).reply()
    else:
        calls = blocking(call_item("fc_1", "call_1"))
        answer = blocking(resp_message("msg_2", "Sunny."))

    result = await tool_runs(calls, answer, stream=stream)

    assert result.error is None
    assert result.runs == ["Paris"]
    assert len(result.requests) == 2
