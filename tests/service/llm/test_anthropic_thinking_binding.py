# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Thinking Anthropic refuses as bound to a different conversation is stripped, resent once and marked.

Every case runs the production client stack over the real SDK; the transport
answers from a script and keeps the requests it received.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import anthropic
import httpx
import pytest

from chrys.foundation.errors import is_thinking_binding_rejection
from chrys.foundation.models.history_markers import ANTHROPIC_THINKING_STRIPPED_KEY
from chrys.foundation.reasoning_origin import ReasoningOrigin
from chrys.kernel import ChatResponse, Content, Message, wire_progress_scope
from chrys.service.llm.anthropic_messages import client as client_module
from chrys.service.llm.clients import create_client
from chrys.service.profiles.models.options import effective_chat_options
from chrys.service.profiles.models.schema import ModelProfile, ThinkingBlockBinding
from tests.kernel._fakes import _OverflowSink, _WireRetryPolicy
from tests.support.provider_errors import anthropic_thinking_binding_body
from tests.support.scripted_wire import ScriptedWire, pin_wire_inputs, route_clients_to
from tests.support.waiting import wait_for
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import (
    anth_events,
    anth_message,
    anth_replies,
    anth_text,
    json_reply,
    lookup_tool,
    sse_reply,
)

MODES = pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
_BASE_URL = "https://gateway.example"


def _error_reply(status: int, body: dict[str, Any]) -> Reply:
    return Reply(status, json.dumps(body).encode("utf-8"), (("content-type", "application/json"),))


def _error(status: int, error_type: str, message: str) -> Reply:
    return _error_reply(status, {"type": "error", "error": {"type": error_type, "message": message}})


_REFUSAL = _error_reply(400, anthropic_thinking_binding_body())
_OVERFLOW = _error(400, "invalid_request_error", "prompt is too long: 213462 tokens > 200000 maximum")


@pytest.fixture(autouse=True)
def _pinned_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    pin_wire_inputs(monkeypatch)


def _ok(text: str = "Done.", *, stream: bool, thinking: tuple[str, str] | None = None) -> Reply:
    content: list[dict[str, Any]] = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking[0], "signature": thinking[1]})
    content.append({"type": "text", "text": text})
    message = anth_message(message_id=f"msg_{text}", content=content)
    return sse_reply(anth_events(message)) if stream else json_reply(message)


def _profile(*, binding: ThinkingBlockBinding = "auto", chat_options: dict[str, Any] | None = None) -> ModelProfile:
    return ModelProfile(
        id="claude-profile",
        name="claude",
        provider="anthropic",
        model_id="claude-opus-5-5",
        api_key="sk-ant-test",
        base_url=_BASE_URL,
        http_max_retries=0,
        chat_options=json.dumps(chat_options) if chat_options is not None else "",
        thinking_block_binding=binding,
    )


def _thinking(text: str, signature: str) -> Content:
    return Content.from_text_reasoning(text=text, protected_data=signature)


def _redacted() -> Content:
    return Content.from_text_reasoning(
        protected_data="opaque", additional_properties={"anthropic_redacted_thinking": True}
    )


def _issued_elsewhere(text: str, signature: str) -> Content:
    """Signed thinking another endpoint issued, which no request here replays."""
    content = _thinking(text, signature)
    ReasoningOrigin(client_module.REASONING_PROTOCOL, "https://api.anthropic.com:443").stamp(
        content.additional_properties
    )
    return content


def _tool_turn(*reasoning: Content) -> list[Message]:
    """A history whose last assistant message thinks, then calls a tool the next message answers."""
    return [
        Message("user", ["What is the weather in Paris?"]),
        Message(
            "assistant",
            [
                *reasoning,
                Content.from_text("Let me check."),
                Content.from_function_call(call_id="toolu_1", name="lookup", arguments='{"city": "Paris"}'),
            ],
        ),
        Message("tool", [Content.from_function_result(call_id="toolu_1", result="sunny")]),
    ]


async def _exchange(
    messages: list[Message],
    replies: list[Reply],
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: bool,
    profile: ModelProfile | None = None,
    extra_options: dict[str, Any] | None = None,
) -> tuple[ChatResponse | anthropic.APIStatusError, list[dict[str, Any]]]:
    """Send *messages* once through the client; its response or error, and every request body sent.

    *extra_options* are added to the profile's options as code passing its own would.
    """
    wire = ScriptedWire(replies)
    route_clients_to(wire.transport, monkeypatch)
    profile = profile or _profile()
    stack = await create_client(profile)
    options = {**(effective_chat_options(profile) or {}), **(extra_options or {})}
    outcome: ChatResponse | anthropic.APIStatusError
    try:
        result = stack.inner.get_response(messages, stream=stream, options=options)
        try:
            outcome = await (result.get_final_response() if stream else result)
        except anthropic.APIStatusError as exc:
            outcome = exc
    finally:
        await stack.aclose()
    return outcome, [json.loads(request.content) for request in wire.requests]


async def _call(
    messages: list[Message],
    replies: list[Reply],
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: bool,
    profile: ModelProfile | None = None,
) -> tuple[ChatResponse, list[dict[str, Any]]]:
    response, sent = await _exchange(messages, replies, monkeypatch, stream=stream, profile=profile)
    assert isinstance(response, ChatResponse), response
    return response, sent


async def _failed_call(
    messages: list[Message],
    replies: list[Reply],
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: bool,
    profile: ModelProfile | None = None,
    extra_options: dict[str, Any] | None = None,
) -> tuple[anthropic.APIStatusError, list[dict[str, Any]]]:
    error, sent = await _exchange(
        messages, replies, monkeypatch, stream=stream, profile=profile, extra_options=extra_options
    )
    assert isinstance(error, anthropic.APIStatusError), error
    return error, sent


_THINKING_BLOCKS = ("thinking", "redacted_thinking")


def _thinking_blocks(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for message in body["messages"] for block in message["content"] if block["type"] in _THINKING_BLOCKS]


def _without_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*messages* with every thinking block left out, and every message that held nothing else."""
    kept = [
        {**message, "content": [block for block in message["content"] if block["type"] not in _THINKING_BLOCKS]}
        for message in messages
    ]
    return [message for message in kept if message["content"]]


def _refusals_chained_to(exc: BaseException) -> int:
    """How many refusals *exc* and the exceptions it was raised while handling are."""
    count = 0
    link: BaseException | None = exc
    while link is not None:
        count += is_thinking_binding_rejection(link)
        link = link.__context__
    return count


def _stripped(content: Content) -> bool:
    return content.additional_properties.get(ANTHROPIC_THINKING_STRIPPED_KEY) is True


@MODES
async def test_a_refused_request_is_resent_once_without_its_thinking(
    stream: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    thinking = _thinking("The user wants the weather.", "sig-1")

    with caplog.at_level(logging.WARNING, logger=client_module.__name__):
        response, (refused, resent) = await _call(
            _tool_turn(thinking), [_REFUSAL, _ok(stream=stream)], monkeypatch, stream=stream
        )

    assert response.text == "Done."
    assert _thinking_blocks(refused) == [
        {"type": "thinking", "thinking": "The user wants the weather.", "signature": "sig-1"}
    ]
    assert resent["messages"] == _without_thinking(refused["messages"])
    assert {key: value for key, value in resent.items() if key != "messages"} == {
        key: value for key, value in refused.items() if key != "messages"
    }
    assert _stripped(thinking)
    [warning] = [record for record in caplog.records if record.name == client_module.__name__]
    assert "resent" in warning.getMessage()


@MODES
async def test_later_requests_leave_out_the_marked_thinking_and_keep_new_thinking(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = _tool_turn(_thinking("old", "sig-old"))
    response, (_, resent) = await _call(
        history,
        [_REFUSAL, _ok("It is sunny.", stream=stream, thinking=("fresh", "sig-new"))],
        monkeypatch,
        stream=stream,
    )
    follow_up = [*history, *response.messages, Message("user", ["And tomorrow?"])]

    _, [next_request] = await _call(follow_up, [_ok(stream=stream)], monkeypatch, stream=stream)

    assert next_request["messages"][: len(resent["messages"])] == resent["messages"]
    assert _thinking_blocks(next_request) == [{"type": "thinking", "thinking": "fresh", "signature": "sig-new"}]


@MODES
async def test_a_second_refusal_is_raised_and_nothing_is_marked(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    thinking = _thinking("t", "sig-1")

    raised, sent = await _failed_call(_tool_turn(thinking), [_REFUSAL, _REFUSAL], monkeypatch, stream=stream)

    assert is_thinking_binding_rejection(raised)
    assert _refusals_chained_to(raised) == 1
    assert len(sent) == 2
    assert not _stripped(thinking)


@MODES
@pytest.mark.parametrize(
    ("second", "status"),
    [
        pytest.param(_error(429, "rate_limit_error", "Slow down."), 429, id="rate_limited"),
        pytest.param(_error(500, "api_error", "Internal error."), 500, id="server_error"),
        pytest.param(_OVERFLOW, 400, id="context_overflow"),
    ],
)
async def test_another_failure_of_the_resend_is_raised_as_it_is(
    second: Reply, status: int, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    thinking = _thinking("t", "sig-1")

    raised, sent = await _failed_call(_tool_turn(thinking), [_REFUSAL, second], monkeypatch, stream=stream)

    assert raised.status_code == status
    assert _refusals_chained_to(raised) == 0
    assert len(sent) == 2
    assert not _stripped(thinking)


@MODES
@pytest.mark.parametrize(
    "profile",
    [
        pytest.param(_profile(binding="error"), id="setting"),
        pytest.param(
            _profile(
                chat_options={"thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}}
            ),
            id="written",
        ),
    ],
)
async def test_a_request_that_asked_for_the_refusal_gets_it(
    profile: ModelProfile, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    thinking = _thinking("t", "sig-1")

    raised, sent = await _failed_call(_tool_turn(thinking), [_REFUSAL], monkeypatch, stream=stream, profile=profile)

    assert is_thinking_binding_rejection(raised)
    assert len(sent) == 1
    assert not _stripped(thinking)


@MODES
@pytest.mark.parametrize("binding", ["auto", "drop_block", "off"])
async def test_every_other_setting_resends(
    binding: ThinkingBlockBinding, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    thinking = _thinking("t", "sig-1")

    response, sent = await _call(
        _tool_turn(thinking),
        [_REFUSAL, _ok(stream=stream)],
        monkeypatch,
        stream=stream,
        profile=_profile(binding=binding),
    )

    assert response.text == "Done."
    assert len(sent) == 2
    assert _stripped(thinking)


@MODES
async def test_a_refusal_with_no_thinking_sent_is_raised(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    foreign = _issued_elsewhere("from elsewhere", "sig-elsewhere")
    history = _tool_turn(foreign, Content.from_text_reasoning(text="unsigned"))

    raised, sent = await _failed_call(history, [_REFUSAL], monkeypatch, stream=stream)

    assert is_thinking_binding_rejection(raised)
    assert len(sent) == 1
    assert _thinking_blocks(sent[0]) == []
    assert not _stripped(foreign)


@MODES
async def test_a_request_sending_its_own_messages_is_not_resent(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Messages ``extra_body`` sends replace the encoded history: none of its thinking was sent to leave out.

    A profile cannot set them; only code passing its own options can.
    """
    thinking = _thinking("t", "sig-1")
    own_messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    raised, sent = await _failed_call(
        _tool_turn(thinking),
        [_REFUSAL],
        monkeypatch,
        stream=stream,
        extra_options={"extra_body": {"messages": own_messages}},
    )

    assert is_thinking_binding_rejection(raised)
    assert [body["messages"] for body in sent] == [own_messages]
    assert not _stripped(thinking)


@MODES
async def test_only_the_reasoning_the_request_sent_is_marked(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    unsigned = Content.from_text_reasoning(text="a")
    signed = _thinking("b", "sig-b")
    # A signature fragment after a signed block signs nothing, here or after the strip.
    stray_signature = Content.from_text_reasoning(protected_data="sig-stray")
    split_text = Content.from_text_reasoning(text="c")
    split_signature = Content.from_text_reasoning(protected_data="sig-c")
    # Thinking streamed without its text: an empty block its signature signs.
    omitted = Content.from_text_reasoning(protected_data="")
    omitted_signature = Content.from_text_reasoning(protected_data="sig-e")
    redacted = _redacted()
    # Thinking without text that no signature follows is never sent.
    lone_omitted = Content.from_text_reasoning(protected_data="")
    foreign = _issued_elsewhere("d", "sig-d")
    history = _tool_turn(
        unsigned,
        signed,
        stray_signature,
        foreign,
        split_text,
        split_signature,
        omitted,
        omitted_signature,
        redacted,
        lone_omitted,
    )

    _, (refused, resent) = await _call(history, [_REFUSAL, _ok(stream=stream)], monkeypatch, stream=stream)

    assert _thinking_blocks(refused) == [
        {"type": "thinking", "thinking": "b", "signature": "sig-b"},
        {"type": "thinking", "thinking": "c", "signature": "sig-c"},
        {"type": "thinking", "thinking": "", "signature": "sig-e"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]
    assert _thinking_blocks(resent) == []
    sent = (signed, split_text, split_signature, omitted, omitted_signature, redacted)
    assert [_stripped(content) for content in sent] == [True] * 6
    assert [_stripped(content) for content in (unsigned, stray_signature, foreign, lone_omitted)] == [False] * 4

    _, [next_request] = await _call(history, [_ok(stream=stream)], monkeypatch, stream=stream)

    assert next_request["messages"] == resent["messages"]


@MODES
async def test_an_equal_copy_of_marked_thinking_is_still_sent(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mark belongs to the contents the refused request replayed, not to every content equal to one."""
    replayed = _thinking("t", "sig-1")
    await _call(_tool_turn(replayed), [_REFUSAL, _ok(stream=stream)], monkeypatch, stream=stream)
    equal = _thinking("t", "sig-1")

    _, [request] = await _call(_tool_turn(replayed, equal), [_ok(stream=stream)], monkeypatch, stream=stream)

    assert _stripped(replayed)
    assert _thinking_blocks(request) == [{"type": "thinking", "thinking": "t", "signature": "sig-1"}]
    assert not _stripped(equal)


@MODES
@pytest.mark.parametrize("kind", ["signed", "redacted"])
async def test_an_assistant_message_of_only_thinking_is_left_out_whole(
    kind: str, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    reasoning = _thinking("t", "sig-1") if kind == "signed" else _redacted()
    history = [
        Message("user", ["Look it up."]),
        Message("assistant", [reasoning]),
        Message("assistant", [Content.from_function_call(call_id="functions.lookup:0", name="lookup", arguments="{}")]),
        Message("tool", [Content.from_function_result(call_id="functions.lookup:0", result="sunny")]),
    ]

    _, (refused, resent) = await _call(history, [_REFUSAL, _ok(stream=stream)], monkeypatch, stream=stream)

    # Equal but for the thinking: the tool ids are mapped the same way in both requests.
    assert resent["messages"] == _without_thinking(refused["messages"])
    assert all(message["content"] for message in resent["messages"])
    tool_ids = [
        (block["type"], block.get("id") or block.get("tool_use_id"))
        for message in resent["messages"]
        for block in message["content"]
        if block["type"] in ("tool_use", "tool_result")
    ]
    [(_, call_id), (_, result_id)] = tool_ids
    assert call_id == result_id
    assert call_id != "functions.lookup:0"


@MODES
async def test_the_resend_restarts_the_stall_timer(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal is the service answering: the resend's wait for its first byte is timed afresh."""
    wire = ScriptedWire([_REFUSAL, _ok(stream=stream)])
    route_clients_to(wire.transport, monkeypatch)
    profile = _profile()
    stack = await create_client(profile)
    sent_at_each_report: list[int] = []
    try:
        with wire_progress_scope(lambda: sent_at_each_report.append(len(wire.requests))):
            result = stack.inner.get_response(
                _tool_turn(_thinking("t", "sig-1")), stream=stream, options=effective_chat_options(profile) or {}
            )
            await (result.get_final_response() if stream else result)
    finally:
        await stack.aclose()

    assert len(wire.requests) == 2
    assert 1 in sent_at_each_report


async def test_concurrent_calls_on_one_client_mark_only_the_resent_calls_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused_thinking = _thinking("refused", "sig-refused")
    accepted_thinking = _thinking("accepted", "sig-accepted")
    both_sent = asyncio.Event()
    seen: list[str] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        seen.append(body)
        if len(seen) == 2:
            both_sent.set()
        await wait_for(both_sent.is_set, description="both calls sent")
        if "sig-refused" in body:
            return httpx.Response(400, json=anthropic_thinking_binding_body(), request=request)
        return httpx.Response(200, json=anth_text("Done.", message_id="msg_ok"), request=request)

    route_clients_to(httpx.MockTransport(answer), monkeypatch)
    profile = _profile()
    stack = await create_client(profile)
    options = effective_chat_options(profile) or {}
    try:
        first, second = await asyncio.gather(
            stack.inner.get_response(_tool_turn(refused_thinking), options=options),
            stack.inner.get_response(_tool_turn(accepted_thinking), options=options),
        )
    finally:
        await stack.aclose()

    assert (first.text, second.text) == ("Done.", "Done.")
    assert len(seen) == 3
    assert _stripped(refused_thinking)
    assert not _stripped(accepted_thinking)


async def test_a_stream_that_fails_after_the_resend_is_accepted_keeps_the_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinking = _thinking("t", "sig-1")
    [accepted] = anth_replies([anth_text("A long answer.", message_id="msg_cut")], stream=True)
    cut_off = Reply(accepted.status, accepted.body[: len(accepted.body) // 2], accepted.headers, breaks_off=True)
    closes: list[object] = []
    close = client_module._close_event_stream

    async def counting_close(events: Any) -> None:
        closes.append(events)
        await close(events)

    monkeypatch.setattr(client_module, "_close_event_stream", counting_close)
    wire = ScriptedWire([_REFUSAL, cut_off])
    route_clients_to(wire.transport, monkeypatch)
    profile = _profile()
    stack = await create_client(profile)
    try:
        result = stack.inner.get_response(
            _tool_turn(thinking), stream=True, options=effective_chat_options(profile) or {}
        )
        with pytest.raises(httpx.ReadError):
            await result.get_final_response()
    finally:
        await stack.aclose()

    assert len(wire.requests) == 2
    assert _stripped(thinking)
    assert len(closes) == 1


def _lookup_call(*, stream: bool) -> Reply:
    message = anth_message(
        message_id="msg_call",
        content=[
            {"type": "thinking", "thinking": "Let me look it up.", "signature": "sig-call"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Paris"}},
        ],
        stop_reason="tool_use",
    )
    return sse_reply(anth_events(message)) if stream else json_reply(message)


@MODES
@pytest.mark.parametrize(
    ("after_the_call", "requests"),
    [
        pytest.param(["refusal", "ok"], 2, id="refusal"),
        pytest.param(["overflow", "refusal", "ok"], 3, id="overflow_then_refusal"),
        pytest.param(["refusal", "overflow", "refusal", "ok"], 4, id="refusal_overflow_refusal"),
    ],
)
async def test_the_resend_stacks_with_the_loops_overflow_resend(
    after_the_call: list[str], requests: int, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each recovery resends once per call: at most four requests for the model call after the tool."""
    replies = {"refusal": _REFUSAL, "overflow": _OVERFLOW, "ok": _ok(stream=stream)}
    wire = ScriptedWire([_lookup_call(stream=stream), *(replies[name] for name in after_the_call)])
    route_clients_to(wire.transport, monkeypatch)
    profile = _profile()
    stack = await create_client(profile)
    sink = _OverflowSink()
    policy = _WireRetryPolicy()
    try:
        result = stack.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=stream,
            options={**(effective_chat_options(profile) or {}), "tools": [lookup_tool()]},
            compaction_strategy=sink,
            client_kwargs={"wire_retry_policy": policy},
        )
        response = await (result.get_final_response() if stream else result)
    finally:
        await stack.aclose()

    assert response.text == "Done."
    assert len(wire.requests) == 1 + requests
    assert len(sink.notes) == after_the_call.count("overflow")
    assert [retry[3] for retry in policy.retries] == sink.notes


@MODES
async def test_a_second_refusal_in_the_loop_ends_the_call(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    wire = ScriptedWire([_lookup_call(stream=stream), _REFUSAL, _REFUSAL])
    route_clients_to(wire.transport, monkeypatch)
    profile = _profile()
    stack = await create_client(profile)
    sink = _OverflowSink()
    policy = _WireRetryPolicy()
    try:
        result = stack.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=stream,
            options={**(effective_chat_options(profile) or {}), "tools": [lookup_tool()]},
            compaction_strategy=sink,
            client_kwargs={"wire_retry_policy": policy},
        )
        with pytest.raises(anthropic.APIStatusError) as raised:
            await (result.get_final_response() if stream else result)
    finally:
        await stack.aclose()

    assert is_thinking_binding_rejection(raised.value)
    assert len(wire.requests) == 3
    assert sink.notes == []
    assert policy.retries == []
