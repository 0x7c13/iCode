# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reasoning only the issuing endpoint can read replays only to that endpoint.

Each protocol's real client stack receives reasoning from one base URL, then
sends the same history to another base URL and to the first one spelled
another way. Anthropic thinking, Responses encrypted reasoning and Chat
Completions ``reasoning_details`` reach only the endpoint that issued them;
plaintext reasoning and history captured before stamps existed replay as
before.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from chrys.foundation.reasoning_origin import REASONING_ORIGIN_KEY, ReasoningOrigin
from chrys.kernel import ChatResponse, Content, Message
from chrys.service.llm.anthropic_messages.history import encode_messages as anthropic_encode
from chrys.service.llm.chat_completions.client import OPENAI
from chrys.service.llm.chat_completions.history import encode_messages as chat_completions_encode
from chrys.service.llm.chat_completions.stream import StreamState as ChatCompletionsStream
from chrys.service.llm.clients import create_client
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.replay import encode_input
from chrys.service.profiles.models.options import effective_chat_options
from chrys.service.profiles.models.schema import API_STYLE_CHAT_COMPLETIONS, API_STYLE_RESPONSES, ModelProfile
from tests.support.scripted_wire import ScriptedWire, pin_wire_inputs, route_clients_to
from tests.support.wire_cases._kit import (
    anth_message,
    anth_replies,
    anth_text,
    cc_chunks,
    cc_completion,
    cc_replies,
    cc_text,
    resp_function_call,
    resp_message,
    resp_reasoning,
    resp_replies,
    resp_response,
    sse_reply,
)

MODES = pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
_SESSION_ID = "origin-session"


def _profile(provider: str, base_url: str, *, api_style: str = API_STYLE_CHAT_COMPLETIONS) -> ModelProfile:
    return ModelProfile(
        id=f"profile-{provider}-{base_url}",
        name="origin",
        provider=provider,
        api_style=api_style,
        model_id="origin-model",
        api_key="sk-origin",
        base_url=base_url,
    )


async def _exchange(
    profile: ModelProfile,
    messages: list[Message],
    replies: tuple[Any, ...],
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: bool,
) -> tuple[ChatResponse, dict[str, Any]]:
    """Send *messages* through the real stack for *profile*; the response and the request body sent."""
    wire = ScriptedWire(replies)
    route_clients_to(wire.transport, monkeypatch)
    stack = await create_client(profile, session_id=_SESSION_ID)
    try:
        result = stack.inner.get_response(messages, stream=stream, options=effective_chat_options(profile) or {})
        response = await (result.get_final_response() if stream else result)
    finally:
        await stack.aclose()
    (request,) = wire.requests
    return response, json.loads(request.content)


def _question() -> list[Message]:
    return [Message("system", ["Be brief."]), Message("user", ["What is the weather in Paris?"])]


# ---------------------------------------------------------------------------
# The real stacks, from one endpoint to another
# ---------------------------------------------------------------------------


@MODES
async def test_anthropic_thinking_replays_only_to_the_endpoint_that_signed_it(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin_wire_inputs(monkeypatch)
    gateway = _profile("anthropic", "https://gateway.example/api/anthropic")
    official = _profile("anthropic", "")
    gateway_respelled = _profile("anthropic", "HTTPS://Gateway.Example:443/api/anthropic/")
    first = anth_message(
        message_id="msg_origin_1",
        content=[
            {"type": "thinking", "thinking": "The user wants the weather.", "signature": "sig-gateway"},
            {"type": "text", "text": "Sunny."},
        ],
    )
    response, _ = await _exchange(
        gateway, _question(), anth_replies([first], stream=stream), monkeypatch, stream=stream
    )
    history = [*_question(), *response.messages, Message("user", ["And tomorrow?"])]

    _, to_official = await _exchange(
        official,
        history,
        anth_replies([anth_text("Rain.", message_id="msg_2")], stream=False),
        monkeypatch,
        stream=False,
    )
    _, to_gateway = await _exchange(
        gateway_respelled,
        history,
        anth_replies([anth_text("Rain.", message_id="msg_3")], stream=False),
        monkeypatch,
        stream=False,
    )

    assert to_official["messages"][1] == {"role": "assistant", "content": [{"type": "text", "text": "Sunny."}]}
    assert to_gateway["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "The user wants the weather.", "signature": "sig-gateway"},
            {"type": "text", "text": "Sunny."},
        ],
    }


@MODES
async def test_responses_encrypted_reasoning_replays_only_to_the_endpoint_that_encrypted_it(
    stream: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pin_wire_inputs(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="chrys.service.llm.openai_responses.replay")
    org = _profile("openai", "https://org-a.example/v1", api_style=API_STYLE_RESPONSES)
    official = _profile("openai", "", api_style=API_STYLE_RESPONSES)
    org_respelled = _profile("openai", "https://ORG-A.example:443/v1/", api_style=API_STYLE_RESPONSES)
    first = resp_response(
        response_id="resp_origin_1",
        output=[
            resp_reasoning("rs_origin_1", "The user wants the weather.", encrypted="enc-org-a"),
            resp_message("msg_origin_1", "Let me check."),
            resp_function_call("fc_origin_1", "call_lookup_1"),
        ],
    )
    response, _ = await _exchange(org, _question(), resp_replies([first], stream=stream), monkeypatch, stream=stream)
    history = [
        *_question(),
        *response.messages,
        Message("tool", [Content.from_function_result(call_id="call_lookup_1", result="sunny")]),
    ]
    final = resp_response(response_id="resp_origin_2", output=[resp_message("msg_origin_2", "Sunny.")])

    _, to_official = await _exchange(official, history, resp_replies([final], stream=False), monkeypatch, stream=False)
    _, to_org = await _exchange(org_respelled, history, resp_replies([final], stream=False), monkeypatch, stream=False)

    def replayed(body: dict[str, Any]) -> list[tuple[str, Any, Any]]:
        return [
            (item["type"], item.get("id"), item.get("encrypted_content"))
            for item in body["input"]
            if item.get("type") in ("reasoning", "function_call")
        ]

    assert replayed(to_official) == [("function_call", None, None)]
    assert replayed(to_org) == [("reasoning", "rs_origin_1", "enc-org-a"), ("function_call", "fc_origin_1", None)]
    # Leaving out another endpoint's reasoning is expected after a switch, not a fault worth a warning.
    replay_logs = [(record.levelno, record.getMessage()) for record in caplog.records]
    assert [message for level, message in replay_logs if level >= logging.WARNING] == []
    assert [message for _, message in replay_logs] == ["Left out reasoning another endpoint issued: groups=[2]"]


async def test_deepseek_responses_plaintext_reasoning_replays_to_any_endpoint(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pin_wire_inputs(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="chrys.service.llm.openai_responses.replay")
    official = _profile("deepseek-openai", "https://api.deepseek.com/v1", api_style=API_STYLE_RESPONSES)
    gateway = _profile("deepseek-openai", "https://gateway.example/v1", api_style=API_STYLE_RESPONSES)
    reasoning = {
        "type": "reasoning",
        "id": "rs_plain_1",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "The user wants the weather."}],
    }
    first = resp_response(
        response_id="resp_plain_1",
        output=[reasoning, resp_message("msg_plain_1", "Let me check."), resp_function_call("fc_plain_1", "call_1")],
    )
    response, _ = await _exchange(official, _question(), resp_replies([first], stream=False), monkeypatch, stream=False)
    history = [
        *_question(),
        *response.messages,
        Message("tool", [Content.from_function_result(call_id="call_1", result="sunny")]),
    ]
    final = resp_response(response_id="resp_plain_2", output=[resp_message("msg_plain_2", "Sunny.")])

    _, to_gateway = await _exchange(gateway, history, resp_replies([final], stream=False), monkeypatch, stream=False)

    (replayed,) = [item for item in to_gateway["input"] if item.get("type") == "reasoning"]
    assert replayed["content"] == [{"type": "reasoning_text", "text": "The user wants the weather."}]
    assert REASONING_ORIGIN_KEY in response.messages[0].contents[0].additional_properties
    assert caplog.records == []


@MODES
async def test_chat_completions_reasoning_details_replay_only_to_their_endpoint_and_plaintext_anywhere(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin_wire_inputs(monkeypatch)
    router = _profile("openai", "https://openrouter.example/api/v1")
    other = _profile("openai", "https://other.example/v1")
    router_respelled = _profile("openai", "https://OPENROUTER.example:443/api/v1")
    details = [{"type": "reasoning.encrypted", "data": "enc-router", "id": "rd_1"}]
    first = cc_completion(
        response_id="chatcmpl-origin-1",
        message={"role": "assistant", "content": "Sunny.", "reasoning_details": details, "reasoning_content": "Hm."},
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    )
    if stream:
        # The kit streams plaintext reasoning only; the details arrive in a delta of their own.
        role, *rest = cc_chunks(first)
        details_delta = {"index": 0, "delta": {"reasoning_details": details}, "finish_reason": None, "logprobs": None}
        replies = (sse_reply([role, (None, {**role[1], "choices": [details_delta]}), *rest]),)
    else:
        replies = cc_replies([first], stream=False)
    response, _ = await _exchange(router, _question(), replies, monkeypatch, stream=stream)
    history = [*_question(), *response.messages, Message("user", ["And tomorrow?"])]
    final = cc_replies([cc_text("Rain.", response_id="chatcmpl-origin-2")], stream=False)

    _, to_other = await _exchange(other, history, final, monkeypatch, stream=False)
    _, to_router = await _exchange(router_respelled, history, final, monkeypatch, stream=False)

    (to_other_reply,) = [message for message in to_other["messages"] if message["role"] == "assistant"]
    (to_router_reply,) = [message for message in to_router["messages"] if message["role"] == "assistant"]
    assert "reasoning_details" not in to_other_reply
    assert to_other_reply["reasoning_content"] == "Hm."
    assert to_router_reply["reasoning_details"] == details
    assert to_router_reply["reasoning_content"] == "Hm."


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------

_HERE = ReasoningOrigin("anthropic_messages", "https://gateway.example:443")
_THERE = ReasoningOrigin("anthropic_messages", "https://api.anthropic.com:443")


def _stamped(content: Content, origin: ReasoningOrigin) -> Content:
    origin.stamp(content.additional_properties)
    return content


def test_anthropic_redacted_thinking_and_a_streamed_signature_skip_another_endpoint() -> None:
    redacted = _stamped(
        Content.from_text_reasoning(
            protected_data="opaque", additional_properties={"anthropic_redacted_thinking": True}
        ),
        _HERE,
    )
    unsigned = Content.from_text_reasoning(text="legacy thinking")
    signature = _stamped(Content.from_text_reasoning(protected_data="sig-here"), _HERE)
    history = [Message("assistant", [redacted, unsigned, signature, Content.from_text("Done.")])]

    assert anthropic_encode(history, origin=_THERE) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    ]
    assert anthropic_encode(history, origin=_HERE) == [
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "thinking", "thinking": "legacy thinking", "signature": "sig-here"},
                {"type": "text", "text": "Done."},
            ],
        }
    ]


def test_reasoning_from_before_stamps_replays_to_any_endpoint() -> None:
    anthropic = [
        Message(
            "assistant",
            [Content.from_text_reasoning(text="thinking", protected_data="sig"), Content.from_text("Done.")],
        )
    ]
    responses = [
        Message("user", ["Hi"]),
        Message(
            "assistant",
            [
                Content.from_text_reasoning(id="rs_1", text="summary", protected_data="enc"),
                Content.from_text("Done."),
            ],
        ),
    ]
    chat_completions = [
        Message(
            "assistant",
            [
                Content.from_text_reasoning(protected_data='[{"type": "reasoning.text", "text": "t"}]'),
                Content.from_text("Done."),
            ],
        )
    ]

    assert anthropic_encode(anthropic, origin=_THERE)[0]["content"][0]["signature"] == "sig"
    reasoning = [
        item
        for item in encode_input(
            responses,
            service_side=False,
            variant=OPENAI_RESPONSES,
            origin=ReasoningOrigin("openai_responses", "https://api.openai.com:443"),
        )
        if item.get("type") == "reasoning"
    ]
    assert [item["encrypted_content"] for item in reasoning] == ["enc"]
    (wire,) = chat_completions_encode(
        chat_completions, variant=OPENAI, origin=ReasoningOrigin("chat_completions", "https://other.example:443")
    )
    assert wire["reasoning_details"] == [{"type": "reasoning.text", "text": "t"}]


@pytest.mark.parametrize(
    ("fault", "level", "message"),
    [
        (None, logging.DEBUG, "Left out reasoning another endpoint issued: groups=[1]"),
        (
            "informational_call",
            logging.WARNING,
            "Degraded stateless reasoning replay: groups=[1] reasoning_ids=['rs_org'] call_ids=['call_1', 'call_2']",
        ),
        (
            "reasoning_without_payload",
            logging.WARNING,
            "Degraded stateless reasoning replay: groups=[1] reasoning_ids=['rs_org', 'rs_here'] call_ids=['call_1']",
        ),
    ],
    ids=["foreign_only", "informational_call", "reasoning_without_payload"],
)
def test_a_responses_group_degraded_for_another_reason_too_still_warns(
    fault: str | None, level: int, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.service.llm.openai_responses.replay")
    official = ReasoningOrigin("openai_responses", "https://api.openai.com:443")
    org = ReasoningOrigin("openai_responses", "https://org-a.example:443")
    contents = [_stamped(Content.from_text_reasoning(id="rs_org", text="t", protected_data="enc-org"), org)]
    if fault == "reasoning_without_payload":
        contents.append(_stamped(Content.from_text_reasoning(id="rs_here", text="u"), official))
    contents.append(Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"))
    if fault == "informational_call":
        contents.append(
            Content.from_function_call(call_id="call_2", name="web", arguments="{}", informational_only=True)
        )
    history = [
        Message("user", ["Hi"]),
        Message("assistant", contents),
        Message("tool", [Content.from_function_result(call_id="call_1", result="sunny")]),
    ]

    encoded = encode_input(history, service_side=False, variant=OPENAI_RESPONSES, origin=official)

    assert [item for item in encoded if item.get("type") == "reasoning"] == []
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [(level, message)]


def test_chat_completions_reasoning_details_kept_only_on_the_message_follow_their_endpoint() -> None:
    router = ReasoningOrigin("chat_completions", "https://openrouter.example:443")
    other = ReasoningOrigin("chat_completions", "https://other.example:443")
    properties: dict[str, Any] = {"reasoning_details": [{"type": "reasoning.encrypted", "data": "enc"}]}
    router.stamp(properties)
    history = [Message("assistant", [Content.from_text("Done.")], additional_properties=properties)]

    (to_other,) = chat_completions_encode(history, variant=OPENAI, origin=other)
    (to_router,) = chat_completions_encode(history, variant=OPENAI, origin=router)

    assert "reasoning_details" not in to_other
    assert to_router["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "enc"}]


def test_streamed_chat_completions_reasoning_details_are_stamped_and_plaintext_is_not() -> None:
    from openai.types.chat import ChatCompletionChunk

    origin = ReasoningOrigin("chat_completions", "https://openrouter.example:443")
    state = ChatCompletionsStream(OPENAI, origin=origin)
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "Hm.",
                        "reasoning_details": [{"type": "reasoning.text", "text": "t"}],
                    },
                    "finish_reason": None,
                }
            ],
        }
    )

    contents = [content for update in state.updates_for(chunk) for content in update.contents]

    stamps = {
        content.additional_properties["openai_reasoning_format"]: content.additional_properties.get(
            REASONING_ORIGIN_KEY
        )
        for content in contents
    }
    assert stamps == {"reasoning_details": origin.stamp_value(), "reasoning_content": None}
