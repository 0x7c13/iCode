# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""OpenAI's prompt cache is routed by the session a Responses request belongs to.

On OpenAI's own endpoint a Responses client sends the session id the
``X-Session-ID`` header names as ``prompt_cache_key``, unless the options set
the key themselves or turn it off with a null. Other endpoints and Chat
Completions get no key.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from typing import Any

import openai
import pytest

from chrys.foundation.util.chrys_headers import X_SESSION_ID_HEADER
from chrys.kernel import Message
from chrys.service.llm.clients import create_client
from chrys.service.llm.openai_responses.client import ResponsesApiClient
from chrys.service.llm.openai_responses.request import set_prompt_cache_key
from chrys.service.llm.route_sessions import derive_llm_route_session_id, llm_route_session_id
from chrys.service.profiles.models.options import effective_chat_options
from chrys.service.profiles.models.schema import API_STYLE_CHAT_COMPLETIONS, API_STYLE_RESPONSES, ModelProfile
from tests.support.scripted_wire import ScriptedWire, pin_wire_inputs, route_clients_to
from tests.support.wire_cases._kit import cc_replies, cc_text, resp_message, resp_replies, resp_response

_KEY = "prompt_cache_key"


def _profile(
    *, base_url: str = "", api_style: str = API_STYLE_RESPONSES, chat_options: dict[str, Any] | None = None
) -> ModelProfile:
    return ModelProfile(
        id="cache-profile",
        name="cache",
        provider="openai",
        api_style=api_style,
        model_id="cache-model",
        api_key="sk-cache",
        base_url=base_url,
        chat_options=json.dumps(chat_options) if chat_options is not None else "",
    )


def _replies(profile: ModelProfile, count: int = 1) -> tuple[Any, ...]:
    if profile.api_style == API_STYLE_CHAT_COMPLETIONS:
        return cc_replies([cc_text("Hi.", response_id=f"chatcmpl-{n}") for n in range(count)], stream=False)
    return resp_replies(
        [resp_response(response_id=f"resp_{n}", output=[resp_message(f"msg_{n}", "Hi.")]) for n in range(count)],
        stream=False,
    )


async def _sent(
    profile: ModelProfile, monkeypatch: pytest.MonkeyPatch, *, calls: int = 1, **client_kwargs: Any
) -> list[tuple[dict[str, Any], str | None]]:
    """The body and ``X-Session-ID`` of each of the *calls* requests a fresh stack for *profile* sends."""
    pin_wire_inputs(monkeypatch)
    wire = ScriptedWire(_replies(profile, count=calls))
    route_clients_to(wire.transport, monkeypatch)
    stack = await create_client(profile, **client_kwargs)
    # One options object for every call, as an agent keeps its default options.
    options = effective_chat_options(profile) or {}
    try:
        for _ in range(calls):
            await stack.inner.get_response([Message("user", ["Hi"])], options=options)
    finally:
        await stack.aclose()
    return [(json.loads(request.content), request.headers.get(X_SESSION_ID_HEADER)) for request in wire.requests]


# ---------------------------------------------------------------------------
# Which session routes the cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    ["", "https://api.openai.com/v1", "HTTPS://API.OpenAI.com:443/v1/"],
    ids=["default", "explicit", "respelled"],
)
async def test_the_main_session_routes_the_prompt_cache_on_openai(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = await _sent(_profile(base_url=base_url), monkeypatch, calls=2, session_id="main-session")

    assert [(body[_KEY], header) for body, header in sent] == [("main-session", "main-session")] * 2


async def test_a_workflow_node_routes_the_prompt_cache_by_its_own_session(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile()
    node_session = derive_llm_route_session_id(
        "parent-session",
        route_kind="workflow-node",
        route_parts=("draft", "Code", "invocation-1"),
        model_profile=profile,
    )

    ((body, header),) = await _sent(profile, monkeypatch, session_id=node_session, parent_session_id="parent-session")

    assert node_session != "parent-session"
    assert (body[_KEY], header) == (node_session, node_session)


async def test_sub_agents_sharing_a_client_route_the_prompt_cache_by_each_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin_wire_inputs(monkeypatch)
    profile = _profile()
    wire = ScriptedWire(_replies(profile, count=2))
    route_clients_to(wire.transport, monkeypatch)
    stack = await create_client(profile, session_id="shared-default", use_route_session_context=True)
    both_building = asyncio.Barrier(2)

    async def invoke(invocation_session: str) -> None:
        llm_route_session_id.set(invocation_session)
        await both_building.wait()
        await stack.inner.get_response([Message("user", ["Hi"])], options=effective_chat_options(profile) or {})

    try:
        await asyncio.gather(invoke("child-a"), invoke("child-b"))
    finally:
        await stack.aclose()

    sent = sorted(
        (json.loads(request.content)[_KEY], request.headers[X_SESSION_ID_HEADER]) for request in wire.requests
    )
    assert sent == [("child-a", "child-a"), ("child-b", "child-b")]


@pytest.mark.parametrize(
    ("base_url", "api_style"),
    [
        ("https://gateway.example/v1", API_STYLE_RESPONSES),
        ("http://api.openai.com/v1", API_STYLE_RESPONSES),
        ("", API_STYLE_CHAT_COMPLETIONS),
    ],
    ids=["other-endpoint", "other-scheme", "chat-completions"],
)
async def test_no_prompt_cache_key_off_openai_responses(
    base_url: str, api_style: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ((body, header),) = await _sent(
        _profile(base_url=base_url, api_style=api_style), monkeypatch, session_id="main-session"
    )

    assert header == "main-session"
    assert _KEY not in body


async def test_no_prompt_cache_key_without_a_session(monkeypatch: pytest.MonkeyPatch) -> None:
    ((body, _),) = await _sent(_profile(), monkeypatch)

    assert _KEY not in body
    async with openai.AsyncOpenAI(api_key="sk-cache") as sdk_client:
        client = ResponsesApiClient.from_sdk_client(sdk_client, model="cache-model")
        assert _KEY not in client._build_request([Message("user", ["Hi"])], {})


# ---------------------------------------------------------------------------
# What the options say
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chat_options", "sent"),
    [
        ({_KEY: "mine"}, "mine"),
        ({"extra_body": {_KEY: "nested"}}, "nested"),
        ({_KEY: "mine", "extra_body": {_KEY: "nested"}}, "nested"),
        ({_KEY: None}, None),
        ({"extra_body": {_KEY: None}}, None),
        ({_KEY: "mine", "extra_body": {_KEY: None}}, None),
        ({_KEY: None, "extra_body": {_KEY: "nested", "user": "u"}}, None),
    ],
    ids=["top-level", "extra-body", "extra-body-wins", "top-level-null", "extra-body-null", "null-wins", "null-beside"],
)
async def test_options_set_or_turn_off_the_prompt_cache_key(
    chat_options: dict[str, Any], sent: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = [
        body
        for body, _ in await _sent(_profile(chat_options=chat_options), monkeypatch, calls=2, session_id="main-session")
    ]

    assert [body.get(_KEY, None) for body in bodies] == [sent, sent]
    assert [_KEY in body for body in bodies] == [sent is not None] * 2


def test_a_top_level_null_rides_in_extra_body_so_the_agent_keeps_it() -> None:
    responses = effective_chat_options(_profile(chat_options={_KEY: None, "extra_body": {"user": "u"}}))
    chat_completions = effective_chat_options(_profile(api_style=API_STYLE_CHAT_COMPLETIONS, chat_options={_KEY: None}))

    assert responses is not None and _KEY not in responses
    assert responses["extra_body"] == {"user": "u", _KEY: None}
    assert chat_completions is not None and chat_completions[_KEY] is None
    assert "extra_body" not in chat_completions


@pytest.mark.parametrize(
    ("options", "session_id", "expected"),
    [
        ({}, "s" * 64, {_KEY: "s" * 64}),
        ({}, "s" * 65, {_KEY: hashlib.sha256(b"s" * 65).hexdigest()}),
        ({}, None, {}),
        ({}, "", {}),
        ({_KEY: "mine"}, "session", {_KEY: "mine"}),
        ({"extra_body": {_KEY: "nested"}}, "session", {"extra_body": {_KEY: "nested"}}),
        ({_KEY: None}, "session", {}),
        ({"extra_body": {_KEY: None, "user": "u"}}, "session", {"extra_body": {"user": "u"}}),
    ],
    ids=["fits", "hashed", "no-session", "empty-session", "top-level", "extra-body", "top-null", "extra-null"],
)
def test_set_prompt_cache_key(options: dict[str, Any], session_id: str | None, expected: dict[str, Any]) -> None:
    # Shallow, as the request a client builds shares nested option values with the options.
    request = {name: value for name, value in options.items() if value is not None}
    before = copy.deepcopy(options)

    set_prompt_cache_key(request, options, session_id=session_id)

    assert request == expected
    assert options == before
