# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What an Anthropic request sends about its thinking, and its one ``anthropic-beta`` header.

Every case runs the production client stack over the real SDK and reads the
request the transport received.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest

from chrys.kernel import Message
from chrys.service.llm.anthropic_messages.thinking_binding import resolve_thinking_binding
from chrys.service.llm.clients import create_client
from chrys.service.profiles.models.options import THINKING_BLOCK_BINDING_OPTION, effective_chat_options
from chrys.service.profiles.models.schema import ModelProfile, ThinkingBlockBinding
from tests.support.scripted_wire import ScriptedWire, pin_wire_inputs, route_clients_to
from tests.support.wire_cases._kit import anth_events, anth_message, anth_replies, anth_text, json_reply, sse_reply

_DEFAULTS = "mcp-client-2025-04-04,code-execution-2025-08-25"
_CONTROLS = "thinking-binding-controls-2026-08-01"
_INTERLEAVED = "interleaved-thinking-2025-05-14"
_ADAPTIVE = {"type": "adaptive"}
_ENABLED = {"type": "enabled", "budget_tokens": 1024}
_ABSENT = object()


def _profile(
    *,
    model_id: str = "claude-opus-5-5",
    base_url: str = "",
    chat_options: dict[str, Any] | None = None,
    http_headers: dict[str, str] | None = None,
    binding: ThinkingBlockBinding = "auto",
    interleaved: bool = True,
) -> ModelProfile:
    return ModelProfile(
        id="claude-profile",
        name="claude",
        provider="anthropic",
        model_id=model_id,
        api_key="sk-ant-test",
        base_url=base_url,
        chat_options=json.dumps(chat_options) if chat_options is not None else "",
        http_headers=json.dumps(http_headers) if http_headers is not None else "",
        thinking_block_binding=binding,
        auto_interleaved_thinking=interleaved,
    )


def _with_thinking(thinking: object, **chat_options: Any) -> dict[str, Any]:
    return chat_options if thinking is _ABSENT else {"thinking": thinking, **chat_options}


async def _sent(
    profile: ModelProfile,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: bool = False,
    client_kwargs: dict[str, Any] | None = None,
    replies: tuple[Any, ...] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """The body and the ``anthropic-beta`` header lines of the one request *profile*'s stack sends."""
    pin_wire_inputs(monkeypatch)
    wire = ScriptedWire(replies or anth_replies([anth_text("Hi.", message_id="msg_1")], stream=stream))
    route_clients_to(wire.transport, monkeypatch)
    stack = await create_client(profile)
    options = effective_chat_options(profile) or {}
    before = copy.deepcopy(options)
    messages = [Message("user", ["Hi"])]
    try:
        if stream:
            updates = stack.inner.get_response(messages, options=options, stream=True, client_kwargs=client_kwargs)
            [_ async for _ in updates]
        else:
            await stack.inner.get_response(messages, options=options, client_kwargs=client_kwargs)
    finally:
        await stack.aclose()
    # Nothing the request needed was written into the options or what they hold.
    assert options == before
    [request] = wire.requests
    return json.loads(request.content), request.headers.get_list("anthropic-beta")


def _binding(behavior: str) -> dict[str, Any]:
    return {"block_binding": {"prefix_mismatch_behavior": behavior}}


# ---------------------------------------------------------------------------
# The block binding a request sends
# ---------------------------------------------------------------------------

_THINKING_FORMS: dict[str, object] = {
    "absent": _ABSENT,
    "null": None,
    "disabled": {"type": "disabled"},
    "enabled": _ENABLED,
    "adaptive": _ADAPTIVE,
    "between_tools": {"type": "between_tools"},
}
_SETTINGS: tuple[ThinkingBlockBinding, ...] = ("auto", "drop_block", "error", "off")
# The behavior each setting, in _SETTINGS order, adds to the thinking.
_ADDED: dict[str, tuple[str | None, ...]] = {
    "absent": (None, None, None, None),
    "null": (None, None, None, None),
    "disabled": (None, None, None, None),
    "enabled": (None, "drop_block", "error", None),
    "adaptive": ("drop_block", "drop_block", "error", None),
    "between_tools": (None, None, None, None),
}


@pytest.mark.parametrize("setting", _SETTINGS)
@pytest.mark.parametrize("form", list(_THINKING_FORMS))
async def test_each_setting_binds_only_enabled_or_adaptive_thinking(
    form: str, setting: ThinkingBlockBinding, monkeypatch: pytest.MonkeyPatch
) -> None:
    thinking = _THINKING_FORMS[form]
    added = _ADDED[form][_SETTINGS.index(setting)]

    body, betas = await _sent(_profile(chat_options=_with_thinking(thinking), binding=setting), monkeypatch)

    if not isinstance(thinking, dict):
        assert "thinking" not in body
    else:
        assert body["thinking"] == ({**thinking, **_binding(added)} if added else thinking)
    expected = [_DEFAULTS, *([_CONTROLS] if added else []), *([_INTERLEAVED] if form == "enabled" else [])]
    assert betas == [",".join(expected)]


@pytest.mark.parametrize(
    ("model_id", "base_url", "added"),
    [
        ("claude-opus-5-5", "", True),
        ("claude-fable-5-1", "", True),
        ("claude-sonnet-5-5", "", True),
        ("claude-opus-5-5", "https://API.Anthropic.com", True),
        ("claude-opus-5-5", "https://api.anthropic.com:443/", True),
        ("claude-opus-4-7", "", False),
        ("claude-opus-5", "", False),
        ("anthropic.claude-opus-5-5-v1:0", "", False),
        ("claude-opus-5-5", "https://api.anthropic.com:444", False),
        ("claude-opus-5-5", "http://api.anthropic.com", False),
        ("claude-opus-5-5", "https://gateway.example/anthropic", False),
    ],
    ids=[
        "opus",
        "fable",
        "sonnet",
        "uppercase-host",
        "explicit-port",
        "older-model",
        "alias",
        "cloud-id",
        "other-port",
        "other-scheme",
        "gateway",
    ],
)
async def test_auto_drops_mismatched_blocks_only_for_binding_models_on_anthropic(
    model_id: str, base_url: str, added: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(model_id=model_id, base_url=base_url, chat_options={"thinking": _ADAPTIVE})

    body, betas = await _sent(profile, monkeypatch)

    assert body["thinking"] == ({**_ADAPTIVE, **_binding("drop_block")} if added else _ADAPTIVE)
    assert betas == [f"{_DEFAULTS},{_CONTROLS}" if added else _DEFAULTS]


@pytest.mark.parametrize("setting", ["drop_block", "error"])
async def test_an_explicit_setting_binds_on_any_endpoint_and_model(
    setting: ThinkingBlockBinding, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(
        model_id="claude-opus-4-7",
        base_url="https://gateway.example/anthropic",
        chat_options={"thinking": _ADAPTIVE},
        binding=setting,
    )

    body, betas = await _sent(profile, monkeypatch)

    assert body["thinking"] == {**_ADAPTIVE, **_binding(setting)}
    assert betas == [f"{_DEFAULTS},{_CONTROLS}"]


@pytest.mark.parametrize(
    ("setting", "thinking"),
    [
        ("auto", {**_ADAPTIVE, **_binding("error")}),
        ("drop_block", {**_ADAPTIVE, **_binding("error")}),
        ("off", {**_ADAPTIVE, **_binding("drop_block")}),
        ("auto", {**_ADAPTIVE, "block_binding": {}}),
    ],
    ids=["auto", "other-setting", "off", "no-behavior"],
)
async def test_a_block_binding_the_options_write_is_sent_as_written_with_the_controls_beta(
    setting: ThinkingBlockBinding, thinking: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    body, betas = await _sent(_profile(chat_options={"thinking": thinking}, binding=setting), monkeypatch)

    assert body["thinking"] == thinking
    assert betas == [f"{_DEFAULTS},{_CONTROLS}"]


async def test_a_block_binding_that_is_no_mapping_is_sent_as_written_without_the_controls_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinking = {**_ADAPTIVE, "block_binding": "drop_block"}

    body, betas = await _sent(_profile(chat_options={"thinking": thinking}), monkeypatch)

    assert body["thinking"] == thinking
    assert betas == [_DEFAULTS]


@pytest.mark.parametrize(
    ("setting", "request_fields", "behavior", "sent"),
    [
        ("auto", {"thinking": _ADAPTIVE}, "drop_block", True),
        ("auto", {"thinking": _ENABLED}, None, False),
        ("auto", {}, None, False),
        ("drop_block", {}, "drop_block", False),
        ("error", {"thinking": {"type": "disabled"}}, "error", False),
        ("off", {"thinking": _ADAPTIVE}, None, False),
        ("off", {"thinking": {**_ADAPTIVE, **_binding("error")}}, "error", True),
        ("drop_block", {"thinking": {**_ADAPTIVE, "block_binding": {}}}, "drop_block", False),
        ("auto", {"thinking": {**_ADAPTIVE, **_binding("warn")}}, "warn", True),
        (
            "error",
            {"thinking": _ADAPTIVE, "extra_body": {"thinking": {**_ADAPTIVE, **_binding("drop_block")}}},
            "drop_block",
            True,
        ),
    ],
    ids=[
        "auto-adds",
        "auto-leaves-enabled",
        "auto-no-thinking",
        "setting-without-thinking",
        "error-without-binding",
        "off",
        "written",
        "written-without-behavior",
        "written-unknown",
        "written-in-extra-body",
    ],
)
def test_the_mismatch_behavior_a_request_asks_for(
    setting: ThinkingBlockBinding, request_fields: dict[str, Any], behavior: str | None, sent: bool
) -> None:
    request = {"model": "claude-opus-5-5", **request_fields}
    options = {THINKING_BLOCK_BINDING_OPTION: setting}
    before = copy.deepcopy((request, options))

    policy = resolve_thinking_binding(request, options, base_url="https://api.anthropic.com")

    assert policy.mismatch_behavior == behavior
    assert policy.mismatch_behavior_sent is sent
    assert (request, options) == before


@pytest.mark.parametrize(
    ("request_fields", "binding_model"),
    [
        ({"model": "claude-fable-5-1"}, True),
        ({"model": "claude-opus-4-7"}, False),
        ({"model": "claude-opus-4-7", "extra_body": {"model": "claude-sonnet-5-5"}}, True),
        ({"model": "claude-opus-5-5", "extra_body": {"model": None}}, False),
        ({"model": ["claude-opus-5-5"]}, False),
    ],
    ids=["binding", "older", "extra-body", "extra-body-null", "not-a-string"],
)
def test_whether_the_final_model_binds_its_thinking(request_fields: dict[str, Any], binding_model: bool) -> None:
    policy = resolve_thinking_binding(request_fields, {}, base_url="https://gateway.example")

    assert policy.binding_model is binding_model


@pytest.mark.parametrize(
    ("request_fields", "thinking_type"),
    [
        ({"thinking": {"type": "adaptive"}}, "adaptive"),
        ({"thinking": {"type": "adaptive"}, "extra_body": {"thinking": {"type": "disabled"}}}, "disabled"),
        ({"thinking": {"type": "adaptive"}, "extra_body": {"thinking": None}}, None),
        ({}, None),
    ],
    ids=["request", "extra-body", "extra-body-null", "none"],
)
def test_the_type_of_the_final_thinking(request_fields: dict[str, Any], thinking_type: str | None) -> None:
    policy = resolve_thinking_binding(
        {"model": "claude-opus-5-5", **request_fields}, {}, base_url="https://gateway.example"
    )

    assert policy.thinking_type == thinking_type


@pytest.mark.parametrize(
    ("request_fields", "setting", "written"),
    [
        ({"thinking": {"type": "adaptive", "block_binding": {}}}, "drop_block", True),
        ({"thinking": {"type": "adaptive", "block_binding": "drop_block"}}, "drop_block", True),
        ({"extra_body": {"thinking": {"type": "adaptive", "block_binding": {}}}}, "auto", True),
        ({"thinking": {"type": "adaptive", "block_binding": {}}, "extra_body": {"thinking": _ADAPTIVE}}, "auto", False),
        ({"thinking": {"type": "adaptive"}}, "drop_block", False),
        ({}, "auto", False),
    ],
    ids=["mapping", "not-a-mapping", "extra-body", "overridden-by-extra-body", "added-by-setting", "no-thinking"],
)
def test_whether_the_request_writes_its_own_block_binding(
    request_fields: dict[str, Any], setting: ThinkingBlockBinding, written: bool
) -> None:
    policy = resolve_thinking_binding(
        {"model": "claude-opus-5-5", **request_fields},
        {THINKING_BLOCK_BINDING_OPTION: setting},
        base_url="https://api.anthropic.com",
    )

    assert policy.block_binding_written is written


# ---------------------------------------------------------------------------
# Which thinking and model the request ends up with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile_model", "chat_options", "client_kwargs", "thinking", "added"),
    [
        ("claude-opus-5-5", {"thinking": _ENABLED, "extra_body": {"thinking": _ADAPTIVE}}, None, _ADAPTIVE, True),
        ("claude-opus-5-5", {"thinking": _ADAPTIVE, "extra_body": {"thinking": None}}, None, None, False),
        (
            "claude-opus-5-5",
            {"thinking": _ADAPTIVE, "extra_body": {"model": "claude-opus-4-7"}},
            None,
            _ADAPTIVE,
            False,
        ),
        ("claude-opus-5-5", {"thinking": _ADAPTIVE, "model": "claude-opus-4-7"}, None, _ADAPTIVE, False),
        ("claude-opus-4-7", {"thinking": _ADAPTIVE, "model": "claude-opus-5-5"}, None, _ADAPTIVE, True),
        ("claude-opus-5-5", {"thinking": _ENABLED}, {"thinking": _ADAPTIVE}, _ADAPTIVE, True),
        ("claude-opus-5-5", {"thinking": _ADAPTIVE}, {"thinking": _ENABLED}, _ENABLED, False),
    ],
    ids=[
        "extra-body-thinking",
        "extra-body-null",
        "extra-body-model",
        "model-option",
        "model-option-binds",
        "call-thinking",
        "call-thinking-enabled",
    ],
)
async def test_the_binding_follows_the_thinking_and_model_the_service_gets(
    profile_model: str,
    chat_options: dict[str, Any],
    client_kwargs: dict[str, Any] | None,
    thinking: dict[str, Any] | None,
    added: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(model_id=profile_model, chat_options=chat_options)

    body, betas = await _sent(profile, monkeypatch, client_kwargs=client_kwargs)

    assert body["thinking"] == ({**thinking, **_binding("drop_block")} if added and thinking else thinking)
    assert (_CONTROLS in betas[0].split(",")) is added


async def test_a_binding_for_extra_body_thinking_leaves_the_top_level_thinking_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin_wire_inputs(monkeypatch)
    wire = ScriptedWire(anth_replies([anth_text("Hi.", message_id="msg_1")], stream=False))
    route_clients_to(wire.transport, monkeypatch)
    profile = _profile(chat_options={"thinking": _ENABLED, "extra_body": {"thinking": _ADAPTIVE, "top_k": 5}})
    stack = await create_client(profile)
    try:
        request = stack.inner._build_request([Message("user", ["Hi"])], effective_chat_options(profile) or {}, {})
    finally:
        await stack.aclose()

    assert request["thinking"] == _ENABLED
    assert request["extra_body"] == {"thinking": {**_ADAPTIVE, **_binding("drop_block")}, "top_k": 5}


# ---------------------------------------------------------------------------
# The interleaved-thinking beta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "base_url", "thinking", "switch", "sent"),
    [
        ("claude-sonnet-4-5", "", _ENABLED, True, True),
        ("claude-future-model", "", _ENABLED, True, True),
        ("claude-sonnet-4-5", "", _ENABLED, False, False),
        ("claude-sonnet-4-5", "https://gateway.example", _ENABLED, True, False),
        ("claude-sonnet-4-5", "", _ADAPTIVE, True, False),
        ("claude-sonnet-4-5", "", _ABSENT, True, False),
        ("claude-haiku-4-5", "", _ENABLED, True, False),
        ("claude-haiku-4-5-20251001", "", _ENABLED, True, False),
        pytest.param(
            "claude-opus-4-6",
            "",
            _ENABLED,
            True,
            False,
            # The SDK warns that this model's budgeted thinking is deprecated.
            marks=pytest.mark.filterwarnings("ignore:Using Claude with claude-opus-4-6:UserWarning"),
        ),
    ],
    ids=[
        "budgeted",
        "unknown-model",
        "turned-off",
        "gateway",
        "adaptive",
        "no-thinking",
        "haiku",
        "haiku-dated",
        "opus-4-6",
    ],
)
async def test_budgeted_thinking_on_anthropic_gets_the_interleaved_beta(
    model_id: str, base_url: str, thinking: object, switch: bool, sent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(model_id=model_id, base_url=base_url, chat_options=_with_thinking(thinking), interleaved=switch)

    _, betas = await _sent(profile, monkeypatch)

    assert betas == [f"{_DEFAULTS},{_INTERLEAVED}" if sent else _DEFAULTS]


@pytest.mark.parametrize("switch", [True, False], ids=["automatic", "turned-off"])
async def test_an_interleaved_beta_the_options_name_is_kept_once(switch: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile(
        model_id="claude-sonnet-4-5",
        chat_options={"thinking": _ENABLED, "additional_beta_flags": [_INTERLEAVED]},
        interleaved=switch,
    )

    _, betas = await _sent(profile, monkeypatch)

    assert betas == [f"{_DEFAULTS},{_INTERLEAVED}"]


# ---------------------------------------------------------------------------
# The one anthropic-beta header
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
@pytest.mark.parametrize("header_name", ["anthropic-beta", "Anthropic-Beta"])
async def test_every_beta_source_joins_one_header(
    header_name: str, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(
        model_id="claude-test",
        chat_options={"additional_beta_flags": ["a-one", "dup"], "betas": " b-one, dup ,, "},
        http_headers={header_name: "p-one, a-one"},
    )

    body, betas = await _sent(profile, monkeypatch, stream=stream)

    assert betas == [f"{_DEFAULTS},a-one,dup,b-one,p-one"]
    assert "betas" not in body
    assert "additional_beta_flags" not in body


async def test_comma_separated_beta_strings_are_split_into_betas(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile(
        model_id="claude-test", chat_options={"additional_beta_flags": "f-one,f-two", "betas": "b-one,b-two"}
    )

    _, betas = await _sent(profile, monkeypatch)

    assert betas == [f"{_DEFAULTS},f-one,f-two,b-one,b-two"]


@pytest.mark.parametrize(
    "extra_headers",
    [
        {"x-note": 5},
        {"anthropic-beta": 5},
        {"anthropic-beta": None},
        {"Anthropic-Beta": ["a-one", "a-two"]},
        {"anthropic-beta": "a-one", "Anthropic-Beta": None},
    ],
    ids=["other-header", "beta-number", "beta-null", "beta-list", "beta-null-beside-a-string"],
)
async def test_a_header_value_http_cannot_carry_is_still_refused(
    extra_headers: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    pin_wire_inputs(monkeypatch)
    wire = ScriptedWire(())
    route_clients_to(wire.transport, monkeypatch)
    profile = _profile(model_id="claude-test", chat_options={"extra_headers": extra_headers})
    stack = await create_client(profile)
    try:
        with pytest.raises(ValueError, match="cannot be sent over HTTP"):
            await stack.inner.get_response([Message("user", ["Hi"])], options=effective_chat_options(profile) or {})
    finally:
        await stack.aclose()

    assert wire.requests == []


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
@pytest.mark.parametrize("header_name", ["anthropic-beta", "Anthropic-Beta"])
async def test_an_anthropic_beta_header_the_options_name_replaces_the_others(
    header_name: str, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(
        model_id="claude-test",
        chat_options={
            "extra_headers": {header_name: "only-mine"},
            "additional_beta_flags": ["a-one"],
            "betas": ["b-one"],
        },
        http_headers={"anthropic-beta": "p-one", "Anthropic-Beta": "p-two"},
    )

    _, betas = await _sent(profile, monkeypatch, stream=stream)

    assert betas == ["only-mine"]


@pytest.mark.parametrize(
    ("model_id", "thinking", "binding", "interleaved", "sent"),
    [
        ("claude-opus-5-5", _ADAPTIVE, "auto", True, f"only-mine,{_CONTROLS}"),
        ("claude-sonnet-4-5", _ENABLED, "auto", True, f"only-mine,{_INTERLEAVED}"),
        ("claude-opus-5-5", _ADAPTIVE, "off", True, "only-mine"),
        ("claude-sonnet-4-5", _ENABLED, "auto", False, "only-mine"),
        ("claude-sonnet-4-5", {**_ENABLED, **_binding("error")}, "off", False, f"only-mine,{_CONTROLS}"),
    ],
    ids=["controls", "interleaved", "binding-off", "interleaved-off", "written-binding"],
)
async def test_the_betas_the_thinking_needs_follow_an_anthropic_beta_header_the_options_name(
    model_id: str,
    thinking: dict[str, Any],
    binding: ThinkingBlockBinding,
    interleaved: bool,
    sent: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(
        model_id=model_id,
        chat_options={"thinking": thinking, "extra_headers": {"anthropic-beta": "only-mine"}},
        binding=binding,
        interleaved=interleaved,
    )

    body, betas = await _sent(profile, monkeypatch)

    assert betas == [sent]
    if binding == "off":
        assert body["thinking"] == thinking


@pytest.mark.parametrize(
    "http_headers", [None, {"anthropic-beta": "p-one", "Anthropic-Beta": "p-two"}], ids=["no-defaults", "defaults"]
)
@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_an_empty_anthropic_beta_header_the_options_name_sends_none(
    stream: bool, http_headers: dict[str, str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(
        model_id="claude-test",
        chat_options={"extra_headers": {"Anthropic-Beta": ""}, "betas": ["b-one"]},
        http_headers=http_headers,
    )

    _, betas = await _sent(profile, monkeypatch, stream=stream)

    assert betas == []


# ---------------------------------------------------------------------------
# What the service reports it changed
# ---------------------------------------------------------------------------

_DROPPED = {"type": "thinking_dropped", "reason": "prefix_binding_mismatch", "path": "messages.1.content.0"}
_LOGGED = "The service changed 1 block(s) of the request: [('thinking_dropped', 'prefix_binding_mismatch', 'messages.1.content.0')]"


def _thinking_reply_message() -> dict[str, Any]:
    return anth_message(
        message_id="msg_1",
        content=[
            {"type": "thinking", "thinking": "secret thought", "signature": "sig-secret"},
            {"type": "text", "text": "Hi."},
        ],
    )


@pytest.mark.parametrize("where", ["send", "message_start", "message_delta"])
async def test_input_transformations_are_logged_without_content(
    where: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    message = _thinking_reply_message()
    if where == "send":
        replies: tuple[Any, ...] = (json_reply({**message, "input_transformations": [_DROPPED]}),)
    else:
        events = anth_events({**message, "input_transformations": [_DROPPED] if where == "message_start" else []})
        if where == "message_delta":
            events = [
                (name, {**payload, "input_transformations": [_DROPPED]} if name == "message_delta" else payload)
                for name, payload in events
            ]
        replies = (sse_reply(events),)

    with caplog.at_level(logging.DEBUG, logger="chrys.service.llm.anthropic_messages"):
        await _sent(_profile(), monkeypatch, stream=where != "send", replies=replies)

    logged = [record.getMessage() for record in caplog.records if record.getMessage().startswith("The service changed")]
    assert logged == [_LOGGED]
    assert "sig-secret" not in caplog.text
    assert "secret thought" not in caplog.text
