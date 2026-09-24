# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the instrumented OpenAI Chat Completions and Responses subclasses: headers, parsing, and usage."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.util.chrys_headers import (
    MODEL_ID_HEADER,
    PARENT_SESSION_ID_HEADER,
    SESSION_ID_HEADER,
    X_PARENT_SESSION_ID_HEADER,
    X_SESSION_ID_HEADER,
)
from chrys.kernel import ChatClientException, ChatResponse, ChatResponseUpdate, FunctionTool, Message
from chrys.service.llm.instrumented import _set_chrys_request_headers
from chrys.service.llm.route_sessions import llm_parent_session_id, llm_route_session_id
from tests.service.llm._instrumented_clients import make_chat_client, make_responses_chat_client


def test_openai_parse_response_accepts_millisecond_created_timestamp() -> None:
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from chrys.service.llm.openai_timestamps import openai_created_at_iso

    created_ms = 1_717_171_717_123
    chat_client = make_chat_client()
    response = ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=created_ms,
        model="gpt-test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
    )

    parsed = chat_client._parse_response_from_openai(response, {})

    assert type(parsed) is ChatResponse
    assert parsed.created_at == openai_created_at_iso(created_ms)
    assert response.created == created_ms


def test_openai_parse_response_update_accepts_millisecond_created_timestamp() -> None:
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta as ChunkChoiceDelta

    from chrys.service.llm.openai_timestamps import openai_created_at_iso

    created_ms = 1_717_171_717_123
    chat_client = make_chat_client()
    chunk = ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=created_ms,
        model="gpt-test",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkChoiceDelta(role="assistant", content="hi"),
                finish_reason=None,
            )
        ],
    )

    parsed = chat_client._parse_response_update_from_openai(chunk)

    assert type(parsed) is ChatResponseUpdate
    assert parsed.created_at == openai_created_at_iso(created_ms)
    assert chunk.created == created_ms


def test_openai_prepare_options_sets_model_header_from_effective_model() -> None:
    chat_client = make_chat_client()

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "model": "gpt-final",
            "extra_headers": {
                "X-Team": "platform",
                "chrys-debug": "drop-me",
                MODEL_ID_HEADER: "wrong",
            },
        },
    )

    assert prepared["model"] == "gpt-final"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "gpt-final"
    assert prepared["extra_headers"]["X-Team"] == "platform"
    assert "chrys-debug" not in prepared["extra_headers"]


def test_openai_prepare_options_sets_model_header_from_default_model() -> None:
    chat_client = make_chat_client()

    prepared = chat_client._prepare_options([Message("user", ["hi"])], {})

    assert prepared["model"] == "gpt-test"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "gpt-test"


def test_openai_prepare_options_sets_session_headers_from_session_id() -> None:
    chat_client = make_chat_client(session_id="sess-123", parent_session_id="parent-123")

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "extra_headers": {
                X_SESSION_ID_HEADER: "wrong",
                X_PARENT_SESSION_ID_HEADER: "wrong-parent",
                "X-Session-Id": "wrong-mixed-case",
                SESSION_ID_HEADER: "wrong",
                PARENT_SESSION_ID_HEADER: "wrong-parent",
            },
        },
    )

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "parent-123"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "parent-123"
    assert "X-Session-Id" not in prepared["extra_headers"]


def test_openai_prepare_options_prefers_context_route_session_headers() -> None:
    chat_client = make_chat_client(
        session_id="default-session",
        parent_session_id="default-parent",
        use_route_session_context=True,
    )
    session_token = llm_route_session_id.set("invocation-session")
    parent_token = llm_parent_session_id.set("root-session")
    try:
        prepared = chat_client._prepare_options([Message("user", ["hi"])], {})
    finally:
        llm_parent_session_id.reset(parent_token)
        llm_route_session_id.reset(session_token)

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "invocation-session"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "invocation-session"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "root-session"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "root-session"


def test_openai_prepare_options_ignores_context_route_session_headers_by_default() -> None:
    chat_client = make_chat_client(session_id="default-session", parent_session_id="default-parent")
    session_token = llm_route_session_id.set("invocation-session")
    parent_token = llm_parent_session_id.set("root-session")
    try:
        prepared = chat_client._prepare_options([Message("user", ["hi"])], {})
    finally:
        llm_parent_session_id.reset(parent_token)
        llm_route_session_id.reset(session_token)

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "default-session"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "default-session"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "default-parent"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "default-parent"


def test_openai_prepare_options_rejects_non_ascii_extra_header_value() -> None:
    """Resolved chat_options.extra_headers hit the wire-charset gate at request time."""
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": "秘密token"}},
        )

    message = str(info.value)
    assert "'X-Test'" in message
    assert "position 1" in message
    assert "秘密" not in message


def test_openai_prepare_options_rejects_non_ascii_model_override() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options([Message("user", ["hi"])], {"model": "模型"})

    message = str(info.value)
    assert "Model ID" in message
    assert "U+6A21" in message


def test_openai_prepare_options_rejects_outer_space_header_value() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": "token "}},
        )

    message = str(info.value)
    assert "'X-Test'" in message
    assert "ends with a space" in message
    assert "token" not in message


def test_openai_prepare_options_rejects_non_string_extra_header_value() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": ["secret-value"]}},
        )

    message = str(info.value)
    assert "Header 'X-Test' value must be a string" in message
    assert "secret-value" not in message


def test_openai_prepare_options_rejects_non_string_extra_header_name() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError, match="Header name at position 1 must be a string"):
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {123: "value"}},
        )


def test_openai_prepare_options_allows_dropped_managed_header_with_unsafe_value() -> None:
    """A Chrys-managed header never reaches the wire, so its value is not validated."""
    chat_client = make_chat_client(session_id="sess-123")

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {"extra_headers": {"chrys-debug": "值"}},
    )

    assert "chrys-debug" not in prepared["extra_headers"]


def test_set_chrys_request_headers_validates_on_early_return_path() -> None:
    """Even with no Chrys metadata to merge, caller headers still get the gate."""
    options: dict[str, Any] = {"extra_headers": {"X-Test": "值"}}

    with pytest.raises(ValueError) as info:
        _set_chrys_request_headers(options, session_id=None)

    message = str(info.value)
    assert "'X-Test'" in message
    assert "值" not in message


async def test_openai_responses_prepare_options_sets_chrys_headers() -> None:
    chat_client = make_responses_chat_client(session_id="sess-123", parent_session_id="parent-123")

    prepared = await chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "model": "gpt-final",
            "extra_headers": {
                "X-Team": "platform",
                MODEL_ID_HEADER: "wrong",
                SESSION_ID_HEADER: "wrong",
                PARENT_SESSION_ID_HEADER: "wrong-parent",
                X_SESSION_ID_HEADER: "wrong",
                X_PARENT_SESSION_ID_HEADER: "wrong-parent",
            },
        },
    )

    assert prepared["model"] == "gpt-final"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "gpt-final"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "parent-123"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "parent-123"
    assert prepared["extra_headers"]["X-Team"] == "platform"


@pytest.mark.parametrize("mode", ["auto", "required"])
async def test_openai_responses_preserves_allowed_tools_mode(mode: str) -> None:
    chat_client = make_responses_chat_client()
    search_tool = FunctionTool(func=lambda query: query, name="search_docs", description="Search documentation")

    prepared = await chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "tools": [search_tool],
            "tool_choice": {"mode": mode, "allowed_tools": ["search_docs"]},
        },
    )

    assert prepared["tool_choice"] == {
        "type": "allowed_tools",
        "mode": mode,
        "tools": [{"type": "function", "name": "search_docs"}],
    }


async def test_openai_responses_required_without_allowlist_stays_plain_required() -> None:
    chat_client = make_responses_chat_client()
    search_tool = FunctionTool(func=lambda query: query, name="search_docs", description="Search documentation")

    prepared = await chat_client._prepare_options(
        [Message("user", ["hi"])],
        {"tools": [search_tool], "tool_choice": {"mode": "required"}},
    )

    assert prepared["tool_choice"] == "required"


# ──────────────── integration: instrumented OpenAI subclass ─────────────
#
# These tests build the actual ``_InstrumentedOpenAIChatCompletionClient``
# subclass via the factory and feed it real ``openai.types`` ``ChatCompletion``
# objects, verifying both that the override is wired in and that valid
# responses still flow through to the base parser.


def test_integration_choices_none_raises_with_gateway_error_body() -> None:
    """A 200 response carrying a gateway error envelope surfaces as ChatClientException."""
    from openai.types.chat.chat_completion import ChatCompletion

    chat_client = make_chat_client()
    bad = ChatCompletion.model_construct(
        id="resp-bad",
        choices=None,
        created=0,
        model="gpt-test",
        object="chat.completion",
        error={"message": "rate limit exceeded", "code": 429},
    )

    with pytest.raises(ChatClientException) as exc_info:
        chat_client._parse_response_from_openai(bad, {})

    msg = str(exc_info.value)
    assert "missing the required 'choices' array" in msg
    assert "rate limit exceeded" in msg
    assert "429" in msg


def test_integration_valid_response_delegates_to_super() -> None:
    """A valid (empty-choices) response should pass through and produce a ChatResponse."""
    from openai.types.chat.chat_completion import ChatCompletion

    chat_client = make_chat_client()
    valid = ChatCompletion.model_construct(
        id="resp-ok",
        choices=[],
        created=0,
        model="gpt-test",
        object="chat.completion",
        usage=None,
    )

    result = chat_client._parse_response_from_openai(valid, {})
    assert result.response_id == "resp-ok"
    assert result.messages == []


def test_integration_class_uses_overridden_method() -> None:
    """Sanity check that the subclass — not the parent — defines the active method."""
    chat_client = make_chat_client()
    cls = type(chat_client.inner.inner)
    assert "_parse_response_from_openai" in cls.__dict__
    assert cls.__name__ == "_InstrumentedOpenAIChatCompletionClient"


# ---------------------------------------------------------------------------
# Cache-token preservation
# ---------------------------------------------------------------------------


def test_parse_usage_preserves_cached_tokens_zero() -> None:
    """OpenAI ``cached_tokens=0`` must survive parsing so the UI shows ``0``,
    not ``-``. A naive ``if tokens := ...:`` truthiness check would drop it."""
    from openai.types.completion_usage import CompletionUsage

    chat_client = make_chat_client()
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert details["prompt/cached_tokens"] == 0
    assert details["cache_read_input_token_count"] == 0
    assert details["completion/reasoning_tokens"] == 0
    assert details["reasoning_output_token_count"] == 0


def test_responses_parse_usage_preserves_cached_tokens_zero() -> None:
    """Responses ``cached_tokens=0`` must survive parsing."""
    from openai.types.responses import ResponseUsage

    chat_client = make_responses_chat_client()
    usage = ResponseUsage.model_validate(
        {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1050,
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert details["openai.cached_input_tokens"] == 0
    assert details["cache_read_input_token_count"] == 0
    assert details["openai.reasoning_tokens"] == 0
    assert details["reasoning_output_token_count"] == 0


def test_parse_usage_preserves_cached_tokens_nonzero() -> None:
    """Override must not overwrite a non-zero value the base parser already set."""
    from openai.types.completion_usage import CompletionUsage

    chat_client = make_chat_client()
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 256},
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert details["prompt/cached_tokens"] == 256
    assert details["cache_read_input_token_count"] == 256


def test_parse_usage_omits_cache_key_when_provider_does_not_report() -> None:
    """Absent ``prompt_tokens_details`` must stay absent — ``None`` semantics."""
    from openai.types.completion_usage import CompletionUsage

    chat_client = make_chat_client()
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert "prompt/cached_tokens" not in details
    assert "cache_read_input_token_count" not in details
