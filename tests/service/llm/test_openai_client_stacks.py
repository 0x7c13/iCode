# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the production OpenAI Chat Completions and Responses client stacks: headers, parsing, and usage."""

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
from chrys.kernel.exceptions import ChatClientInvalidResponseException
from chrys.service.llm.chat_completions.client import OPENAI
from chrys.service.llm.chat_completions.decode import decode_completion
from chrys.service.llm.route_sessions import llm_parent_session_id, llm_route_session_id
from chrys.service.llm.wire_client import RequestHeaders
from tests.service.llm._wire_stacks import make_chat_client, make_responses_chat_client
from tests.support.openai_chat_wire import parse_stream_chunks


def test_openai_decode_completion_accepts_millisecond_created_timestamp() -> None:
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from chrys.service.llm.openai_timestamps import openai_created_at_iso

    created_ms = 1_717_171_717_123
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

    parsed = decode_completion(response, {}, variant=OPENAI)

    assert type(parsed) is ChatResponse
    assert parsed.created_at == openai_created_at_iso(created_ms)
    assert response.created == created_ms


def test_openai_stream_update_accepts_millisecond_created_timestamp() -> None:
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

    (parsed,) = parse_stream_chunks(chat_client, chunk)

    assert type(parsed) is ChatResponseUpdate
    assert parsed.created_at == openai_created_at_iso(created_ms)
    assert chunk.created == created_ms
    # The update keeps the seconds-normalized copy as its raw representation.
    assert isinstance(parsed.raw_representation, ChatCompletionChunk)
    assert parsed.raw_representation.created == created_ms / 1000


def test_openai_build_request_sets_model_header_from_effective_model() -> None:
    chat_client = make_chat_client()

    prepared = chat_client._build_request(
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


def test_openai_build_request_sets_model_header_from_default_model() -> None:
    chat_client = make_chat_client()

    prepared = chat_client._build_request([Message("user", ["hi"])], {})

    assert prepared["model"] == "gpt-test"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "gpt-test"


def test_openai_build_request_sets_session_headers_from_session_id() -> None:
    chat_client = make_chat_client(session_id="sess-123", parent_session_id="parent-123")

    prepared = chat_client._build_request(
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


def test_openai_build_request_prefers_context_route_session_headers() -> None:
    chat_client = make_chat_client(
        session_id="default-session",
        parent_session_id="default-parent",
        use_route_session_context=True,
    )
    session_token = llm_route_session_id.set("invocation-session")
    parent_token = llm_parent_session_id.set("root-session")
    try:
        prepared = chat_client._build_request([Message("user", ["hi"])], {})
    finally:
        llm_parent_session_id.reset(parent_token)
        llm_route_session_id.reset(session_token)

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "invocation-session"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "invocation-session"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "root-session"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "root-session"


def test_openai_build_request_ignores_context_route_session_headers_by_default() -> None:
    chat_client = make_chat_client(session_id="default-session", parent_session_id="default-parent")
    session_token = llm_route_session_id.set("invocation-session")
    parent_token = llm_parent_session_id.set("root-session")
    try:
        prepared = chat_client._build_request([Message("user", ["hi"])], {})
    finally:
        llm_parent_session_id.reset(parent_token)
        llm_route_session_id.reset(session_token)

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "default-session"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "default-session"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "default-parent"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "default-parent"


def test_openai_build_request_rejects_non_ascii_extra_header_value() -> None:
    """Resolved chat_options.extra_headers hit the wire-charset gate at request time."""
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._build_request(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": "秘密token"}},
        )

    message = str(info.value)
    assert "'X-Test'" in message
    assert "position 1" in message
    assert "秘密" not in message


def test_openai_build_request_rejects_non_ascii_model_override() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._build_request([Message("user", ["hi"])], {"model": "模型"})

    message = str(info.value)
    assert "Model ID" in message
    assert "U+6A21" in message


def test_openai_build_request_rejects_outer_space_header_value() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._build_request(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": "token "}},
        )

    message = str(info.value)
    assert "'X-Test'" in message
    assert "ends with a space" in message
    assert "token" not in message


def test_openai_build_request_rejects_non_string_extra_header_value() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._build_request(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": ["secret-value"]}},
        )

    message = str(info.value)
    assert "Header 'X-Test' value must be a string" in message
    assert "secret-value" not in message


def test_openai_build_request_rejects_non_string_extra_header_name() -> None:
    chat_client = make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError, match="Header name at position 1 must be a string"):
        chat_client._build_request(
            [Message("user", ["hi"])],
            {"extra_headers": {123: "value"}},
        )


def test_openai_build_request_allows_dropped_managed_header_with_unsafe_value() -> None:
    """A Chrys-managed header never reaches the wire, so its value is not validated."""
    chat_client = make_chat_client(session_id="sess-123")

    prepared = chat_client._build_request(
        [Message("user", ["hi"])],
        {"extra_headers": {"chrys-debug": "值"}},
    )

    assert "chrys-debug" not in prepared["extra_headers"]


def test_request_headers_validate_on_early_return_path() -> None:
    """Even with no Chrys metadata to merge, caller headers still get the gate."""
    options: dict[str, Any] = {"extra_headers": {"X-Test": "值"}}

    with pytest.raises(ValueError) as info:
        RequestHeaders().stamp(options)

    message = str(info.value)
    assert "'X-Test'" in message
    assert "值" not in message


def test_openai_responses_request_sets_chrys_headers() -> None:
    chat_client = make_responses_chat_client(session_id="sess-123", parent_session_id="parent-123")

    prepared = chat_client._build_request(
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
def test_openai_responses_preserves_allowed_tools_mode(mode: str) -> None:
    chat_client = make_responses_chat_client()
    search_tool = FunctionTool(func=lambda query: query, name="search_docs", description="Search documentation")

    prepared = chat_client._build_request(
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


def test_openai_responses_required_without_allowlist_stays_plain_required() -> None:
    chat_client = make_responses_chat_client()
    search_tool = FunctionTool(func=lambda query: query, name="search_docs", description="Search documentation")

    prepared = chat_client._build_request(
        [Message("user", ["hi"])],
        {"tools": [search_tool], "tool_choice": {"mode": "required"}},
    )

    assert prepared["tool_choice"] == "required"


# ──────────────── integration: the production OpenAI stack ─────────────
#
# These tests build the client stack via the factory and feed it real
# ``openai.types`` ``ChatCompletion`` objects: a choiceless gateway envelope
# is rejected and a valid response still decodes.


def test_integration_choices_none_raises_with_gateway_error_body() -> None:
    """A 200 response carrying a gateway error envelope surfaces as ChatClientException."""
    from openai.types.chat.chat_completion import ChatCompletion

    bad = ChatCompletion.model_construct(
        id="resp-bad",
        choices=None,
        created=0,
        model="gpt-test",
        object="chat.completion",
        error={"message": "rate limit exceeded", "code": 429},
    )

    with pytest.raises(ChatClientException) as exc_info:
        decode_completion(bad, {}, variant=OPENAI)

    assert isinstance(exc_info.value.__cause__, ChatClientInvalidResponseException)
    msg = str(exc_info.value.__cause__)
    assert "missing the required 'choices' array" in msg
    assert "rate limit exceeded" in msg
    assert "429" in msg


def test_integration_valid_empty_choices_response_decodes() -> None:
    """A valid (empty-choices) response should pass through and produce a ChatResponse."""
    from openai.types.chat.chat_completion import ChatCompletion

    valid = ChatCompletion.model_construct(
        id="resp-ok",
        choices=[],
        created=0,
        model="gpt-test",
        object="chat.completion",
        usage=None,
    )

    result = decode_completion(valid, {}, variant=OPENAI)
    assert result.response_id == "resp-ok"
    assert result.messages == []


# ---------------------------------------------------------------------------
# Cache-token preservation
# ---------------------------------------------------------------------------


def test_decode_usage_preserves_cached_tokens_zero() -> None:
    """OpenAI ``cached_tokens=0`` must survive parsing so the UI shows ``0``,
    not ``-``. A naive ``if tokens := ...:`` truthiness check would drop it."""
    from openai.types.completion_usage import CompletionUsage

    from chrys.service.llm.chat_completions.decode import decode_usage

    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    details = decode_usage(usage, variant=OPENAI)
    assert details["prompt/cached_tokens"] == 0
    assert details["cache_read_input_token_count"] == 0
    assert details["completion/reasoning_tokens"] == 0
    assert details["reasoning_output_token_count"] == 0


def test_responses_decode_usage_preserves_cached_tokens_zero() -> None:
    """Responses ``cached_tokens=0`` must survive parsing."""
    from openai.types.responses import ResponseUsage

    from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
    from chrys.service.llm.openai_responses.decode import decode_usage

    usage = ResponseUsage.model_validate(
        {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1050,
        }
    )

    details = decode_usage(usage, variant=OPENAI_RESPONSES)
    assert details["openai.cached_input_tokens"] == 0
    assert details["cache_read_input_token_count"] == 0
    assert details["openai.reasoning_tokens"] == 0
    assert details["reasoning_output_token_count"] == 0


def test_decode_usage_preserves_cached_tokens_nonzero() -> None:
    """A non-zero ``cached_tokens`` is reported as-is."""
    from openai.types.completion_usage import CompletionUsage

    from chrys.service.llm.chat_completions.decode import decode_usage

    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 256},
        }
    )

    details = decode_usage(usage, variant=OPENAI)
    assert details["prompt/cached_tokens"] == 256
    assert details["cache_read_input_token_count"] == 256


def test_decode_usage_omits_cache_key_when_provider_does_not_report() -> None:
    """Absent ``prompt_tokens_details`` must stay absent — ``None`` semantics."""
    from openai.types.completion_usage import CompletionUsage

    from chrys.service.llm.chat_completions.decode import decode_usage

    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
        }
    )

    details = decode_usage(usage, variant=OPENAI)
    assert "prompt/cached_tokens" not in details
    assert "cache_read_input_token_count" not in details
