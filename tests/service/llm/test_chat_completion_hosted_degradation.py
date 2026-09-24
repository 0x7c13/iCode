# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chat Completions degrades provider-hosted history to plain assistant text.

The Chat Completions wire has no hosted item types: replayed raw hosted
contents reach strict endpoints (GLM, DeepSeek) as unknown content parts or
dangling ``tool_calls`` entries and 400 the request. The degradation runs as
a message rewrite before serialization, so every prep path — including the
reasoning-coalescer delegation — sees only plain contents.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chrys.kernel import Content, Message
from chrys.service.llm.deepseek import DeepSeekChatCompletionClient
from chrys.service.llm.glm import GLMChatCompletionClient
from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient


class _UnusedAsyncOpenAI:
    base_url = "https://api.test"


def _base() -> RawOpenAIChatCompletionClient:
    return RawOpenAIChatCompletionClient(model="gpt-test", async_client=_UnusedAsyncOpenAI())


def _glm() -> GLMChatCompletionClient:
    return GLMChatCompletionClient(model="glm-5.2", async_client=_UnusedAsyncOpenAI())


def _deepseek() -> DeepSeekChatCompletionClient:
    return DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_UnusedAsyncOpenAI())


_CLIENTS = [_base, _glm, _deepseek]


def _search_pair(source_provider: str | None) -> tuple[Content, Content]:
    call = Content.from_search_tool_call(
        "foreign-1",
        tool_name="web_search",
        arguments={"query": "Chrys"},
        status="running",
        hosted_provider=source_provider,
        provider_phase="start",
    )
    result = Content.from_search_tool_result(
        "foreign-1",
        tool_name="web_search",
        result="found",
        status="completed",
        hosted_provider=source_provider,
        provider_phase="terminal",
    )
    return call, result


def _mcp_pair(source_provider: str | None) -> tuple[Content, Content]:
    call = Content.from_mcp_server_tool_call(
        "foreign-1",
        "lookup",
        server_name="docs",
        arguments={"query": "Chrys"},
        provider_status="running",
        hosted_provider=source_provider,
        provider_phase="start",
    )
    result = Content.from_mcp_server_tool_result(
        "foreign-1",
        output="found",
        provider_status="completed",
        hosted_provider=source_provider,
        provider_phase="terminal",
    )
    return call, result


def _assert_clean_wire(prepared: list[dict[str, Any]]) -> None:
    wire = json.dumps(prepared)
    assert "search_tool_call" not in wire
    assert "mcp_server_tool_call" not in wire
    assert all("tool_calls" not in message for message in prepared)


@pytest.mark.parametrize("client_factory", _CLIENTS)
@pytest.mark.parametrize("source_provider", ["openai", "deepseek-openai", "anthropic", None])
@pytest.mark.parametrize(
    ("pair_factory", "expected_tool_line"),
    [(_search_pair, "Tool: web_search"), (_mcp_pair, "Tool: lookup")],
)
def test_foreign_hosted_pair_degrades_to_assistant_context(
    client_factory: Any,
    source_provider: str,
    pair_factory: Any,
    expected_tool_line: str,
) -> None:
    call, result = pair_factory(source_provider)
    prepared = client_factory()._prepare_messages_for_openai(
        [Message("user", ["hi"]), Message("assistant", [call, result])]
    )

    _assert_clean_wire(prepared)
    assert [message["role"] for message in prepared] == ["user", "assistant"]
    summary = prepared[1]["content"]
    assert isinstance(summary, str)
    assert "[Provider-hosted tool context]" in summary
    assert expected_tool_line in summary
    assert "Status: completed" in summary


@pytest.mark.parametrize("client_factory", _CLIENTS)
def test_partner_only_message_is_elided(client_factory: Any) -> None:
    # The result partner degrades to nothing; a message left with zero
    # contents must vanish from the wire instead of emitting an empty record.
    call, result = _search_pair("openai")
    prepared = client_factory()._prepare_messages_for_openai(
        [Message("assistant", [call]), Message("assistant", [result])]
    )

    _assert_clean_wire(prepared)
    assert len(prepared) == 1
    assert "[Provider-hosted tool context]" in prepared[0]["content"]


@pytest.mark.parametrize("client_factory", _CLIENTS)
def test_standalone_hosted_result_degradation_preserves_payload(client_factory: Any) -> None:
    result = Content.from_hosted_tool_result(
        "standalone",
        tool_name="server_task",
        result="valuable output",
        status="completed",
        hosted_family="generic",
        hosted_provider="openai",
        provider_phase="terminal",
        provider_status="completed",
    )

    prepared = client_factory()._prepare_messages_for_openai([Message("assistant", [result])])

    _assert_clean_wire(prepared)
    assert len(prepared) == 1
    assert "valuable output" in prepared[0]["content"]


@pytest.mark.parametrize("client_factory", _CLIENTS)
def test_summary_precedes_final_text_in_the_same_message(client_factory: Any) -> None:
    call, result = _search_pair("openai")
    final = Content.from_text(text="Here's what I found.")
    prepared = client_factory()._prepare_messages_for_openai([Message("assistant", [call, result, final])])

    _assert_clean_wire(prepared)
    # The base client emits one wire message per content while DeepSeek's
    # override coalesces same-role runs; both orderings keep summary first.
    assert all(message["role"] == "assistant" for message in prepared)
    joined = "\n".join(message["content"] for message in prepared)
    assert joined.index("[Provider-hosted tool context]") < joined.index("Here's what I found.")


def test_reasoning_delegation_path_sees_degraded_contents() -> None:
    # A reasoning-bearing assistant message takes the coalescer delegation
    # instead of the plain per-content loop; the rewrite must land before
    # that fork or hosted contents leak through it untouched.
    call, result = _search_pair("openai")
    final = Content.from_text(text="Here's what I found.")
    message = Message(
        "assistant",
        [call, result, final],
        additional_properties={"reasoning_content": "thinking"},
    )
    prepared = _base()._prepare_messages_for_openai([message])

    _assert_clean_wire(prepared)
    assert len(prepared) == 1
    assert prepared[0]["reasoning_content"] == "thinking"
    assert "[Provider-hosted tool context]" in prepared[0]["content"]
    assert "Here's what I found." in prepared[0]["content"]


@pytest.mark.parametrize("client_factory", _CLIENTS)
def test_local_function_history_is_untouched(client_factory: Any) -> None:
    call = Content.from_function_call(call_id="local-1", name="get_weather", arguments='{"city": "SF"}')
    result = Content.from_function_result(call_id="local-1", result="sunny")
    prepared = client_factory()._prepare_messages_for_openai([Message("assistant", [call]), Message("tool", [result])])

    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert prepared[1]["role"] == "tool"
