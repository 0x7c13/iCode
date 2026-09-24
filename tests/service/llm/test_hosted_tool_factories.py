# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Contract pins for the hosted-tool config factories on the Responses client."""

from __future__ import annotations

from types import SimpleNamespace

from chrys.kernel import Content, Message
from chrys.service.llm.anthropic_chat import RawAnthropicClient
from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient
from chrys.service.llm.openai_responses import RawOpenAIChatClient


def _openai_client() -> RawOpenAIChatClient:
    return RawOpenAIChatClient(model="gpt-test", async_client=SimpleNamespace())


def _anthropic_client() -> RawAnthropicClient:
    return RawAnthropicClient(model="claude-test", anthropic_client=SimpleNamespace())


def _chat_completions_client() -> RawOpenAIChatCompletionClient:
    return RawOpenAIChatCompletionClient(model="gpt-test", async_client=SimpleNamespace(base_url=""))


def test_get_mcp_tool_forces_require_approval_never() -> None:
    """Chrys has no ``mcp_approval_request`` round-trip; the platform default
    of ``"always"`` would stall every hosted tool call, so the factory must
    pin ``"never"`` explicitly."""
    tool = RawOpenAIChatClient.get_mcp_tool(name="my mcp", url="https://mcp.example.com")

    assert tool == {
        "type": "mcp",
        "server_label": "my_mcp",
        "server_url": "https://mcp.example.com",
        "require_approval": "never",
    }


def test_get_mcp_tool_optional_fields() -> None:
    tool = RawOpenAIChatClient.get_mcp_tool(
        name="github",
        url="https://mcp.github.com",
        description="GitHub MCP server",
        allowed_tools=["search_issues"],
        headers={"Authorization": "Bearer token"},
    )

    assert tool["require_approval"] == "never"
    assert tool["server_description"] == "GitHub MCP server"
    assert tool["allowed_tools"] == ["search_issues"]
    assert tool["headers"] == {"Authorization": "Bearer token"}


def test_get_mcp_tool_preserves_explicit_empty_allow_list() -> None:
    """``allowed_tools=[]`` means "expose no tools" (matching local MCP
    semantics) — it must not be conflated with ``None`` ("expose all"),
    especially with ``require_approval`` pinned to ``"never"``."""
    tool = RawOpenAIChatClient.get_mcp_tool(name="locked", url="https://mcp.example.com", allowed_tools=[])

    assert tool["allowed_tools"] == []

    unrestricted = RawOpenAIChatClient.get_mcp_tool(name="open", url="https://mcp.example.com")
    assert "allowed_tools" not in unrestricted


def test_openai_custom_stays_informational_while_tool_search_is_hosted() -> None:
    client = _openai_client()
    custom = SimpleNamespace(
        type="custom_tool_call",
        call_id="custom-1",
        id="ctc-1",
        name="python",
        input="print('safe')",
        namespace="container",
    )
    search = SimpleNamespace(
        type="tool_search_call",
        call_id="search-1",
        id="tsc-1",
        arguments={"query": "weather"},
        status="completed",
        execution="server",
        created_by="model",
    )

    custom_content = client._parse_hosted_function_call_content(custom, name=custom.name, arguments=custom.input)
    search_content = client._parse_tool_search_call_content(search)

    assert custom_content.informational_only is True
    assert custom_content.to_dict()["informational_only"] is True
    assert custom_content.additional_properties["item_type"] == "custom_tool_call"
    assert search_content.type == "hosted_tool_call"
    assert search_content.hosted_family == "tool_discovery"
    assert search_content.call_id == "search-1"
    assert search_content.hosted_provider == "openai"


def test_openai_hosted_call_streaming_and_non_streaming_parsers_match() -> None:
    item = SimpleNamespace(
        type="custom_tool_call",
        call_id="custom-1",
        id="ctc-1",
        name="python",
        input="print('safe')",
        namespace="container",
    )
    response = SimpleNamespace(
        metadata=None,
        output=[item],
        id="response-1",
        created_at=0,
        model="gpt-test",
        conversation=None,
        usage=None,
        status="completed",
        incomplete_details=None,
    )
    event = SimpleNamespace(type="response.output_item.done", item=item)
    client = _openai_client()

    non_streaming = client._parse_response_from_openai(response, {"store": False})
    streaming = client._parse_chunk_from_openai(event, {}, {})

    non_streaming_call = non_streaming.messages[0].contents[0]
    streaming_call = streaming.contents[0]
    assert non_streaming_call.to_dict(exclude={"raw_representation"}) == streaming_call.to_dict(
        exclude={"raw_representation"}
    )
    assert non_streaming_call.informational_only is True


def test_anthropic_server_tool_use_is_normalized_as_hosted_search() -> None:
    block = SimpleNamespace(type="server_tool_use", id="server-1", name="web_search", input={"query": "chrys"})

    contents = _anthropic_client()._parse_contents_from_anthropic([block])

    assert len(contents) == 1
    assert contents[0].type == "search_tool_call"
    assert contents[0].hosted_family == "search"
    assert contents[0].hosted_provider == "anthropic"


def test_anthropic_mixed_tool_blocks_split_into_required_roles_without_reordering() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_text("before"),
            Content.from_function_call("local-1", "echo", arguments={"text": "hi"}),
            Content.from_function_result("local-1", result="ok"),
            Content.from_mcp_server_tool_call("mcp-1", "search", server_name="remote", arguments={}),
            Content.from_mcp_server_tool_result("mcp-1", output=[Content.from_text("found")]),
            Content.from_text("after"),
        ],
    )

    prepared = _anthropic_client()._prepare_messages_for_anthropic([message])

    assert [item["role"] for item in prepared] == ["assistant", "user", "assistant"]
    assert [[block["type"] for block in item["content"]] for item in prepared] == [
        ["text", "tool_use"],
        ["tool_result"],
        ["mcp_tool_use", "mcp_tool_result", "text"],
    ]


def test_anthropic_normalizes_a_single_tool_mapping() -> None:
    tool_definition = {"type": "web_search_20250305", "name": "web_search"}

    prepared = _anthropic_client()._prepare_tools_for_anthropic({"tools": tool_definition})

    assert prepared == {"tools": [tool_definition]}


def test_anthropic_drops_format_stamped_reasoning_but_keeps_own_thinking() -> None:
    """Chat-completions dialect reasoning is another provider's state; replaying
    its text as an Anthropic thinking block would forge a thinking signature."""
    message = Message(
        role="assistant",
        contents=[
            Content.from_text_reasoning(
                text="GLM chain of thought",
                additional_properties={"openai_reasoning_format": "reasoning_content"},
            ),
            Content.from_text_reasoning(text="genuine thinking", protected_data="sig-1"),
            Content.from_text("visible answer"),
        ],
    )

    prepared = _anthropic_client()._prepare_messages_for_anthropic([message])

    blocks = prepared[0]["content"]
    thinking_blocks = [block for block in blocks if block["type"] == "thinking"]
    assert thinking_blocks == [{"type": "thinking", "thinking": "genuine thinking", "signature": "sig-1"}]
    assert blocks[-1] == {"type": "text", "text": "visible answer"}


def test_chat_completions_drops_unsigned_reasoning_instead_of_exposing_or_serializing_it() -> None:
    message = Message(
        "assistant",
        [
            Content.from_text_reasoning(text="private chain", id="reasoning-1"),
            Content.from_text("visible answer"),
        ],
    )

    prepared = _chat_completions_client()._prepare_message_for_openai(message)

    assert prepared == [{"role": "assistant", "content": "visible answer"}]
