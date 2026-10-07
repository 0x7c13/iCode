# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which MCP failures the model reads (the server's own error text, never local transport detail), and what a failed result records."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from typing import Any, Literal

import pytest
from mcp import types
from mcp.shared.exceptions import McpError

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationToolCallResult
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.kernel import ChatResponse, Content, FunctionTool, Message, tool
from chrys.kernel.exceptions import ModelVisibleToolError, ToolExecutionException
from chrys.kernel.middleware import ChatMiddlewareLayer, FunctionMiddleware
from chrys.service.agent_middleware import SubAgentEventMiddleware, ToolEventMiddleware
from chrys.service.mcp._http_transport import _HTTPMCPTool
from chrys.service.mcp.content import INVALID_IMAGE_TEXT
from chrys.service.mcp.owned import LOCAL_HTTP_FAILURE_ERROR_DATA, MCPTool
from tests.service.mcp._helpers import _as_client_session, _ScriptedClientSession
from tests.support.event_capture import capture_events
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

_FIXED = "Error: Function failed."
_SECRET = "s3cret"
_LOCAL_DETAIL = (
    f"HTTP MCP request failed: Client error '401 Unauthorized' for url 'https://mcp.example/mcp?token={_SECRET}'"
)


class _ScriptedWire:
    """Innermost chat client: one call to the MCP tool, then a final answer."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        compaction_strategy: object = None,
        tokenizer: object = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        if stream:
            raise NotImplementedError("these tests run the tool loop without streaming")
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            call = Content.from_function_call(call_id="c1", name="remote", arguments={})
            return ChatResponse(messages=[Message(role="assistant", contents=[call])])
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])])


def _tool_error_result(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)


def _mcp_error(message: str, *, code: int = -32602, data: Any = None) -> McpError:
    return McpError(types.ErrorData(code=code, message=message, data=data))


def _remote_tool(*, task_support: Literal["forbidden", "optional", "required"] | None = None) -> types.Tool:
    return types.Tool(
        name="remote",
        description="Remote tool",
        inputSchema={"type": "object", "properties": {}},
        execution=None if task_support is None else types.ToolExecution(taskSupport=task_support),
    )


async def _connected_tool(session: _ScriptedClientSession) -> MCPTool:
    # The HTTP transport's tool class, with the parser that reads structuredContent.
    tool = _HTTPMCPTool(name="srv", url="http://localhost/mcp")
    tool.session = _as_client_session(session)
    await tool.load_tools()
    return tool


async def _loaded_tool(call_outcome: types.CallToolResult | Exception) -> MCPTool:
    return await _connected_tool(_ScriptedClientSession(tools=[_remote_tool()], call_tool=call_outcome))


async def _sent_result(tool: MCPTool) -> Content:
    """Run one call through the kernel tool loop and return the result sent back to the model."""
    wire = _ScriptedWire()
    layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
    await layer.get_response([Message(role="user", contents=["go"])], options={"tools": tool.functions})

    (result,) = [c for m in wire.calls[1] for c in m.contents if c.type == "function_result"]
    return result


async def _failed_result(tool: MCPTool) -> Content:
    """Run one call through the kernel tool loop and return the failed result sent back to the model."""
    result = await _sent_result(tool)
    # A failed call still reads as failed to the loop and the providers' error flag; this
    # record-only field carries the full exception and never goes on the wire.
    assert result.exception is not None
    return result


async def _what_the_model_reads(tool: MCPTool) -> str:
    return str((await _failed_result(tool)).result)


@pytest.mark.parametrize(
    ("call_outcome", "model_reads"),
    [
        (
            _tool_error_result("ENOENT: 'a.txt' not found. Did you mean 'b.txt'?"),
            "Error: ENOENT: 'a.txt' not found. Did you mean 'b.txt'?",
        ),
        (_mcp_error("Invalid params: field 'owner' is required"), "Error: Invalid params: field 'owner' is required"),
        (
            _mcp_error("Timed out while waiting for response to CallToolRequest. Waited 30.0 seconds.", code=408),
            "Error: Timed out while waiting for response to CallToolRequest. Waited 30.0 seconds.",
        ),
        (_mcp_error(_LOCAL_DETAIL, code=-32000, data=LOCAL_HTTP_FAILURE_ERROR_DATA), _FIXED),
        (RuntimeError(f"unexpected failure, token={_SECRET}"), _FIXED),
    ],
    ids=[
        "tool-execution-error",
        "server-json-rpc-error",
        "sdk-timeout",
        "local-http-failure",
        "unexpected-exception",
    ],
)
async def test_mcp_tool_failure_reaches_the_model_only_when_the_server_wrote_it(
    call_outcome: types.CallToolResult | Exception, model_reads: str
) -> None:
    tool = await _loaded_tool(call_outcome)

    assert await _what_the_model_reads(tool) == model_reads


def _image_only_error_result() -> types.CallToolResult:
    image = types.ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png")
    return types.CallToolResult(content=[image], isError=True)


def _empty_error_result() -> types.CallToolResult:
    return types.CallToolResult(content=[], isError=True)


def _structured_error_result(structured: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(content=[], structuredContent=structured, isError=True)


@pytest.mark.parametrize(
    "call_outcome",
    [_image_only_error_result(), _empty_error_result(), _tool_error_result(" Error: "), _mcp_error("  ")],
    ids=["image-only", "empty", "bare-error-prefix", "blank-json-rpc-error"],
)
async def test_mcp_tool_error_without_text_says_so(call_outcome: types.CallToolResult | Exception) -> None:
    tool = await _loaded_tool(call_outcome)

    # Not the repr of the parsed contents, nor the parser's "null" for no content.
    assert await _what_the_model_reads(tool) == "Error: MCP tool 'remote' reported an error without an error message."


async def test_mcp_result_with_an_undecodable_image_still_reaches_the_model() -> None:
    text = types.TextContent(type="text", text="chart:")
    image = types.ImageContent(type="image", data="not base64!", mimeType="image/png")
    tool = await _loaded_tool(types.CallToolResult(content=[text, image]))

    result = await _sent_result(tool)

    assert result.exception is None
    assert [item.text for item in result.items or ()] == ["chart:", INVALID_IMAGE_TEXT]


async def test_mcp_tool_error_sent_only_as_structured_content_reaches_the_model() -> None:
    tool = await _loaded_tool(_structured_error_result({"error": "field 'owner' is required"}))

    assert await _what_the_model_reads(tool) == """Error: {"error": "field 'owner' is required"}"""


async def test_mcp_tool_needing_task_support_tells_the_model_why() -> None:
    session = _ScriptedClientSession(
        tools=[_remote_tool(task_support="required")], call_tool=_tool_error_result("unreachable")
    )
    tool = await _connected_tool(session)

    result = await _what_the_model_reads(tool)

    assert result.startswith("Error: MCP tool 'remote' requires long-running task support")
    assert session.tool_calls == []


async def test_unexpected_mcp_failure_records_every_grouped_error() -> None:
    group = ExceptionGroup("unhandled errors in a TaskGroup", [ValueError("bad protocol bytes")])
    tool = await _loaded_tool(group)

    result = await _failed_result(tool)

    assert result.result == _FIXED
    assert result.exception == (
        "ToolExecutionException: Failed to call tool 'remote'. "
        "(caused by ExceptionGroup: unhandled errors in a TaskGroup [ValueError: bad protocol bytes])"
    )


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        (_mcp_error("Unknown prompt 'p'"), ModelVisibleToolError),
        (_mcp_error(_LOCAL_DETAIL, code=-32000, data=LOCAL_HTTP_FAILURE_ERROR_DATA), ToolExecutionException),
    ],
    ids=["server-json-rpc-error", "local-http-failure"],
)
async def test_mcp_prompt_error_is_model_visible_only_when_the_server_sent_it(
    error: McpError, raised: type[ToolExecutionException]
) -> None:
    tool = MCPTool(name="srv")
    tool.session = _as_client_session(_ScriptedClientSession(get_prompt=error))

    with pytest.raises(ToolExecutionException) as info:
        await tool.get_prompt("p")

    assert type(info.value) is raised


async def test_blank_mcp_prompt_error_says_so() -> None:
    tool = MCPTool(name="srv")
    tool.session = _as_client_session(_ScriptedClientSession(get_prompt=_mcp_error("")))

    with pytest.raises(ModelVisibleToolError) as info:
        await tool.get_prompt("p")

    assert info.value.result_text == "Error: MCP prompt 'p' reported an error without an error message."


async def _mcp_tools(call_outcome: types.CallToolResult | Exception) -> list[FunctionTool]:
    return (await _loaded_tool(call_outcome)).functions


async def _raising_tools(error: Callable[[], Exception]) -> list[FunctionTool]:
    @tool(name="remote")
    async def remote() -> str:
        raise error()

    return [remote]


def _written_over_its_cause() -> Exception:
    clash = ValueError("Duplicate tool name 'remote'. Tool names must be unique.")
    raised = ModelVisibleToolError("Cannot load 'remote': another tool already has that name.", inner_exception=clash)
    raised.__cause__ = clash
    return raised


def _main_turn_cards(bus: EventBus) -> FunctionMiddleware:
    return ToolEventMiddleware(bus, session_id="s", origin=InvocationOrigin("turn", "s", "turn-1", None))


def _sub_agent_cards(bus: EventBus) -> FunctionMiddleware:
    return SubAgentEventMiddleware(
        bus,
        agent_name="Explore",
        invocation_id="inv-1",
        session_id="s",
        origin=InvocationOrigin("sub_agent", "s", "inv-1", None),
    )


@pytest.mark.parametrize("cards", [_main_turn_cards, _sub_agent_cards], ids=["main-turn", "sub-agent"])
@pytest.mark.parametrize(
    ("tools", "model_reads"),
    [
        (
            partial(_mcp_tools, _tool_error_result("ENOENT: 'a.txt' not found.")),
            "Error: ENOENT: 'a.txt' not found.",
        ),
        (
            partial(_mcp_tools, _tool_error_result("Error:")),
            "Error: MCP tool 'remote' reported an error without an error message.",
        ),
        (
            partial(_raising_tools, _written_over_its_cause),
            "Error: Cannot load 'remote': another tool already has that name.",
        ),
        (partial(_raising_tools, lambda: ModelVisibleToolError("Error: ")), _FIXED),
    ],
    ids=["server-error-text", "bare-error-prefix", "written-over-its-cause", "blank-message"],
)
async def test_tool_card_shows_what_the_model_read(
    cards: Callable[[EventBus], FunctionMiddleware],
    tools: Callable[[], Awaitable[list[FunctionTool]]],
    model_reads: str,
) -> None:
    bus = EventBus()
    card_results = await capture_events(bus, InvocationToolCallResult)
    wire = _ScriptedWire()
    layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))

    await layer.get_response(
        [Message(role="user", contents=["go"])], options={"tools": await tools()}, middleware=[cards(bus)]
    )

    (result,) = [c for m in wire.calls[1] for c in m.contents if c.type == "function_result"]
    assert result.result == model_reads
    assert [card.result for card in card_results] == [model_reads]
