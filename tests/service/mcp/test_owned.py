# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the owned MCP engine: allowed-tools filtering, closure argument hygiene, sampling, catalog reloads, lifecycle."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest

import chrys.service.mcp.owned as owned_mcp
from chrys.kernel import ChatResponse, Message
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.service.mcp._http_transport import _HTTPMCPTool
from chrys.service.mcp.adapter import MCPAdapter
from chrys.service.mcp.errors import (
    MCPToolNameCollisionError,
    MCPToolNameValidationError,
)
from chrys.service.mcp.owned import (
    _MCP_FRAMEWORK_DENYLIST,
    _MCP_NORMALIZED_NAME_KEY,
    _MCP_REMOTE_NAME_KEY,
    MCPStreamableHTTPTool,
    MCPTool,
)
from chrys.service.profiles.agents.schema import MCPServerConfig
from tests.service.mcp._helpers import (
    _FakeConnectionTool,
    _function_tool,
    _load_fake_remote_tools,
    _mcp_remote_prompt,
    _mcp_remote_tool,
)


async def _load_calling_remote_tools(tool: MCPTool, *remote_tools: SimpleNamespace) -> SimpleNamespace:
    from mcp import types

    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=list(remote_tools), nextCursor=None)),
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
                isError=False,
            )
        ),
    )
    tool.session = session
    tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await tool.load_tools()
    return session


def _sampling_params(max_tokens: int = 9999) -> Any:
    from mcp import types

    return types.CreateMessageRequestParams(
        messages=[
            types.SamplingMessage(
                role="user",
                content=types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=max_tokens,
    )


def _tool_list_changed_notification() -> Any:
    from mcp import types

    return types.ServerNotification(root=types.ToolListChangedNotification())


class _SamplingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def get_response(self, messages: Any, options: Any = None) -> ChatResponse:
        self.calls.append({"messages": messages, "options": dict(options or {})})
        return ChatResponse(messages=[Message(role="assistant", contents=["sampled"])], model="m-test")


# ---------------------------------------------------------------------------
# MCPTool.functions — allowed_tools filtering
# ---------------------------------------------------------------------------


def test_mcp_tool_empty_allowed_tools_exposes_no_functions() -> None:
    tool = MCPTool(name="m", allowed_tools=[])
    tool._functions = [_function_tool("one"), _function_tool("two")]

    assert tool.functions == []


def test_mcp_allowed_tools_does_not_match_lossy_normalized_alias() -> None:
    exposed = _function_tool("delete-everything")
    exposed.additional_properties = {
        _MCP_REMOTE_NAME_KEY: "delete/everything",
        _MCP_NORMALIZED_NAME_KEY: "delete-everything",
    }
    tool = MCPTool(name="m", allowed_tools=["delete-everything"])
    tool._functions = [exposed]

    assert tool.functions == []


def test_mcp_allowed_tools_accepts_local_name_when_remote_is_already_normalized() -> None:
    exposed = _function_tool("srv_echo")
    exposed.additional_properties = {
        _MCP_REMOTE_NAME_KEY: "echo",
        _MCP_NORMALIZED_NAME_KEY: "echo",
    }
    tool = MCPTool(name="m", allowed_tools=["srv_echo"])
    tool._functions = [exposed]

    assert tool.functions == [exposed]


# ---------------------------------------------------------------------------
# generated closure argument hygiene, header providers, request meta precedence
# ---------------------------------------------------------------------------


class _RecordingMCPTool(MCPTool):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="m", **kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        self.calls.append((tool_name, dict(kwargs)))
        return "ok"


async def test_mcp_generated_tool_closure_strips_model_supplied_meta() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(tool, _mcp_remote_tool("remote"))
    func = tool._functions[0]

    await func.invoke(arguments={"_meta": {"forged": "bad"}}, skip_parsing=True)

    assert tool.calls == [("remote", {})]


async def test_mcp_generated_tool_closure_preserves_trusted_runtime_meta() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(tool, _mcp_remote_tool("remote"))
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={},
        kwargs={"_meta": {"trusted": "ok"}},
    )

    await func.invoke(arguments={"_meta": {"forged": "bad"}}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {"_meta": {"trusted": "ok"}})]


async def test_mcp_generated_tool_separates_declared_arguments_from_runtime_kwargs() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(
        tool,
        _mcp_remote_tool(
            "remote",
            input_schema={
                "type": "object",
                "properties": {"session": {"type": "string"}},
            },
        ),
    )
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"session": "sr-design"},
        kwargs={"session": object(), "future_runtime_key": object()},
    )

    await func.invoke(arguments={"session": "sr-design"}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {"session": "sr-design"})]


async def test_mcp_generated_tool_does_not_substitute_runtime_value_for_omitted_optional_argument() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(
        tool,
        _mcp_remote_tool(
            "remote",
            input_schema={
                "type": "object",
                "properties": {"session": {"type": "string"}},
            },
        ),
    )
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={},
        kwargs={"session": object()},
    )

    await func.invoke(arguments={}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {})]


async def test_mcp_generated_tool_forwards_only_explicit_runtime_extras() -> None:
    tool = _RecordingMCPTool(additional_tool_argument_names={"remote": ["tenant_id"]})
    await _load_fake_remote_tools(tool, _mcp_remote_tool("remote"))
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={},
        kwargs={"tenant_id": "trusted-tenant", "internal": object()},
    )

    await func.invoke(arguments={}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {"tenant_id": "trusted-tenant"})]


async def test_mcp_generated_tool_cannot_override_bound_remote_name() -> None:
    tool = MCPTool(name="m", allowed_tools=["safe"])
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        ),
        _mcp_remote_tool("danger"),
    )
    assert [func.name for func in tool.functions] == ["safe"]

    await tool.functions[0].invoke(
        arguments={"value": "ok", "_remote_tool_name": "danger"},
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"value": "ok"}


async def test_mcp_declared_remote_name_argument_is_data_not_dispatch_control() -> None:
    tool = MCPTool(name="m", allowed_tools=["safe"])
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"_remote_tool_name": {"type": "string"}},
            },
        ),
        _mcp_remote_tool("danger"),
    )

    await tool.functions[0].invoke(
        arguments={"_remote_tool_name": "danger"},
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"_remote_tool_name": "danger"}


async def test_mcp_declared_ctx_argument_survives_context_injection() -> None:
    tool = MCPTool(name="m")
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"ctx": {"type": "string"}},
                "required": ["ctx"],
            },
        ),
    )

    await tool.functions[0].invoke(
        arguments={"ctx": "business-value"},
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"ctx": "business-value"}


async def test_mcp_trusted_runtime_extra_overrides_same_named_model_argument() -> None:
    tool = MCPTool(
        name="m",
        additional_tool_argument_names={"safe": ["tenant_id"]},
    )
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"tenant_id": {"type": "string"}},
            },
        ),
    )
    func = tool.functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"tenant_id": "model-tenant"},
        kwargs={"tenant_id": "trusted-tenant"},
    )

    await func.invoke(
        arguments={"tenant_id": "model-tenant"},
        context=context,
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"tenant_id": "trusted-tenant"}


@pytest.mark.parametrize("argument_name", sorted(_MCP_FRAMEWORK_DENYLIST - {"_meta"}))
def test_mcp_declared_framework_named_argument_is_forwarded(argument_name: str) -> None:
    tool = MCPTool(name="m")
    tool._tool_param_names_by_name = {"remote": {argument_name}}

    filtered, _meta = tool._prepare_call_kwargs("remote", {argument_name: "declared-value"})

    assert filtered == {argument_name: "declared-value"}


async def test_mcp_declared_collision_reaches_client_session_without_model_meta() -> None:
    from mcp import types

    session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    _mcp_remote_tool(
                        "remote",
                        input_schema={
                            "type": "object",
                            "properties": {"session": {"type": "string"}},
                        },
                    )
                ],
                nextCursor=None,
            )
        ),
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
                isError=False,
            )
        ),
    )
    tool = MCPTool(name="m", session=session)
    tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await tool.load_tools()
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"session": "sr-design", "_meta": {"forged": "bad"}},
        kwargs={"session": object()},
    )

    await func.invoke(
        arguments={"session": "sr-design", "_meta": {"forged": "bad"}},
        context=context,
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("remote",)
    assert call.kwargs["arguments"] == {"session": "sr-design"}
    assert call.kwargs["meta"] is None or "forged" not in call.kwargs["meta"]


async def test_mcp_header_provider_sees_model_arguments_and_explicit_runtime_extras_only() -> None:
    from mcp import types

    provider_inputs: list[dict[str, Any]] = []

    def provide_headers(arguments: dict[str, Any]) -> dict[str, str]:
        provider_inputs.append(dict(arguments))
        return {}

    session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    _mcp_remote_tool(
                        "remote",
                        input_schema={
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    )
                ],
                nextCursor=None,
            )
        ),
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
                isError=False,
            )
        ),
    )
    tool = MCPStreamableHTTPTool(
        name="m",
        url="https://mcp.example/mcp",
        session=session,
        header_provider=provide_headers,
        additional_tool_argument_names={"remote": ["tenant_id"]},
    )
    tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await tool.load_tools()
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"value": "model-value", "_meta": {"forged": "bad"}},
        kwargs={"tenant_id": "trusted-tenant", "session": object(), "internal": object()},
    )

    await func.invoke(
        arguments={"value": "model-value", "_meta": {"forged": "bad"}},
        context=context,
        skip_parsing=True,
    )

    assert provider_inputs == [{"tenant_id": "trusted-tenant", "value": "model-value"}]
    call = session.call_tool.await_args
    assert call.kwargs["arguments"] == {"tenant_id": "trusted-tenant", "value": "model-value"}


def test_mcp_request_meta_precedence_is_tool_meta_over_otel_over_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    def inject(carrier: dict[str, str]) -> None:
        carrier.update({"traceparent": "otel", "baggage": "otel"})

    monkeypatch.setattr(owned_mcp.propagate, "inject", inject)
    tool = MCPTool(name="m")
    tool._tool_param_names_by_name = {"remote": {"value"}}
    tool._tool_call_meta_by_name = {"remote": {"traceparent": "tool", "tool": "meta"}}

    filtered, meta = tool._prepare_call_kwargs(
        "remote",
        {
            "value": "ok",
            "_meta": {"traceparent": "caller", "caller": "meta"},
        },
    )

    assert filtered == {"value": "ok"}
    assert meta == {
        "traceparent": "tool",
        "caller": "meta",
        "baggage": "otel",
        "tool": "meta",
    }


# ---------------------------------------------------------------------------
# MCPTool.sampling_callback
# ---------------------------------------------------------------------------


def _approve(_params: Any) -> bool:
    return True


async def _approve_async(_params: Any) -> bool:
    return True


async def _deny_async(_params: Any) -> bool:
    return False


def _fail(_params: Any) -> bool:
    raise RuntimeError("nope")


@pytest.mark.parametrize(
    ("tool_kwargs", "params_kwargs", "expected_max_tokens", "expected_message"),
    [
        pytest.param({}, {}, None, "disabled by default", id="denied-by-default"),
        pytest.param(
            {"sampling_approval_callback": _approve, "sampling_max_tokens": 10},
            {"max_tokens": 99},
            10,
            None,
            id="sync-approve-clamps-to-cap",
        ),
        pytest.param(
            {"sampling_approval_callback": _approve_async, "sampling_max_tokens": None},
            {"max_tokens": 99},
            99,
            None,
            id="async-approve-uncapped",
        ),
        pytest.param(
            {"sampling_approval_callback": _approve_async, "sampling_max_tokens": 100},
            {"max_tokens": 99},
            99,
            None,
            id="async-approve-under-cap",
        ),
        pytest.param({"sampling_approval_callback": _deny_async}, {}, None, None, id="async-deny"),
        pytest.param({"sampling_approval_callback": _fail}, {}, None, None, id="callback-error"),
    ],
)
async def test_mcp_sampling_callback_gates_and_caps_requests(
    tool_kwargs: dict[str, Any],
    params_kwargs: dict[str, Any],
    expected_max_tokens: int | None,
    expected_message: str | None,
) -> None:
    """Sampling is denied unless the approval callback allows it; the client only sees the capped budget.

    The denial rows pass only the argument they are about: ``denied-by-default``
    supplies no callback and no cap at all, so it exercises ``MCPTool``'s own
    defaults, and the two rejecting-callback rows leave the cap unset because no
    budget is negotiated on a request the callback never lets through.
    """
    from mcp import types

    client = _SamplingClient()
    tool = MCPTool(name="m", client=client, **tool_kwargs)

    result = await tool.sampling_callback(SimpleNamespace(), _sampling_params(**params_kwargs))

    if expected_max_tokens is None:
        assert isinstance(result, types.ErrorData)
        assert result.code == types.INVALID_REQUEST
        if expected_message is not None:
            assert expected_message in result.message
        assert client.calls == []
    else:
        assert isinstance(result, types.CreateMessageResult)
        assert result.model == "m-test"
        assert client.calls[0]["options"]["max_tokens"] == expected_max_tokens


async def test_mcp_sampling_rate_limit_resets_with_session_state() -> None:
    from mcp import types

    client = _SamplingClient()
    tool = MCPTool(
        name="m",
        client=client,
        sampling_approval_callback=lambda params: True,
        sampling_max_requests=1,
    )

    first = await tool.sampling_callback(SimpleNamespace(), _sampling_params())
    second = await tool.sampling_callback(SimpleNamespace(), _sampling_params())
    tool._reset_session_state()
    third = await tool.sampling_callback(SimpleNamespace(), _sampling_params())

    assert isinstance(first, types.CreateMessageResult)
    assert isinstance(second, types.ErrorData)
    assert second.code == types.INVALID_REQUEST
    assert isinstance(third, types.CreateMessageResult)
    assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# owned catalog — normalization, collisions, notification-driven reloads
# ---------------------------------------------------------------------------


async def test_owned_catalog_normalizes_periods_to_provider_safe_names() -> None:
    owned_tool = MCPTool(name="s")
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool("github.v1.search"))

    assert [
        (function.name, function.additional_properties[_MCP_REMOTE_NAME_KEY]) for function in owned_tool.functions
    ] == [("github-v1-search", "github.v1.search")]


@pytest.mark.parametrize("connection_path", ["test", "agent"])
async def test_prefix_and_remote_name_combination_over_64_characters_fails_early(connection_path: str) -> None:
    prefix = "a" * 50
    remote_name = "b" * 14
    owned_tool = MCPTool(name="s", tool_name_prefix=prefix)
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool(remote_name))
    assert owned_tool.functions[0].name == f"{prefix}_{remote_name}"

    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python", tool_name_prefix=prefix)
    fake = _FakeConnectionTool(functions=owned_tool.functions)
    with (
        patch("chrys.service.mcp._connection._create_mcp_tool", return_value=fake),
        pytest.raises(MCPToolNameValidationError, match=r"65 characters.*maximum is 64"),
    ):
        if connection_path == "test":
            await adapter.test_connection(config)
        else:
            await adapter.connect(config)

    await adapter.disconnect_all()


@pytest.mark.parametrize("connection_path", ["test", "agent"])
async def test_owned_catalog_normalized_collision_fails_test_and_agent_connection(connection_path: str) -> None:
    owned_tool = MCPTool(name="s")
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool("a/b"), _mcp_remote_tool("a-b"))
    exposed = [
        (function.name, function.additional_properties[_MCP_REMOTE_NAME_KEY]) for function in owned_tool.functions
    ]
    assert exposed == [("a-b", "a/b"), ("a-b", "a-b")]

    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fake = _FakeConnectionTool(functions=owned_tool.functions)
    with (
        patch("chrys.service.mcp._connection._create_mcp_tool", return_value=fake),
        pytest.raises(MCPToolNameCollisionError, match=r"invalid tool configuration.*a-b"),
    ):
        if connection_path == "test":
            await adapter.test_connection(config)
        else:
            await adapter.connect(config)

    await adapter.disconnect_all()


@pytest.mark.parametrize("remote_order", [("a/b", "a-b"), ("a-b", "a/b")])
async def test_owned_catalog_filters_allowlist_before_collision_validation(remote_order: tuple[str, str]) -> None:
    owned_tool = MCPTool(name="s", allowed_tools=["a/b"])
    await _load_fake_remote_tools(owned_tool, *(_mcp_remote_tool(name) for name in remote_order))

    assert [
        (function.name, function.additional_properties[_MCP_REMOTE_NAME_KEY]) for function in owned_tool.functions
    ] == [("a-b", "a/b")]


async def test_owned_catalog_tool_and_prompt_collision_fails_connection() -> None:
    owned_tool = MCPTool(name="s")
    owned_tool.session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_mcp_remote_tool("shared")], nextCursor=None)),
        list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[_mcp_remote_prompt("shared")], nextCursor=None)),
    )
    owned_tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await owned_tool.load_tools()
    await owned_tool.load_prompts()
    assert [function.name for function in owned_tool.functions] == ["shared", "shared"]

    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fake = _FakeConnectionTool(functions=owned_tool.functions)
    with (
        patch("chrys.service.mcp._connection._create_mcp_tool", return_value=fake),
        pytest.raises(MCPToolNameCollisionError, match=r"invalid tool configuration.*shared") as exc_info,
    ):
        await adapter.connect(config)

    assert "disable 'Expose server prompts'" in str(exc_info.value)
    await adapter.disconnect_all()


async def test_owned_catalog_reload_deduplicates_only_the_same_remote_declaration() -> None:
    owned_tool = MCPTool(name="s")
    owned_tool.session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_mcp_remote_tool("tool")], nextCursor=None)),
        list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[_mcp_remote_prompt("prompt")], nextCursor=None)),
    )
    owned_tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]

    await owned_tool.load_tools()
    await owned_tool.load_prompts()
    await owned_tool.load_tools()
    await owned_tool.load_prompts()

    assert [function.name for function in owned_tool.functions] == ["tool", "prompt"]


async def test_owned_catalog_notification_reloads_without_duplicate_remote_tools() -> None:
    owned_tool = MCPTool(name="s")
    list_tools = AsyncMock(return_value=SimpleNamespace(tools=[_mcp_remote_tool("tool")], nextCursor=None))
    owned_tool.session = SimpleNamespace(list_tools=list_tools)
    owned_tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await owned_tool.load_tools()

    await owned_tool.message_handler(_tool_list_changed_notification())
    tasks = list(owned_tool._pending_reload_tasks)
    assert len(tasks) == 1
    await asyncio.gather(*tasks)

    assert [function.name for function in owned_tool.functions] == ["tool"]
    assert list_tools.await_count == 2


async def test_owned_catalog_notifications_coalesce_by_cancelling_first_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_tool = MCPTool(name="s")
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    calls = 0
    cancelled = 0

    async def slow_load_tools() -> None:
        nonlocal calls, cancelled
        calls += 1
        if calls == 1:
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled += 1
                raise
        second_started.set()
        await second_release.wait()

    monkeypatch.setattr(owned_tool, "load_tools", slow_load_tools)
    await owned_tool.message_handler(_tool_list_changed_notification())
    first_task = next(iter(owned_tool._pending_reload_tasks))
    await asyncio.wait_for(first_started.wait(), timeout=5)

    await owned_tool.message_handler(_tool_list_changed_notification())
    second_task = next(task for task in owned_tool._pending_reload_tasks if task is not first_task)
    await asyncio.wait_for(second_started.wait(), timeout=5)
    await asyncio.gather(first_task, return_exceptions=True)

    assert cancelled == 1
    assert calls == 2
    second_release.set()
    await second_task


async def test_owned_catalog_close_cancels_pending_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    owned_tool = MCPTool(name="s")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def pending_load_tools() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(owned_tool, "load_tools", pending_load_tools)
    await owned_tool.message_handler(_tool_list_changed_notification())
    await asyncio.wait_for(started.wait(), timeout=5)

    await owned_tool.close()

    assert cancelled.is_set()
    assert owned_tool._pending_reload_tasks == set()


async def test_owned_catalog_reload_exception_is_logged_not_propagated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    owned_tool = MCPTool(name="s")

    async def failing_load_tools() -> None:
        raise RuntimeError("catalog failed")

    monkeypatch.setattr(owned_tool, "load_tools", failing_load_tools)
    with caplog.at_level(logging.WARNING, logger=owned_mcp.__name__):
        await owned_tool.message_handler(_tool_list_changed_notification())
        tasks = list(owned_tool._pending_reload_tasks)
        assert len(tasks) == 1
        await asyncio.gather(*tasks)

    assert "Background MCP reload failed" in caplog.text
    assert "catalog failed" in caplog.text


# ---------------------------------------------------------------------------
# lifecycle owner task, reconnect wrapping, session-state reset
# ---------------------------------------------------------------------------


async def test_owned_mcp_close_runs_on_lifecycle_owner_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open/close must happen in the same task for anyio cancel scopes."""
    enter_task: asyncio.Task[object] | None = None
    exit_task: asyncio.Task[object] | None = None

    @asynccontextmanager
    async def transport() -> Any:
        nonlocal enter_task, exit_task
        enter_task = asyncio.current_task()
        yield object(), object()
        exit_task = asyncio.current_task()

    class _FakeClientSession:
        _request_id = 1

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeClientSession:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def initialize(self) -> SimpleNamespace:
            return SimpleNamespace(protocolVersion="2024-11-05", capabilities=None)

    class _TaskRecordingTool(MCPTool):
        def get_mcp_client(self) -> Any:
            return transport()

    monkeypatch.setattr("mcp.client.session.ClientSession", _FakeClientSession)

    tool = _TaskRecordingTool(name="task-recorder")
    await tool.connect()
    assert enter_task is not None
    assert enter_task is not asyncio.current_task()

    close_task = asyncio.create_task(tool.close())
    await close_task

    assert exit_task is enter_task


async def test_owned_mcp_initialize_failure_does_not_commit_closed_session(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def transport() -> Any:
        yield object(), object()

    class _FakeClientSession:
        _request_id = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False

        async def __aenter__(self) -> _FakeClientSession:
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.closed = True

        async def initialize(self) -> SimpleNamespace:
            raise RuntimeError("init failed")

    class _InitFailTool(MCPTool):
        def get_mcp_client(self) -> Any:
            return transport()

    monkeypatch.setattr("mcp.client.session.ClientSession", _FakeClientSession)

    tool = _InitFailTool(name="init-fail")
    with pytest.raises(Exception, match="init failed"):
        await tool.connect()

    assert tool.session is None
    assert tool.is_connected is False


async def test_load_retries_reconnect_without_recursive_configured_load() -> None:
    """ClosedResourceError during list pagination must not deadlock on the load lock."""
    from mcp import types

    tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
    reconnect_calls = 0

    async def reconnect_without_loading() -> None:
        nonlocal reconnect_calls
        reconnect_calls += 1
        assert tool._function_load_lock.locked()

    tool._ensure_connected = AsyncMock()
    tool._reconnect_without_loading = AsyncMock(side_effect=reconnect_without_loading)
    tool.session = SimpleNamespace(
        list_tools=AsyncMock(side_effect=[anyio.ClosedResourceError(), types.ListToolsResult(tools=[])]),
        list_prompts=AsyncMock(side_effect=[anyio.ClosedResourceError(), types.ListPromptsResult(prompts=[])]),
    )

    await asyncio.wait_for(tool.load_tools(), timeout=20.0)
    await asyncio.wait_for(tool.load_prompts(), timeout=20.0)

    assert reconnect_calls == 2
    assert tool._reconnect_without_loading.await_count == 2


async def test_load_reconnect_failure_is_wrapped() -> None:
    tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
    tool._ensure_connected = AsyncMock()
    tool._reconnect_without_loading = AsyncMock(side_effect=RuntimeError("reconnect failed"))
    tool.session = SimpleNamespace(list_tools=AsyncMock(side_effect=anyio.ClosedResourceError()))

    with pytest.raises(Exception) as info:
        await tool.load_tools()

    assert type(info.value).__name__ == "ToolExecutionException"
    assert "Failed to reconnect to MCP server." in str(info.value)


async def test_get_prompt_reconnect_failure_is_wrapped() -> None:
    tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
    tool.connect = AsyncMock(side_effect=RuntimeError("reconnect failed"))
    tool.session = SimpleNamespace(get_prompt=AsyncMock(side_effect=anyio.ClosedResourceError()))

    with pytest.raises(Exception) as info:
        await tool.get_prompt("p")

    assert type(info.value).__name__ == "ToolExecutionException"
    assert "Failed to reconnect to MCP server." in str(info.value)
    tool.connect.assert_awaited_once_with(reset=True)


def test_server_instructions_reset_on_session_state_reset() -> None:
    """_server_instructions is cleared after _reset_session_state."""
    tool = MCPTool(name="test")
    tool._server_instructions = "Some instructions"
    assert tool._server_instructions == "Some instructions"

    tool._reset_session_state()
    assert tool._server_instructions is None
