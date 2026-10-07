# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fail-loudly contracts for MCP symbols chrys depends on.

``chrys.service.mcp.adapter`` reaches into a handful of **private** symbols of the
official ``mcp`` SDK and subclasses the Chrys-owned MCP engine.  Each private
SDK touchpoint either has a runtime fallback (ImportError → stock client /
literal defaults) or is an owned method whose shape adapter mixins rely on.

These tests pin the **existence and shape** of every such symbol so a dependency
upgrade that moves, renames, or reshapes one fails *here* — loudly, in one
obvious place — instead of silently disabling a workaround (stdout banner
tolerance, ping-storm suppression, structured-content fallback) at runtime.

The SDK ``StreamableHTTPTransport._handle_post_request`` hook that
``_chrys_streamable_http_client`` overrides is pinned here as well; the
behaviour it enables (POST-failure wake-up, stock-client fallbacks) is
exercised in ``test_http_transport.py``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from pydantic import BaseModel


def _assert_keyword_call_shape(
    func: object,
    *,
    required_keywords: set[str],
    call_label: str,
) -> inspect.Signature:
    """Assert chrys can keep calling a dependency hook with named arguments."""
    signature = inspect.signature(func)
    params = signature.parameters
    assert required_keywords <= set(params), f"{call_label} is missing parameters chrys passes"
    keywordable = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    for name in required_keywords:
        assert params[name].kind in keywordable, f"{call_label}.{name} is no longer keyword-callable"
    new_required = {
        name
        for name, param in params.items()
        if name not in required_keywords
        and param.kind in {inspect.Parameter.POSITIONAL_ONLY, *keywordable}
        and param.default is inspect.Parameter.empty
    }
    assert not new_required, f"{call_label} added required parameters chrys does not pass: {sorted(new_required)}"
    return signature


# --------------------------------------------------------------------------- #
# mcp.client.stdio — private primitives behind ``tolerant_stdio_client``
# --------------------------------------------------------------------------- #


def test_stdio_private_primitives_contract() -> None:
    """``tolerant_stdio_client`` is built on these private stdio symbols.

    If any disappears the adapter degrades to the stock ``stdio_client`` (losing
    banner tolerance) — but a *rename* of the kwargs we pass would instead break
    at runtime without tripping the ImportError fallback, so pin the signatures.
    """
    import mcp.client.stdio as stdio

    # Process-termination knobs used in the cleanup ``finally``.
    assert stdio.PROCESS_TERMINATION_TIMEOUT == 2.0

    terminate = getattr(stdio, "_terminate_process_tree", None)
    assert terminate is not None, "mcp SDK removed mcp.client.stdio._terminate_process_tree"
    assert inspect.iscoroutinefunction(terminate)

    spawn = getattr(stdio, "_create_platform_compatible_process", None)
    assert spawn is not None, "mcp SDK removed mcp.client.stdio._create_platform_compatible_process"
    assert inspect.iscoroutinefunction(spawn)
    _assert_keyword_call_shape(
        spawn,
        required_keywords={"command", "args", "env", "errlog", "cwd"},
        call_label="mcp.client.stdio._create_platform_compatible_process",
    )

    resolve = getattr(stdio, "_get_executable_command", None)
    assert resolve is not None, "mcp SDK removed mcp.client.stdio._get_executable_command"
    assert callable(resolve)


def test_stdio_stock_fallback_contract() -> None:
    """The stock-client fallback calls ``stdio_client(server, errlog=...)``.

    It rebuilds the server params via ``StdioServerParameters.model_copy`` to
    re-inject chrys's sanitized inherited environment, so the params type must remain
    a pydantic model and the stock client must keep its ``(server, errlog)`` API.
    """
    import mcp.client.stdio as stdio

    assert issubclass(stdio.StdioServerParameters, BaseModel), (
        "mcp SDK changed StdioServerParameters off pydantic — model_copy(update=...) fallback breaks"
    )
    assert hasattr(stdio.StdioServerParameters, "model_copy")

    client = getattr(stdio, "stdio_client", None)
    assert client is not None, "mcp SDK removed the public mcp.client.stdio.stdio_client"
    signature = _assert_keyword_call_shape(
        client,
        required_keywords={"server", "errlog"},
        call_label="mcp.client.stdio.stdio_client",
    )
    params = signature.parameters
    assert {"server", "errlog"} <= set(params)
    assert params["errlog"].default is not inspect.Parameter.empty


# --------------------------------------------------------------------------- #
# mcp.shared.message — message wrapper used by patched transports
# --------------------------------------------------------------------------- #


def test_session_message_contract() -> None:
    """Patched HTTP/stdio transports construct ``SessionMessage`` directly."""
    from mcp.shared.message import SessionMessage

    signature = _assert_keyword_call_shape(
        SessionMessage,
        required_keywords={"message"},
        call_label="mcp.shared.message.SessionMessage",
    )
    params = signature.parameters
    assert params["message"].default is inspect.Parameter.empty
    assert params["message"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
        "SessionMessage.message must stay positional-or-keyword: stdio calls SessionMessage(message) "
        "positionally, while HTTP uses message=."
    )


# --------------------------------------------------------------------------- #
# mcp.shared._httpx_utils — timeout constants behind ``_build_httpx_client``
# --------------------------------------------------------------------------- #


def test_httpx_utils_default_constants_contract() -> None:
    """``_build_httpx_client`` mirrors these defaults.

    Its literal ImportError fallback (30.0s connect / 300.0s SSE read) is only
    the correct degraded value while these constants hold; pin them so a drift in
    the upstream defaults is visible rather than silently diverging.
    """
    from mcp.shared._httpx_utils import (
        MCP_DEFAULT_SSE_READ_TIMEOUT,
        MCP_DEFAULT_TIMEOUT,
        create_mcp_http_client,
    )

    assert MCP_DEFAULT_TIMEOUT == 30.0
    assert MCP_DEFAULT_SSE_READ_TIMEOUT == 300.0
    assert callable(create_mcp_http_client)


# --------------------------------------------------------------------------- #
# chrys.service.mcp.owned — owned base methods
# --------------------------------------------------------------------------- #


def test_owned_mcptool_overridden_methods_exist() -> None:
    """Adapter mixins override these owned base methods."""
    from chrys.service.mcp.owned import MCPTool

    ensure = getattr(MCPTool, "_ensure_connected", None)
    assert ensure is not None
    assert inspect.iscoroutinefunction(ensure)
    assert list(inspect.signature(ensure).parameters) == ["self"]

    parse = getattr(MCPTool, "_parse_tool_result_from_mcp", None)
    assert parse is not None
    assert not inspect.iscoroutinefunction(parse)
    assert list(inspect.signature(parse).parameters) == ["self", "mcp_type"]


def test_owned_mcptool_public_surface_contract() -> None:
    """Adapter / cache read ``MCPTool.functions`` and await ``MCPTool.call_tool``."""
    from chrys.service.mcp.owned import MCPTool

    assert isinstance(MCPTool.functions, property)
    assert inspect.iscoroutinefunction(MCPTool.call_tool)


# --------------------------------------------------------------------------- #
# mcp.client.streamable_http — private POST hook overridden by the patched client
# --------------------------------------------------------------------------- #


def test_streamable_http_private_post_hook_contract() -> None:
    """Fail loudly if MCP SDK private HTTP hook changes under our patch.

    ``_chrys_streamable_http_client`` subclasses ``StreamableHTTPTransport``
    and overrides ``_handle_post_request`` to wake pending MCP requests when
    an HTTP POST task fails.  That is a private SDK hook, so dependency updates
    must surface here instead of silently disabling the workaround.
    """
    import mcp.client.streamable_http as streamable_http
    from mcp.client.streamable_http import RequestContext, StreamableHTTPTransport

    hook = getattr(StreamableHTTPTransport, "_handle_post_request", None)
    assert hook is not None, "MCP SDK removed StreamableHTTPTransport._handle_post_request"
    assert inspect.iscoroutinefunction(hook), "MCP SDK changed _handle_post_request away from async"

    signature = inspect.signature(hook)
    assert list(signature.parameters) == ["self", "ctx"]
    assert signature.parameters["ctx"].annotation is RequestContext
    assert signature.return_annotation is None

    context_fields = set(getattr(RequestContext, "__annotations__", {}))
    assert {"client", "session_message", "read_stream_writer"} <= context_fields

    assert streamable_http.CONTENT_TYPE == "content-type"
    assert streamable_http.JSON == "application/json"
    assert streamable_http.SSE == "text/event-stream"

    helper_names = [
        "_prepare_headers",
        "_is_initialization_request",
        "_maybe_extract_session_id_from_response",
        "_handle_json_response",
        "_handle_sse_response",
        "_handle_unexpected_content_type",
        "_send_session_terminated_error",
    ]
    for name in helper_names:
        assert hasattr(StreamableHTTPTransport, name), f"MCP SDK removed StreamableHTTPTransport.{name}"


# --------------------------------------------------------------------------- #
# mcp client side — error text the SDK makes up reaches the model
# --------------------------------------------------------------------------- #

# Every ErrorData message the SDK builds outside ``mcp/server``, as (module,
# message source). ``MCPTool`` hands an McpError's message to the model as the
# server's own error text; the SDK's client-side errors pass as well because
# none of these messages interpolates local detail — exception text, paths,
# stderr, URLs or headers. (Chrys's own HTTP transport marks its synthesized
# errors with ``LOCAL_HTTP_FAILURE_ERROR_DATA`` for that reason.)
_SDK_CLIENT_SIDE_ERROR_MESSAGES = {
    ("client/experimental/task_handlers.py", "'Task-augmented elicitation not supported'"),
    ("client/experimental/task_handlers.py", "'Task-augmented sampling not supported'"),
    ("client/experimental/task_handlers.py", "'tasks/cancel not supported'"),
    ("client/experimental/task_handlers.py", "'tasks/get not supported'"),
    ("client/experimental/task_handlers.py", "'tasks/list not supported'"),
    ("client/experimental/task_handlers.py", "'tasks/result not supported'"),
    ("client/session.py", "'Elicitation not supported'"),
    ("client/session.py", "'List roots not supported'"),
    ("client/session.py", "'Sampling not supported'"),
    ("client/session_group.py", "'Provided session is not managed or already disconnected.'"),
    ("client/session_group.py", "f'{matching_prompts} already exist in group prompts.'"),
    ("client/session_group.py", "f'{matching_resources} already exist in group resources.'"),
    ("client/session_group.py", "f'{matching_tools} already exist in group tools.'"),
    ("client/streamable_http.py", "'Session terminated'"),
    ("shared/exceptions.py", "message"),
    ("shared/experimental/tasks/capabilities.py", "'Client does not support task-augmented elicitation'"),
    ("shared/experimental/tasks/capabilities.py", "'Client does not support task-augmented sampling'"),
    ("shared/experimental/tasks/helpers.py", "f'Task not found: {task_id}'"),
    ("shared/experimental/tasks/helpers.py", "f\"Cannot cancel task in terminal state '{task.status}'\""),
    ("shared/session.py", "'Connection closed'"),
    ("shared/session.py", "'Invalid request parameters'"),
    ("shared/session.py", "'Request cancelled'"),
    (
        "shared/session.py",
        "f'Timed out while waiting for response to {request.__class__.__name__}. Waited {timeout} seconds.'",
    ),
}


def _sdk_client_side_error_messages() -> set[tuple[str, str]]:
    import mcp

    root = Path(mcp.__file__).parent
    found: set[tuple[str, str]] = set()
    for path in root.rglob("*.py"):
        module = path.relative_to(root).as_posix()
        if module.startswith(("server/", "cli/")):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name != "ErrorData":
                continue
            message = next((keyword.value for keyword in node.keywords if keyword.arg == "message"), None)
            found.add((module, ast.unparse(message) if message is not None else "<positional>"))
    return found


def test_sdk_client_side_error_messages_carry_no_local_detail() -> None:
    """A new or changed SDK error message fails here until someone checks it can't leak local detail."""
    assert _sdk_client_side_error_messages() == _SDK_CLIENT_SIDE_ERROR_MESSAGES
