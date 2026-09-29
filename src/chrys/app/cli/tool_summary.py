# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One-line tool call summaries and outcomes for headless progress output.

Every string returned here is display text: single-line, control characters
replaced, surrogates neutralized and clamped. Arguments come from the model or
a remote agent, so each value is shape-checked and an unexpected shape falls
back to the label alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from chrys.app.acp.tool_status import tool_result_failed
from chrys.foundation.hosted_tools import HostedToolFamily, HostedToolStatus, normalize_hosted_tool_status
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.tool_kinds import (
    KIND_DOC_CONVERTER,
    KIND_FILESYSTEM_READ,
    KIND_FILESYSTEM_WRITE,
    KIND_MCP,
    KIND_SEARCH,
    KIND_SHELL,
    KIND_SKILL,
    KIND_SLEEP,
    KIND_SUB_AGENT,
    KIND_TODO,
    KIND_WEB_FETCH,
    KIND_WEB_SEARCH,
)
from chrys.foundation.tool_result_metadata import (
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_INTERRUPTED_METADATA_KEY,
    process_exit_code_from_metadata,
    process_timed_out_from_metadata,
    shell_exit_code_from_metadata,
    shell_timed_out_from_metadata,
)

DETAIL_LIMIT: Final = 160
REASON_LIMIT: Final = 120
TODO_TOOL_NAME: Final = "todo_write"

_HOSTED_LABELS: Final[dict[str, str]] = {
    HostedToolFamily.SEARCH: "search",
    HostedToolFamily.FETCH: "fetch",
    HostedToolFamily.MCP: "mcp",
    HostedToolFamily.CODE: "code",
    HostedToolFamily.IMAGE: "image",
    HostedToolFamily.SHELL: "shell",
    HostedToolFamily.TOOL_DISCOVERY: "tools",
    HostedToolFamily.FILE_OPERATION: "file",
}
_SKILL_LABELS: Final[dict[str, tuple[str, str]]] = {
    "load_skill": ("skill", ""),
    "read_skill_resource": ("resource", "resource_name"),
    "run_skill_script": ("script", "script_name"),
}
_HOSTED_SHELL_EXIT_CODE_KEY: Final = "exit_code"
_HOSTED_SHELL_TIMED_OUT_KEY: Final = "timed_out"
"""The hosted shell adapter's result keys, which the shared result classifier does not read."""
_FAILED_HOSTED_STATUSES: Final = frozenset({HostedToolStatus.FAILED, HostedToolStatus.INTERRUPTED})
_TERMINAL_HOSTED_STATUSES: Final = frozenset({HostedToolStatus.COMPLETED, *_FAILED_HOSTED_STATUSES})


@dataclass(frozen=True, slots=True)
class ToolSummary:
    """What a progress line shows for one call: a short verb-like label and its main argument."""

    label: str
    detail: str = ""


def clamp(text: str, limit: int) -> str:
    """Clamp display text to *limit* characters, marking the cut with an ellipsis."""
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def display_line(value: object, limit: int = DETAIL_LIMIT) -> str:
    """First non-empty line of a string value as clamped display text; anything else is empty."""
    if not isinstance(value, str):
        return ""
    first = next((line.strip() for line in value.splitlines() if line.strip()), "")
    return clamp(sanitize_legacy_scalar(surrogate_safe_text(first)), limit)


def is_todo_tool(tool_name: str, tool_kind: str) -> bool:
    """The builtin todo tool, whose list updates print as their own lines."""
    return tool_kind == KIND_TODO and tool_name == TODO_TOOL_NAME


def summarize_tool(tool_name: str, tool_kind: str, args: object, *, hosted_family: str = "") -> ToolSummary:
    """Label and detail for a tool call; unknown tools and malformed arguments show the tool name alone."""
    arguments: Mapping[str, Any] = args if isinstance(args, Mapping) else {}
    name = display_line(tool_name, 60) or "tool"
    if hosted_family:
        return _summarize_hosted(name, hosted_family, arguments)
    if tool_kind == KIND_SHELL:
        command = display_line(arguments.get("command"))
        return ToolSummary("shell", command) if command else ToolSummary(name)
    if tool_kind == KIND_FILESYSTEM_READ:
        return ToolSummary("view" if tool_name == "view_image" else "read", display_line(arguments.get("path")))
    if tool_kind == KIND_FILESYSTEM_WRITE:
        return ToolSummary("edit" if tool_name == "edit_file" else "write", display_line(arguments.get("path")))
    if tool_kind == KIND_SEARCH:
        return _summarize_search(tool_name, arguments)
    if tool_kind == KIND_WEB_SEARCH:
        return ToolSummary("search", display_line(arguments.get("query")))
    if tool_kind == KIND_WEB_FETCH:
        return ToolSummary("fetch", display_line(arguments.get("url")))
    if tool_kind == KIND_SUB_AGENT:
        prompt = display_line(arguments.get("prompt"))
        return ToolSummary("agent", clamp(f"{name}: {prompt}", DETAIL_LIMIT) if prompt else name)
    if tool_kind == KIND_SKILL:
        return _summarize_skill(tool_name, name, arguments)
    if tool_kind == KIND_MCP:
        return ToolSummary("mcp", name)
    if tool_kind == KIND_SLEEP:
        seconds = arguments.get("seconds")
        valid = isinstance(seconds, int) and not isinstance(seconds, bool)
        return ToolSummary("sleep", f"{seconds}s" if valid else "")
    if tool_kind == KIND_DOC_CONVERTER:
        return ToolSummary("convert", display_line(arguments.get("path")))
    if tool_kind == KIND_TODO:
        return ToolSummary("todo")
    return ToolSummary(name)


def _summarize_search(tool_name: str, arguments: Mapping[str, Any]) -> ToolSummary:
    pattern = display_line(arguments.get("pattern"))
    path = display_line(arguments.get("path"))
    if pattern and path and path != ".":
        pattern = clamp(f"{pattern} in {path}", DETAIL_LIMIT)
    return ToolSummary("glob" if tool_name == "glob" else "grep", pattern)


def _summarize_skill(tool_name: str, name: str, arguments: Mapping[str, Any]) -> ToolSummary:
    label, item_key = _SKILL_LABELS.get(tool_name, (name, ""))
    skill = display_line(arguments.get("skill_name"), 80)
    item = display_line(arguments.get(item_key), 80) if item_key else ""
    return ToolSummary(label, f"{skill}/{item}" if skill and item else skill or item)


def _summarize_hosted(name: str, hosted_family: str, arguments: Mapping[str, Any]) -> ToolSummary:
    label = _HOSTED_LABELS.get(hosted_family, name)
    if hosted_family == HostedToolFamily.SHELL:
        commands = arguments.get("commands")
        lines = [line for line in map(display_line, commands) if line] if isinstance(commands, list) else []
        if not lines:
            return ToolSummary(label)
        more = f" (+{len(lines) - 1} more)" if len(lines) > 1 else ""
        return ToolSummary(label, clamp(f"{lines[0]}{more}", DETAIL_LIMIT + len(more)))
    if hosted_family == HostedToolFamily.MCP:
        server = display_line(arguments.get("server"), 60)
        return ToolSummary(label, f"{server}: {name}" if server else name)
    if hosted_family == HostedToolFamily.CODE:
        return ToolSummary(label, display_line(arguments.get("code")))
    for key in ("query", "url", "pattern", "path"):
        detail = display_line(arguments.get(key))
        if detail:
            return ToolSummary(label, detail)
    queries = arguments.get("queries")
    if isinstance(queries, list) and queries:
        return ToolSummary(label, display_line(queries[0]))
    return ToolSummary(label)


def hosted_status_is_terminal(status: str) -> bool:
    """Whether a hosted lifecycle status ends the call: completed, failed or interrupted."""
    return normalize_hosted_tool_status(status) in _TERMINAL_HOSTED_STATUSES


def tool_failed(
    result: str,
    metadata: Mapping[str, Any] | None,
    *,
    tool_name: str,
    tool_kind: str,
    hosted_family: str = "",
    provider_status: str = "",
) -> bool:
    """Whether a tool result is a failure, by the same rules ACP uses plus the hosted lifecycle."""
    if tool_result_failed(result, metadata, tool_kind=tool_kind, tool_name=tool_name):
        return True
    if hosted_family and normalize_hosted_tool_status(provider_status) in _FAILED_HOSTED_STATUSES:
        return True
    return hosted_family == HostedToolFamily.SHELL and _hosted_shell_failed(metadata)


def _hosted_shell_failed(metadata: Mapping[str, Any] | None) -> bool:
    if not metadata:
        return False
    exit_code = metadata.get(_HOSTED_SHELL_EXIT_CODE_KEY)
    if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
        return True
    return metadata.get(_HOSTED_SHELL_TIMED_OUT_KEY) is True


def failure_reason(result: str, metadata: Mapping[str, Any] | None, *, hosted_family: str = "") -> str:
    """A short cause for a failed call: exit status, timeout, interruption, or the error's first line."""
    meta: Mapping[str, Any] = metadata or {}
    if hosted_family == HostedToolFamily.SHELL:
        if meta.get(_HOSTED_SHELL_TIMED_OUT_KEY) is True:
            return "timed out"
        exit_code = meta.get(_HOSTED_SHELL_EXIT_CODE_KEY)
        if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
            return f"exit {exit_code}"
    if shell_timed_out_from_metadata(meta) or process_timed_out_from_metadata(meta):
        return "timed out"
    exit_code = shell_exit_code_from_metadata(meta)
    if exit_code is None:
        exit_code = process_exit_code_from_metadata(meta)
    if exit_code is not None and exit_code != 0:
        return f"exit {exit_code}"
    if meta.get(TOOL_INTERRUPTED_METADATA_KEY) is True:
        return "interrupted"
    reason = display_line(meta.get(TOOL_ERROR_MESSAGE_METADATA_KEY), REASON_LIMIT)
    if reason:
        return reason
    text = result.strip()
    return display_line(text.removeprefix("Error:").strip(), REASON_LIMIT) or "failed"
