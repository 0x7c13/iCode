# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One-line tool summaries and failure reasons for headless progress."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.app.cli.tool_summary import (
    DETAIL_LIMIT,
    ToolSummary,
    display_line,
    failure_reason,
    hosted_status_is_terminal,
    is_todo_tool,
    summarize_tool,
    tool_failed,
)
from chrys.foundation.hosted_tools import HostedToolFamily
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


@pytest.mark.parametrize(
    ("tool_name", "tool_kind", "args", "expected"),
    [
        (
            "shell",
            KIND_SHELL,
            {"command": "\n  uv run pytest -q\nsecond line"},
            ToolSummary("shell", "uv run pytest -q"),
        ),
        ("shell", KIND_SHELL, {"command": ["not", "a", "string"]}, ToolSummary("shell")),
        ("read_file", KIND_FILESYSTEM_READ, {"path": "src/app.py"}, ToolSummary("read", "src/app.py")),
        ("view_image", KIND_FILESYSTEM_READ, {"path": "shot.png"}, ToolSummary("view", "shot.png")),
        ("write_file", KIND_FILESYSTEM_WRITE, {"path": "new.py"}, ToolSummary("write", "new.py")),
        ("edit_file", KIND_FILESYSTEM_WRITE, {"path": "old.py"}, ToolSummary("edit", "old.py")),
        ("grep", KIND_SEARCH, {"pattern": "TODO", "path": "src"}, ToolSummary("grep", "TODO in src")),
        ("grep", KIND_SEARCH, {"pattern": "TODO", "path": "."}, ToolSummary("grep", "TODO")),
        ("glob", KIND_SEARCH, {"pattern": "**/*.py"}, ToolSummary("glob", "**/*.py")),
        ("web_search", KIND_WEB_SEARCH, {"query": "rich console"}, ToolSummary("search", "rich console")),
        ("web_fetch", KIND_WEB_FETCH, {"url": "https://example.com"}, ToolSummary("fetch", "https://example.com")),
        (
            "Explore",
            KIND_SUB_AGENT,
            {"prompt": "Find the parser\nmore"},
            ToolSummary("agent", "Explore: Find the parser"),
        ),
        ("Explore", KIND_SUB_AGENT, {}, ToolSummary("agent", "Explore")),
        ("load_skill", KIND_SKILL, {"skill_name": "pdf"}, ToolSummary("skill", "pdf")),
        (
            "read_skill_resource",
            KIND_SKILL,
            {"skill_name": "pdf", "resource_name": "forms.md"},
            ToolSummary("resource", "pdf/forms.md"),
        ),
        (
            "run_skill_script",
            KIND_SKILL,
            {"skill_name": "pdf", "script_name": "fill.py"},
            ToolSummary("script", "pdf/fill.py"),
        ),
        ("search_issues", KIND_MCP, {"q": "bug"}, ToolSummary("mcp", "search_issues")),
        ("sleep", KIND_SLEEP, {"seconds": 5}, ToolSummary("sleep", "5s")),
        ("sleep", KIND_SLEEP, {"seconds": True}, ToolSummary("sleep")),
        ("convert_document", KIND_DOC_CONVERTER, {"path": "a.pdf"}, ToolSummary("convert", "a.pdf")),
        ("todo_write", KIND_TODO, {"todos": []}, ToolSummary("todo")),
        ("custom_tool", "", {"anything": 1}, ToolSummary("custom_tool")),
        ("read_file", KIND_FILESYSTEM_READ, ["not", "a", "mapping"], ToolSummary("read")),
    ],
)
def test_summaries_name_the_main_argument(tool_name: str, tool_kind: str, args: Any, expected: ToolSummary) -> None:
    assert summarize_tool(tool_name, tool_kind, args) == expected


@pytest.mark.parametrize(
    ("family", "args", "expected"),
    [
        (HostedToolFamily.SHELL, {"commands": ["ls -la", "pwd", "whoami"]}, ToolSummary("shell", "ls -la (+2 more)")),
        (HostedToolFamily.SHELL, {"commands": ["ls"]}, ToolSummary("shell", "ls")),
        (HostedToolFamily.SHELL, {"commands": "ls"}, ToolSummary("shell")),
        (HostedToolFamily.MCP, {"server": "github"}, ToolSummary("mcp", "github: remote_tool")),
        (HostedToolFamily.CODE, {"code": "print(1)\nprint(2)"}, ToolSummary("code", "print(1)")),
        (HostedToolFamily.SEARCH, {"queries": ["first", "second"]}, ToolSummary("search", "first")),
        (HostedToolFamily.FETCH, {"url": "https://example.com"}, ToolSummary("fetch", "https://example.com")),
        (HostedToolFamily.IMAGE, {}, ToolSummary("image")),
        ("unknown_family", {}, ToolSummary("remote_tool")),
    ],
)
def test_hosted_summaries_read_the_adapter_arguments(family: str, args: dict[str, Any], expected: ToolSummary) -> None:
    assert summarize_tool("remote_tool", "", args, hosted_family=family) == expected


def test_display_text_is_one_sanitized_clamped_line() -> None:
    assert display_line("\x1b[2Jred\x07 text\nnext") == "�[2Jred� text"
    # A lone surrogate from a filesystem path becomes printable text the console can encode.
    assert display_line("bad \udcff byte").encode("utf-8").startswith(b"bad ")
    long = display_line("x" * 500)
    assert len(long) == DETAIL_LIMIT
    assert long.endswith("…")
    assert display_line(42) == ""
    assert display_line("   \n\n  ") == ""


def test_only_the_builtin_todo_tool_is_the_todo_tool() -> None:
    assert is_todo_tool("todo_write", KIND_TODO)
    # A remote tool that happens to share the name is a normal call.
    assert not is_todo_tool("todo_write", KIND_MCP)
    assert not is_todo_tool("other", KIND_TODO)


@pytest.mark.parametrize(
    ("status", "terminal"),
    [("completed", True), ("failed", True), ("interrupted", True), ("in_progress", False), ("searching", False)],
)
def test_hosted_terminal_statuses(status: str, terminal: bool) -> None:
    assert hosted_status_is_terminal(status) is terminal


@pytest.mark.parametrize(
    ("result", "metadata", "kwargs", "failed"),
    [
        ("ok", {}, {"tool_name": "read_file", "tool_kind": KIND_FILESYSTEM_READ}, False),
        ("Error: nope", {"failed": True}, {"tool_name": "read_file", "tool_kind": KIND_FILESYSTEM_READ}, True),
        ("output\n[exit_code: 2]", {}, {"tool_name": "shell", "tool_kind": KIND_SHELL}, True),
        ("", {}, {"tool_name": "x", "tool_kind": "", "hosted_family": "search", "provider_status": "failed"}, True),
        ("", {}, {"tool_name": "x", "tool_kind": "", "hosted_family": "search", "provider_status": "completed"}, False),
        ("", {"exit_code": 1}, {"tool_name": "x", "tool_kind": "", "hosted_family": "shell"}, True),
        ("", {"timed_out": True}, {"tool_name": "x", "tool_kind": "", "hosted_family": "shell"}, True),
        (
            "",
            {"exit_code": 0, "timed_out": False},
            {"tool_name": "x", "tool_kind": "", "hosted_family": "shell"},
            False,
        ),
        # The hosted shell keys mean nothing on another family.
        ("", {"exit_code": 1}, {"tool_name": "x", "tool_kind": "", "hosted_family": "code"}, False),
    ],
)
def test_failure_follows_the_acp_rules_plus_the_hosted_lifecycle(
    result: str, metadata: dict[str, Any], kwargs: dict[str, str], failed: bool
) -> None:
    assert tool_failed(result, metadata, **kwargs) is failed


@pytest.mark.parametrize(
    ("result", "metadata", "family", "reason"),
    [
        ("", {"exit_code": 3}, "shell", "exit 3"),
        ("", {"timed_out": True, "exit_code": 3}, "shell", "timed out"),
        ("", {"shell_timed_out": True}, "", "timed out"),
        ("", {"process_timed_out": True}, "", "timed out"),
        ("", {"shell_exit_code": 2}, "", "exit 2"),
        ("", {"process_exit_code": 7}, "", "exit 7"),
        ("", {"interrupted": True}, "", "interrupted"),
        ("Error: raw", {"tool_error_message": "File not found: a.py\ndetail"}, "", "File not found: a.py"),
        ("Error: Permission denied\nmore", {}, "", "Permission denied"),
        ("", {}, "", "failed"),
    ],
)
def test_failure_reasons_are_short(result: str, metadata: dict[str, Any], family: str, reason: str) -> None:
    assert failure_reason(result, metadata, hosted_family=family) == reason
