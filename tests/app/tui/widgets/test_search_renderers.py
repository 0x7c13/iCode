# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for search tool renderer regex patterns and format_result logic.

Validates that _HEADER_RE and _HEADER_PARTS_RE correctly extract header,
path (subtitle), and limit suffix from real ripgrep output produced by
the grep/glob tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui.widgets.chat.renderers.search import (
    _HEADER_PARTS_RE,
    _HEADER_RE,
    GlobToolCall,
    GrepToolCall,
)
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY
from chrys.service.tools.builtins.search import glob, grep

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# _HEADER_RE — basic match/no-match
# ---------------------------------------------------------------------------


def test_header_re_matches_grep_header() -> None:
    assert _HEADER_RE.match("Found 3 match(es) in /Users/foo/repo")


def test_header_re_matches_glob_header() -> None:
    assert _HEADER_RE.match("Found 12 file(s) matching '*.py' in /tmp/project")


def test_header_re_matches_with_limit() -> None:
    assert _HEADER_RE.match("Found 50 match(es) in /repo (limited to 50)")


def test_header_re_no_match_on_error() -> None:
    assert not _HEADER_RE.match("Error: ripgrep not found")


def test_header_re_no_match_on_empty() -> None:
    assert not _HEADER_RE.match("No matches found for /foo/ in /bar")


def test_header_re_no_match_on_file_line() -> None:
    assert not _HEADER_RE.match("  src/main.py")


# ---------------------------------------------------------------------------
# _HEADER_PARTS_RE — group extraction
# ---------------------------------------------------------------------------


def test_parts_re_grep_basic() -> None:
    m = _HEADER_PARTS_RE.match("Found 3 match(es) in /Users/foo/repo")
    assert m
    assert m.group(1) == "Found 3 match(es)"
    assert m.group(2) == "/Users/foo/repo"
    assert m.group(3) is None


def test_parts_re_grep_with_limit() -> None:
    m = _HEADER_PARTS_RE.match("Found 50 match(es) in /Users/foo/repo (limited to 50)")
    assert m
    assert m.group(1) == "Found 50 match(es)"
    assert m.group(2) == "/Users/foo/repo"
    assert m.group(3) == " (limited to 50)"


def test_parts_re_glob_basic() -> None:
    m = _HEADER_PARTS_RE.match("Found 5 file(s) matching '*.py' in /tmp/project")
    assert m
    assert m.group(1) == "Found 5 file(s)"
    assert m.group(2) == "/tmp/project"
    assert m.group(3) is None


def test_parts_re_glob_with_limit() -> None:
    m = _HEADER_PARTS_RE.match("Found 100 file(s) matching '*.py' in /tmp/project (limited to 100)")
    assert m
    assert m.group(1) == "Found 100 file(s)"
    assert m.group(2) == "/tmp/project"
    assert m.group(3) == " (limited to 100)"


def test_parts_re_dot_path() -> None:
    """Path '.' should be captured correctly."""
    m = _HEADER_PARTS_RE.match("Found 2 match(es) in .")
    assert m
    assert m.group(1) == "Found 2 match(es)"
    assert m.group(2) == "."
    assert m.group(3) is None


def test_parts_re_windows_path() -> None:
    m = _HEADER_PARTS_RE.match(r"Found 7 file(s) matching '*.py' in C:\Users\dev\project")
    assert m
    assert m.group(1) == "Found 7 file(s)"
    assert m.group(2) == r"C:\Users\dev\project"
    assert m.group(3) is None


def test_parts_re_path_with_spaces() -> None:
    m = _HEADER_PARTS_RE.match("Found 1 match(es) in /home/user/my project/src")
    assert m
    assert m.group(1) == "Found 1 match(es)"
    assert m.group(2) == "/home/user/my project/src"
    assert m.group(3) is None


def test_parts_re_path_with_spaces_and_limit() -> None:
    m = _HEADER_PARTS_RE.match("Found 50 match(es) in /home/user/my project (limited to 50)")
    assert m
    assert m.group(1) == "Found 50 match(es)"
    assert m.group(2) == "/home/user/my project"
    assert m.group(3) == " (limited to 50)"


# ---------------------------------------------------------------------------
# GrepToolCall._format_result
# ---------------------------------------------------------------------------


def _make_grep(args: dict | None = None) -> GrepToolCall:
    return GrepToolCall("c1", "grep", "", args=args)


def test_search_renderers_use_base_tool_card_contract() -> None:
    grep_tool = _make_grep({"pattern": "hello"})
    glob_tool = _make_glob({"pattern": "*.py"})

    assert isinstance(grep_tool, BaseToolCard)
    assert isinstance(glob_tool, BaseToolCard)
    assert grep_tool.status == "running"
    assert glob_tool.result_text == ""


@pytest.mark.asyncio
async def test_search_renderer_records_rejected_completion_approval() -> None:
    class ToolApp(App):
        def compose(self) -> ComposeResult:
            yield GrepToolCall("c1", "grep", args={"pattern": "hello"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(GrepToolCall)

        tool.set_complete("Rejected by user", duration_ms=10, approval="user_rejected")
        await pilot.pause()

        assert tool.approval == "user_rejected"
        assert tool.status == "rejected"
        assert tool.has_class("-rejected")
        assert tool.query_one("#search-panel").border_subtitle == "Rejected"


@pytest.mark.asyncio
async def test_search_renderer_shows_parsed_error_even_when_structured_metadata_says_success() -> None:
    class ToolApp(App):
        def compose(self) -> ComposeResult:
            yield GrepToolCall("c1", "grep", args={"pattern": "hello"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(GrepToolCall)

        tool.set_complete(
            "Error: ripgrep failed",
            duration_ms=10,
            metadata={TOOL_FAILED_METADATA_KEY: False},
        )
        await pilot.pause()

        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-success")
        assert tool.query_one("#search-panel", Static).render().plain == "Error: ripgrep failed"


def test_grep_format_result_basic() -> None:
    result = "Found 1 match(es) in /repo\n\nfoo.py:1\n  >  1 | hello"
    tc = _make_grep({"pattern": "hello"})
    header, body, subtitle, is_error = tc._format_result(result)
    assert header == "Found 1 match(es)"
    assert subtitle == "/repo"
    assert not is_error
    assert "foo.py:1" in body


def test_grep_format_result_with_limit() -> None:
    result = "Found 50 match(es) in /repo (limited to 50)\n\nfoo.py:1\n  >  1 | x"
    tc = _make_grep()
    header, _body, subtitle, is_error = tc._format_result(result)
    assert header == "Found 50 match(es) (limited to 50)"
    assert subtitle == "/repo"
    assert not is_error


def test_grep_format_result_error() -> None:
    tc = _make_grep()
    _header, body, _subtitle, is_error = tc._format_result("Error: ripgrep not found")
    assert is_error
    assert "Error" in body


def test_grep_format_result_no_matches() -> None:
    tc = _make_grep()
    _header, _body, _subtitle, is_error = tc._format_result("No matches found for /foo/ in /bar")
    assert is_error


def test_grep_format_result_truncates_blocks() -> None:
    """Only _GREP_MAX_BLOCKS (1) block should be shown."""
    blocks = "\n\n".join(f"file{i}.py:{i}\n  >  {i} | line" for i in range(5))
    result = f"Found 5 match(es) in /repo\n\n{blocks}"
    tc = _make_grep()
    _header, body, _subtitle, _is_error = tc._format_result(result)
    assert body.count("file") == 1
    assert body.endswith("...")


# ---------------------------------------------------------------------------
# GlobToolCall._format_result
# ---------------------------------------------------------------------------


def _make_glob(args: dict | None = None) -> GlobToolCall:
    return GlobToolCall("c1", "glob", "", args=args)


def test_glob_format_result_basic() -> None:
    result = "Found 3 file(s) matching '*.py' in /repo\n  a.py\n  b.py\n  c.py"
    tc = _make_glob({"pattern": "*.py"})
    header, body, subtitle, is_error = tc._format_result(result)
    assert header == "Found 3 file(s)"
    assert subtitle == "/repo"
    assert not is_error
    assert "a.py" in body


def test_glob_format_result_with_limit() -> None:
    result = "Found 100 file(s) matching '*.py' in /repo (limited to 100)\n  a.py"
    tc = _make_glob()
    header, _body, subtitle, _is_error = tc._format_result(result)
    assert header == "Found 100 file(s) (limited to 100)"
    assert subtitle == "/repo"


def test_glob_format_result_error() -> None:
    tc = _make_glob()
    _header, _body, _subtitle, is_error = tc._format_result("Error: path not found")
    assert is_error


def test_glob_format_result_no_files() -> None:
    tc = _make_glob()
    _header, _body, _subtitle, is_error = tc._format_result("No files matching '*.rs' found in /repo")
    assert is_error


def test_glob_format_result_truncates_files() -> None:
    """Only _GLOB_MAX_FILES (3) files should be shown."""
    files = "\n".join(f"  file{i}.py" for i in range(10))
    result = f"Found 10 file(s) matching '*.py' in /repo\n{files}"
    tc = _make_glob()
    _header, body, _subtitle, _is_error = tc._format_result(result)
    assert body.count("file") == 3
    assert body.endswith("...")


# ---------------------------------------------------------------------------
# Integration: real ripgrep calls → renderer regex extraction
# ---------------------------------------------------------------------------


def _create_test_files(tmp_path: Path) -> None:
    """Create a small test tree for search tests."""
    (tmp_path / "hello.py").write_text("def hello():\n    print('hello world')\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("from hello import hello\nhello()\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "utils.py").write_text("def util():\n    pass\n", encoding="utf-8")
    (sub / "data.txt").write_text("some data here\n", encoding="utf-8")


async def test_grep_real_output_header_regex(tmp_path: Path) -> None:
    """Real ripgrep grep output must match _HEADER_RE and _HEADER_PARTS_RE."""
    _create_test_files(tmp_path)
    result = await grep("hello", path=str(tmp_path))

    lines = result.split("\n")
    assert _HEADER_RE.match(lines[0]), f"_HEADER_RE did not match: {lines[0]!r}"

    m = _HEADER_PARTS_RE.match(lines[0])
    assert m, f"_HEADER_PARTS_RE did not match: {lines[0]!r}"
    assert m.group(1).startswith("Found ")
    assert m.group(2) == str(tmp_path)
    assert m.group(3) is None  # not limited


async def test_grep_real_output_format_result(tmp_path: Path) -> None:
    """GrepToolCall._format_result correctly parses real ripgrep output."""
    _create_test_files(tmp_path)
    result = await grep("hello", path=str(tmp_path))

    tc = _make_grep({"pattern": "hello"})
    header, body, subtitle, is_error = tc._format_result(result)
    assert not is_error
    assert header.startswith("Found ")
    assert "match" in header
    assert subtitle == str(tmp_path)
    assert "hello" in body


async def test_grep_real_output_limited(tmp_path: Path) -> None:
    """When max_results is hit, '(limited to N)' appears and regex captures it."""
    _create_test_files(tmp_path)
    # "hello" appears in both hello.py and main.py — limit to 1
    result = await grep("hello", path=str(tmp_path), max_results=1)

    lines = result.split("\n")
    m = _HEADER_PARTS_RE.match(lines[0])
    assert m, f"_HEADER_PARTS_RE did not match limited output: {lines[0]!r}"
    assert m.group(3) is not None
    assert "limited to 1" in m.group(3)

    tc = _make_grep()
    header, _body, subtitle, is_error = tc._format_result(result)
    assert not is_error
    assert "(limited to 1)" in header
    assert subtitle == str(tmp_path)


async def test_glob_real_output_header_regex(tmp_path: Path) -> None:
    """Real ripgrep glob output must match _HEADER_RE and _HEADER_PARTS_RE."""
    _create_test_files(tmp_path)
    result = await glob("*.py", path=str(tmp_path))

    lines = result.split("\n")
    assert _HEADER_RE.match(lines[0]), f"_HEADER_RE did not match: {lines[0]!r}"

    m = _HEADER_PARTS_RE.match(lines[0])
    assert m, f"_HEADER_PARTS_RE did not match: {lines[0]!r}"
    assert m.group(1).startswith("Found ")
    assert "file" in m.group(1)
    assert m.group(2) == str(tmp_path)
    assert m.group(3) is None


async def test_glob_real_output_format_result(tmp_path: Path) -> None:
    """GlobToolCall._format_result correctly parses real ripgrep output."""
    _create_test_files(tmp_path)
    result = await glob("*.py", path=str(tmp_path))

    tc = _make_glob({"pattern": "*.py"})
    header, body, subtitle, is_error = tc._format_result(result)
    assert not is_error
    assert header.startswith("Found ")
    assert "file" in header
    assert subtitle == str(tmp_path)
    assert "hello.py" in body


async def test_glob_real_output_limited(tmp_path: Path) -> None:
    """When max_results is hit, '(limited to N)' appears and regex captures it."""
    _create_test_files(tmp_path)
    result = await glob("*.py", path=str(tmp_path), max_results=1)

    lines = result.split("\n")
    m = _HEADER_PARTS_RE.match(lines[0])
    assert m, f"_HEADER_PARTS_RE did not match limited output: {lines[0]!r}"
    assert m.group(3) is not None
    assert "limited to 1" in m.group(3)

    tc = _make_glob()
    header, _body, subtitle, is_error = tc._format_result(result)
    assert not is_error
    assert "(limited to 1)" in header
    assert subtitle == str(tmp_path)


async def test_grep_real_no_match_is_error(tmp_path: Path) -> None:
    """No-match output should be treated as error by the renderer."""
    _create_test_files(tmp_path)
    result = await grep("nonexistent_xyz_123", path=str(tmp_path))

    tc = _make_grep()
    _header, _body, _subtitle, is_error = tc._format_result(result)
    assert is_error


async def test_glob_real_no_match_is_error(tmp_path: Path) -> None:
    """No-match output should be treated as error by the renderer."""
    _create_test_files(tmp_path)
    result = await glob("*.rs", path=str(tmp_path))

    tc = _make_glob()
    _header, _body, _subtitle, is_error = tc._format_result(result)
    assert is_error
