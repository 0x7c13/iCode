# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for code search tools."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import Workspace
from chrys.service.tools.builtins.search import (
    _MAX_LINE_DISPLAY_CHARS,
    GrepParseResult,
    LongLine,
    SearchTools,
    _GrepStream,
    _to_posix,
    glob,
    grep,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_test_files(tmp_path: Path) -> None:
    """Create a small test tree for search tests."""
    (tmp_path / "hello.py").write_text("def hello():\n    print('hello world')\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("from hello import hello\nhello()\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "utils.py").write_text("def util():\n    pass\n", encoding="utf-8")
    (sub / "data.txt").write_text("some data here\n", encoding="utf-8")


async def test_grep_finds_match(tmp_path: Path) -> None:
    _create_test_files(tmp_path)
    result = await grep("hello", path=str(tmp_path))
    assert "hello.py:1" in result
    assert "Found" in result


async def test_grep_with_glob(tmp_path: Path) -> None:
    _create_test_files(tmp_path)
    result = await grep("hello", path=str(tmp_path), glob="*.py")
    assert "hello.py" in result
    # .txt files should not be searched
    assert "data.txt" not in result


async def test_grep_no_match(tmp_path: Path) -> None:
    _create_test_files(tmp_path)
    result = await grep("nonexistent_pattern_xyz", path=str(tmp_path))
    assert "No matches found" in result


async def test_grep_invalid_regex(tmp_path: Path) -> None:
    result = await grep("[invalid", path=str(tmp_path))
    assert "Error" in result


async def test_grep_nonexistent_path() -> None:
    result = await grep("test", path="/nonexistent/path")
    assert "not found" in result


async def test_glob_by_extension(tmp_path: Path) -> None:
    _create_test_files(tmp_path)
    result = await glob("*.py", path=str(tmp_path))
    assert "hello.py" in result
    assert "main.py" in result
    assert "utils.py" in result
    assert "data.txt" not in result


async def test_glob_specific_name(tmp_path: Path) -> None:
    _create_test_files(tmp_path)
    result = await glob("data.txt", path=str(tmp_path))
    assert "data.txt" in result
    assert "hello.py" not in result


async def test_glob_no_match(tmp_path: Path) -> None:
    _create_test_files(tmp_path)
    result = await glob("*.rs", path=str(tmp_path))
    assert "No files matching" in result


async def test_glob_nonexistent_path() -> None:
    result = await glob("*.py", path="/nonexistent/path")
    assert "not found" in result


async def test_runtime_bound_search_tools_resolve_default_path_from_runtime_cwd(tmp_path: Path, monkeypatch) -> None:
    """Instance search tools should not resolve '.' against process cwd."""
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    (project / "target.txt").write_text("needle\n", encoding="utf-8")
    (other / "target.txt").write_text("wrong\n", encoding="utf-8")
    monkeypatch.chdir(other)

    runtime = SessionEnvironment.capture(workspace=Workspace.from_cwd(str(project)))
    tools = SearchTools(runtime)

    glob_result = await tools.glob("*.txt")
    assert "target.txt" in glob_result
    assert f"in {project}" in glob_result
    grep_result = await tools.grep("needle")
    assert "Found 1" in grep_result
    assert f"in {project}" in grep_result


async def test_grep_recursive_glob(tmp_path: Path) -> None:
    """The old fnmatch-based implementation broke on **/*.py — verify ripgrep handles it."""
    _create_test_files(tmp_path)
    result = await grep("def", path=str(tmp_path), glob="**/*.py")
    assert "hello.py" in result
    assert "utils.py" in result
    assert "data.txt" not in result


async def test_glob_recursive_glob(tmp_path: Path) -> None:
    """Verify **/*.py finds files in subdirectories."""
    _create_test_files(tmp_path)
    result = await glob("**/*.py", path=str(tmp_path))
    assert "hello.py" in result
    assert "utils.py" in result
    assert "data.txt" not in result


async def test_grep_multi_encoding(tmp_path: Path) -> None:
    """Non-ASCII pattern should find matches in both UTF-8 and GBK files."""
    (tmp_path / "utf8.py").write_bytes("# 测试文件\nvalue = 1\n".encode())
    (tmp_path / "gbk.py").write_bytes("# 测试文件\nvalue = 2\n".encode("gbk"))

    result = await grep("测试", path=str(tmp_path))
    assert "utf8.py" in result
    assert "gbk.py" in result
    assert "Found 2" in result


async def test_grep_ascii_pattern_single_pass(tmp_path: Path) -> None:
    """ASCII-only patterns should not trigger multi-encoding passes."""
    (tmp_path / "test.py").write_text("hello world\n", encoding="utf-8")

    result = await grep("hello", path=str(tmp_path))
    assert "test.py" in result
    assert "Found 1" in result


async def test_grep_skips_hidden_dirs(tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("secret_value = 42\n", encoding="utf-8")
    (tmp_path / "visible.py").write_text("visible_value = 42\n", encoding="utf-8")

    result = await grep("42", path=str(tmp_path))
    assert "visible.py" in result
    assert "secret.py" not in result


# -- grep — long-line truncation -----------------------------------------------


async def test_grep_long_match_line(tmp_path: Path) -> None:
    """A match on a line exceeding _MAX_LINE_DISPLAY_CHARS is truncated with summary."""
    long_line = "MATCH_HERE " + "x" * (_MAX_LINE_DISPLAY_CHARS + 500)
    (tmp_path / "long.py").write_text(f"short line\n{long_line}\nshort line\n", encoding="utf-8")

    result = await grep("MATCH_HERE", path=str(tmp_path))
    assert "Found 1" in result
    # Match content is truncated
    assert "... [truncated]" in result
    # Full line is NOT present
    assert "x" * (_MAX_LINE_DISPLAY_CHARS + 1) not in result
    # Summary footer reports the file, line, and actual length
    assert "long.py:2" in result or "long.py" in result
    assert f"Long lines truncated to {_MAX_LINE_DISPLAY_CHARS}" in result


async def test_grep_long_context_line(tmp_path: Path) -> None:
    """Context lines (non-match) that exceed the limit are also truncated."""
    long_context = "x" * (_MAX_LINE_DISPLAY_CHARS + 500)
    (tmp_path / "ctx.py").write_text(f"{long_context}\nMATCH\nshort\n", encoding="utf-8")

    result = await grep("MATCH", path=str(tmp_path))
    assert "Found 1" in result
    # The context line (line 1) should be truncated
    assert "... [truncated]" in result
    assert "x" * (_MAX_LINE_DISPLAY_CHARS + 1) not in result
    assert f"Long lines truncated to {_MAX_LINE_DISPLAY_CHARS}" in result


async def test_grep_match_line_exactly_at_limit(tmp_path: Path) -> None:
    """A match line with exactly _MAX_LINE_DISPLAY_CHARS is NOT truncated."""
    # "MATCH" is 5 chars, pad the rest
    exact_line = "MATCH" + "a" * (_MAX_LINE_DISPLAY_CHARS - 5)
    assert len(exact_line) == _MAX_LINE_DISPLAY_CHARS
    (tmp_path / "exact.py").write_text(exact_line + "\n", encoding="utf-8")

    result = await grep("MATCH", path=str(tmp_path))
    assert "Found 1" in result
    assert "[truncated]" not in result
    assert "[Long lines" not in result


async def test_grep_multiple_long_lines_across_files(tmp_path: Path) -> None:
    """Long lines in different files are all truncated and listed in the summary."""
    (tmp_path / "a.py").write_text("FIND_ME " + "a" * (_MAX_LINE_DISPLAY_CHARS + 500) + "\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("normal\nFIND_ME " + "b" * (_MAX_LINE_DISPLAY_CHARS + 1000) + "\n", encoding="utf-8")

    result = await grep("FIND_ME", path=str(tmp_path))
    assert "Found 2" in result
    # Both long lines truncated
    assert result.count("... [truncated]") >= 2
    assert f"Long lines truncated to {_MAX_LINE_DISPLAY_CHARS}" in result
    # Both files mentioned in the summary
    assert "a.py" in result
    assert "b.py" in result


# -- _GrepStream unit tests for truncation --------------------------------------


def _parse(stdout: str) -> GrepParseResult:
    """Feed *stdout* as rg prints it, every record ending in a newline."""
    stream = _GrepStream("/root", 50)
    stream.feed(stdout.encode() + b"\n")
    return stream.result


def _make_rg_json_line(entry_type: str, path: str, line_num: int, text: str) -> str:
    """Build a single rg --json output line."""
    return json.dumps(
        {
            "type": entry_type,
            "data": {
                "path": {"text": path},
                "line_number": line_num,
                "lines": {"text": text + "\n"},
            },
        }
    )


def test_parse_grep_jsonl_truncates_long_lines() -> None:
    """_GrepStream truncates lines > _MAX_LINE_DISPLAY_CHARS."""
    long_text = "x" * (_MAX_LINE_DISPLAY_CHARS + 500)
    stdout = _make_rg_json_line("match", "/root/file.py", 1, long_text)

    parsed = _parse(stdout)
    assert parsed.match_count == 1
    assert len(parsed.entries) == 1
    # Entry text is truncated
    assert parsed.entries[0].text.endswith("... [truncated]")
    assert len(parsed.entries[0].text) < len(long_text)
    # long_lines records original length
    assert len(parsed.long_lines) == 1
    assert parsed.long_lines[0] == LongLine("file.py", 1, len(long_text))


def test_parse_grep_jsonl_exact_limit_not_truncated() -> None:
    """Line with exactly _MAX_LINE_DISPLAY_CHARS is NOT truncated."""
    exact_text = "a" * _MAX_LINE_DISPLAY_CHARS
    stdout = _make_rg_json_line("match", "/root/file.py", 1, exact_text)

    parsed = _parse(stdout)
    assert parsed.match_count == 1
    assert parsed.entries[0].text == exact_text
    assert parsed.long_lines == []


class TestToPosix:
    """``_to_posix`` normalises path separators to ``/`` for model-facing output.

    Backed by ``PurePath.as_posix()`` — platform-aware: only converts
    ``\\`` on Windows.  Tests run on all three CI platforms; Windows-only
    backslash conversion uses ``PureWindowsPath`` to verify the underlying
    contract regardless of host OS.
    """

    def test_forward_slash_input_unchanged_everywhere(self) -> None:
        """POSIX-style paths pass through verbatim on Win/Mac/Linux."""
        assert _to_posix("docs/README.md") == "docs/README.md"
        assert _to_posix("a/b/c/d.py") == "a/b/c/d.py"

    def test_os_path_join_output_normalised(self) -> None:
        """Realistic input from ``os.path.join`` (uses ``os.sep``) → POSIX.

        On Windows this exercises the ``\\`` → ``/`` conversion; on POSIX
        ``os.path.join`` already produces forward slashes, so the test is
        a verification of invariance.
        """
        joined = os.path.join("docs", "src", "main.py")
        assert _to_posix(joined) == "docs/src/main.py"

    def test_relative_paths_from_relpath_normalised(self) -> None:
        """Realistic call-site shape: ``_to_posix(os.path.relpath(...))``."""
        # ``os.path.relpath`` uses ``os.sep`` on Windows — same conversion path.
        rel = os.path.relpath(os.path.join("a", "b", "c.txt"), start="a")
        assert _to_posix(rel) == "b/c.txt"

    def test_single_filename_unchanged(self) -> None:
        """No separators to normalise — pass-through."""
        assert _to_posix("README.md") == "README.md"

    def test_purewindows_simulation_converts_backslashes(self) -> None:
        """``PurePath`` on Windows = ``PureWindowsPath`` — verify the contract.

        Runs on all platforms by constructing ``PureWindowsPath`` directly,
        which always treats ``\\`` as a separator regardless of host OS.
        """
        from pathlib import PureWindowsPath

        assert PureWindowsPath("docs\\README.md").as_posix() == "docs/README.md"
        assert PureWindowsPath("a\\b\\c.txt").as_posix() == "a/b/c.txt"

    def test_pureposix_simulation_preserves_backslash_in_filename(self) -> None:
        """``PurePath`` on POSIX = ``PurePosixPath`` — backslash is a regular char.

        Critical correctness property: legitimate POSIX filenames containing
        backslashes must NOT be mangled.  This is the bug raw
        ``str.replace("\\", "/")`` would have caused on POSIX hosts.
        """
        from pathlib import PurePosixPath

        # Backslash is a legal filename character on POSIX
        assert PurePosixPath("weird\\name.txt").as_posix() == "weird\\name.txt"


def test_parse_grep_jsonl_context_line_truncated() -> None:
    """Context lines (non-match) are also truncated."""
    long_ctx = "y" * 3000
    lines = [
        _make_rg_json_line("context", "/root/file.py", 1, long_ctx),
        _make_rg_json_line("match", "/root/file.py", 2, "short match"),
    ]
    stdout = "\n".join(lines)

    parsed = _parse(stdout)
    assert parsed.match_count == 1
    assert len(parsed.entries) == 2
    # Context line truncated
    assert parsed.entries[0].text.endswith("... [truncated]")
    assert parsed.long_lines[0] == LongLine("file.py", 1, 3000)
    # Match line untouched
    assert parsed.entries[1].text == "short match"
