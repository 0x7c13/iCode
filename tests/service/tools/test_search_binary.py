# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Binary filtering for directory searches and explicitly requested files."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from chrys.service.tools.builtins import search


@pytest.mark.parametrize("respect_gitignore", [True, False])
@pytest.mark.parametrize("glob", [None, "*", "*.py", "**/*.py", "!*.txt"])
async def test_directory_grep_skips_binary_content(tmp_path: Path, respect_gitignore: bool, glob: str | None) -> None:
    (tmp_path / "binary.py").write_bytes(b"prefix\0NEEDLE " + b"x" * 3000 + b"\n")
    (tmp_path / "text.py").write_text("NEEDLE\n", encoding="utf-8")

    result = await search._grep_impl(
        "NEEDLE", path=str(tmp_path), glob=glob, respect_gitignore=respect_gitignore, context_lines=0
    )

    assert "Found 1 match(es)" in result
    assert "text.py:1" in result
    assert "binary.py" not in result
    assert "Long lines truncated" not in result


@pytest.mark.parametrize("split_batches", [True, False])
@pytest.mark.parametrize(
    ("pattern", "text_content"), [("NEEDLE", b"NEEDLE\n"), ("NEEDLE|测试", "测试\n".encode("gbk"))]
)
async def test_binary_matches_do_not_consume_globbed_grep_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_batches: bool, pattern: str, text_content: bytes
) -> None:
    binary = tmp_path / "binary.py"
    text = tmp_path / "text.py"
    binary.write_bytes(b"prefix\0NEEDLE\nNEEDLE\n")
    text.write_bytes(text_content)

    async def ordered_files(root: str, pattern: str | None, respect_gitignore: bool) -> list[str]:
        return [str(binary), str(text)]

    def single_file_batches(files: list[str], *, command: list[str]) -> Iterator[list[str]]:
        for name in files:
            yield [name]

    monkeypatch.setattr(search, "_search_files", ordered_files)
    if split_batches:
        monkeypatch.setattr(search, "_file_batches", single_file_batches)
    result = await search.grep(pattern, path=str(tmp_path), glob="*.py", context_lines=0, max_results=1)

    assert "Found 1 match(es)" in result
    assert "text.py:1" in result
    assert "binary.py" not in result


async def test_globbed_grep_with_only_binary_matches_reports_no_matches(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"\0NEEDLE\n")

    result = await search.grep("NEEDLE", path=str(tmp_path), glob="*.py")

    assert result.startswith("No matches found")


@pytest.mark.parametrize("glob", [None, "*.py"])
async def test_explicit_binary_file_remains_searchable(tmp_path: Path, glob: str | None) -> None:
    binary = tmp_path / "binary.py"
    binary.write_bytes(b"prefix\0NEEDLE\n")

    result = await search.grep("NEEDLE", path=str(binary), glob=glob, context_lines=0)

    assert "Found 1 match(es)" in result
    assert "NEEDLE" in result


async def test_glob_still_lists_binary_files(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"prefix\0NEEDLE\n")

    result = await search.glob("*.py", path=str(tmp_path))

    assert "Found 1 file(s)" in result
    assert "binary.py" in result


async def test_dense_globbed_grep_lets_the_event_loop_run_between_output_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "dense.py").write_text("NEEDLE\n" * 10_000, encoding="utf-8")
    loop = asyncio.get_running_loop()
    loop_progressed = asyncio.Event()
    chunks_after_progress = 0

    class ObservedStream(search._GrepStream):
        def feed(self, chunk: bytes) -> bool:
            nonlocal chunks_after_progress
            if loop_progressed.is_set():
                chunks_after_progress += 1
            else:
                # This callback cannot run until parsing gives the event loop a turn.
                loop.call_soon(loop_progressed.set)
            return super().feed(chunk)

    monkeypatch.setattr(search, "_GrepStream", ObservedStream)
    result = await search.grep("NEEDLE", path=str(tmp_path), glob="*.py", context_lines=0)

    assert chunks_after_progress, "Search parsing held the event loop for the whole output"
    assert "Found 50 match(es)" in result
    assert "limited to 50" in result
