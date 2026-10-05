# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bound ripgrep diagnostics at the tool-result and session-metadata boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
)
from chrys.service.agent_middleware.events.result_persistence import persistable_result_metadata
from chrys.service.tools.builtins import search
from chrys.service.tools.result_metadata import tool_result_metadata


@pytest.fixture
def error_flood() -> str:
    first = "rg: first_unreadable.py: Permission denied " + "路径😀" * 1000 + "FIRST_LINE_END"
    rest = [f"rg: later_{index}.py: Permission denied " + "x" * 160 for index in range(2999)]
    return "\r\n".join([first, *rest, ""])


def _assert_bounded_diagnostics(result: str, metadata: dict[str, object]) -> None:
    persisted = persistable_result_metadata(metadata)
    message = persisted[TOOL_ERROR_MESSAGE_METADATA_KEY]
    assert "first_unreadable.py" in message
    assert "[truncated]" in message
    assert "2999 additional diagnostics omitted" in message
    assert "FIRST_LINE_END" not in result
    assert "later_" not in result
    assert len(message) < 1024
    assert len(result) < 2048
    assert len(json.dumps(persisted, ensure_ascii=False).encode("utf-8")) < 4096
    assert persisted[TOOL_FAILED_METADATA_KEY] is True
    assert persisted[PROCESS_EXIT_CODE_METADATA_KEY] == 2


@pytest.mark.parametrize("glob_pattern", [None, "*.py"])
@pytest.mark.parametrize("has_matches", [True, False])
async def test_content_error_flood_is_bounded_in_results_and_persisted_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_flood: str, glob_pattern: str | None, has_matches: bool
) -> None:
    (tmp_path / "visible.py").write_text("NEEDLE\n" if has_matches else "nothing\n", encoding="utf-8")
    original_run = search._run_rg

    async def failing_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        stdout, stderr, code = await original_run(args, timeout=timeout, cwd=cwd, consume=consume)
        return (stdout, error_flood, 2) if "--json" in args else (stdout, stderr, code)

    monkeypatch.setattr(search, "_run_rg", failing_run)
    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        result = await search.grep("NEEDLE", path=str(tmp_path), glob=glob_pattern, context_lines=0)
    finally:
        tool_result_metadata.reset(token)

    _assert_bounded_diagnostics(result, metadata)
    if has_matches:
        assert "Found 1 match(es)" in result and "visible.py:1" in result
        assert "search results may be incomplete" in result
        assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "search_incomplete"
    else:
        assert result.startswith("Error: ") and "No matches found" not in result
        assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "search_process_failed"


@pytest.mark.parametrize("tool", ["grep", "glob"])
@pytest.mark.parametrize("failed_listing", [1, 2])
async def test_listing_error_flood_is_bounded_before_entering_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_flood: str, tool: str, failed_listing: int
) -> None:
    (tmp_path / "visible.py").write_text("NEEDLE\n", encoding="utf-8")
    original_run = search._run_rg
    listings = 0

    async def failing_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        nonlocal listings
        assert "--files" in args  # Discovery failures must not be mistaken for a complete candidate set.
        listings += 1
        stdout, stderr, code = await original_run(args, timeout=timeout, cwd=cwd, consume=consume)
        return (stdout, error_flood, 2) if listings == failed_listing else (stdout, stderr, code)

    monkeypatch.setattr(search, "_run_rg", failing_run)
    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        result = (
            await search.grep("NEEDLE", path=str(tmp_path), glob="*.py")
            if tool == "grep"
            else await search.glob("*.py", path=str(tmp_path))
        )
    finally:
        tool_result_metadata.reset(token)

    _assert_bounded_diagnostics(result, metadata)
    assert result.startswith("Error: ") and "Found" not in result
    assert listings == failed_listing
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "search_process_failed"


async def test_errors_accumulate_across_batches_and_encodings_without_overwriting_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("NEEDLE\n", encoding="utf-8")
    original_run = search._run_rg
    content_calls = 0

    async def failing_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        nonlocal content_calls
        stdout, stderr, code = await original_run(args, timeout=timeout, cwd=cwd, consume=consume)
        if "--json" in args:
            content_calls += 1
            diagnostics = {1: "first error\nits cause\n", 2: "", 4: "rg: third error\n  \n"}
            if content_calls in diagnostics:
                return stdout, diagnostics[content_calls], 2
        return stdout, stderr, code

    def single_file_batches(files: list[str], *, command: list[str]) -> Iterator[list[str]]:
        for name in files:
            yield [name]

    monkeypatch.setattr(search, "_get_encodings", lambda: ["utf-8", "gbk", "windows-1252"])
    monkeypatch.setattr(search, "_file_batches", single_file_batches)
    monkeypatch.setattr(search, "_run_rg", failing_run)
    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        result = await search.grep("NEEDLE|测试", path=str(tmp_path), glob="*.py", context_lines=0)
    finally:
        tool_result_metadata.reset(token)

    assert content_calls == 9
    assert "Found 3 match(es)" in result
    assert "first error\nits cause [2 additional diagnostics omitted]" in result
    assert "third error" not in result
    assert metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] == (
        "search results may be incomplete — first error\nits cause [2 additional diagnostics omitted]"
    )
    assert metadata[TOOL_FAILED_METADATA_KEY] is True
    assert metadata[PROCESS_EXIT_CODE_METADATA_KEY] == 2


@pytest.mark.parametrize("stderr", ["", " \r\n\t"])
async def test_missing_diagnostics_keep_the_exit_code_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: str
) -> None:
    async def failing_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        return "", stderr, 2

    monkeypatch.setattr(search, "_run_rg", failing_run)

    assert await search.grep("NEEDLE", path=str(tmp_path)) == "Error: rg exited with code 2"


def test_multiline_diagnostic_has_one_shared_size_limit() -> None:
    errors = search._SearchErrors()
    errors.add("rg: first diagnostic\n" + "    detail line\n" * 100 + "rg: second diagnostic\n    other cause\n", 2)

    message = errors.summary()

    assert message.startswith("rg: first diagnostic\n    detail line\n")
    assert "second diagnostic" not in message and "other cause" not in message
    assert message.endswith("... [truncated] [1 additional diagnostics omitted]")
    assert len(message) == 512 + len(" [1 additional diagnostics omitted]")


async def test_repeated_file_error_across_encodings_is_reported_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("visible.py", "deleted.py"):
        (tmp_path / name).write_text("NEEDLE\n", encoding="utf-8")
    original_files = search._search_files
    original_run = search._run_rg
    failed_passes = 0

    async def delete_after_listing(root: str, pattern: str | None, respect_gitignore: bool) -> list[str]:
        files = await original_files(root, pattern, respect_gitignore)
        assert isinstance(files, list)
        (tmp_path / "deleted.py").unlink()
        return files

    async def recording_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        nonlocal failed_passes
        printed = bytearray()

        def tee(chunk: bytes) -> bool:
            printed.extend(chunk)
            return consume is None or consume(chunk)

        stdout, stderr, code = await original_run(args, timeout=timeout, cwd=cwd, consume=tee if consume else None)
        if "--json" in args:
            assert code == 2 and printed
            failed_passes += 1
        return stdout, stderr, code

    monkeypatch.setattr(search, "_search_files", delete_after_listing)
    monkeypatch.setattr(search, "_get_encodings", lambda: ["utf-8", "gbk", "windows-1252"])
    monkeypatch.setattr(search, "_run_rg", recording_run)
    result = await search.grep("NEEDLE|测试", path=str(tmp_path), glob="*.py", context_lines=0)

    assert failed_passes == 3
    assert "Found 1 match(es)" in result
    assert "deleted.py" in result and "may be incomplete" in result
    assert "additional diagnostics" not in result
