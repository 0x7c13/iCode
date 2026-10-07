# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fatal ripgrep diagnostics, encoding rejection and system-codepage fallback."""

from __future__ import annotations

import ctypes
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.foundation.platform import get_platform
from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
)
from chrys.service.tools.builtins import search
from chrys.service.tools.result_metadata import tool_result_metadata


def _single_file_batches(files: list[str], *, command: list[str]) -> Iterator[list[str]]:
    for name in sorted(files):
        yield [name]


@pytest.mark.parametrize("glob_pattern", [None, "*.py"])
@pytest.mark.parametrize(
    ("pattern", "context_lines", "cause", "hint"),
    [
        ("foo(?=bar)", 0, "error: look-around", "--pcre2"),
        ("foo(", 0, "error: unclosed group", ""),
        ("测试(?<=x)", 0, "error: look-around", "--pcre2"),
        ("NEEDLE|测试", -1, "error parsing flag -C:", ""),
    ],
)
async def test_fatal_errors_keep_the_cause_and_stop_after_one_content_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    glob_pattern: str | None,
    pattern: str,
    context_lines: int,
    cause: str,
    hint: str,
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("NEEDLE\n", encoding="utf-8")
    original_run = search._run_rg
    failures: list[str] = []

    async def recording_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        printed = bytearray()

        def tee(chunk: bytes) -> bool:
            printed.extend(chunk)
            return consume is None or consume(chunk)

        stdout, stderr, code = await original_run(args, timeout=timeout, cwd=cwd, consume=tee if consume else None)
        if "--json" in args:
            assert code == 2 and not printed
            failures.append(stderr)
        return stdout, stderr, code

    monkeypatch.setattr(search, "_get_encodings", lambda: ["utf-8", "gbk", "windows-1252"])
    monkeypatch.setattr(search, "_file_batches", _single_file_batches)
    monkeypatch.setattr(search, "_run_rg", recording_run)
    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        result = await search.grep(pattern, path=str(tmp_path), glob=glob_pattern, context_lines=context_lines)
    finally:
        tool_result_metadata.reset(token)

    assert len(failures) == 1
    diagnostic = "\n".join(failures[0].strip().splitlines())
    assert cause in result and hint in result
    assert result == "Error: " + diagnostic
    assert metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] == diagnostic
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "search_process_failed"
    assert metadata[TOOL_FAILED_METADATA_KEY] is True
    assert metadata[PROCESS_EXIT_CODE_METADATA_KEY] == 2


@pytest.mark.parametrize("rejected_index", [1, 2])
@pytest.mark.parametrize("legacy_message", [False, True])
async def test_rejected_encoding_is_attempted_once_while_other_encodings_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rejected_index: int, legacy_message: bool
) -> None:
    (tmp_path / "a.py").write_text("NEEDLE\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("NEEDLE\n", encoding="utf-8")
    (tmp_path / "c.py").write_bytes("测试\n".encode("gbk"))
    original_run = search._run_rg
    content_encodings: list[str] = []
    rejected_encoding = "chrys-invalid-encoding"
    encodings = ["utf-8", "gbk"]
    encodings.insert(rejected_index, rejected_encoding)

    async def recording_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        if "--json" in args:
            content_encodings.append(args[args.index("-E") + 1])
        stdout, stderr, code = await original_run(args, timeout=timeout, cwd=cwd, consume=consume)
        if legacy_message and "--json" in args and args[args.index("-E") + 1] == rejected_encoding:
            assert code == 2
            stderr = "unsupported character encoding: chrys-invalid-encoding"
        return stdout, stderr, code

    monkeypatch.setattr(search, "_get_encodings", lambda: encodings)
    monkeypatch.setattr(search, "_file_batches", _single_file_batches)
    monkeypatch.setattr(search, "_run_rg", recording_run)
    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        result = await search.grep("NEEDLE|测试", path=str(tmp_path), glob="*.py", context_lines=0)
    finally:
        tool_result_metadata.reset(token)

    assert Counter(content_encodings) == {"utf-8": 3, "gbk": 3, rejected_encoding: 1}
    assert "Found 3 match(es)" in result
    assert all(f"{name}:1" in result for name in ("a.py", "b.py", "c.py"))
    assert "chrys-invalid-encoding" in result
    assert "additional diagnostics" not in result
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "search_incomplete"
    assert metadata[TOOL_FAILED_METADATA_KEY] is True


async def test_thai_system_codepage_maps_to_a_supported_encoding_and_finds_thai_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "thai.py").write_bytes("สวัสดี\n".encode("cp874"))
    # GetACP is the external Windows boundary; exercise the real codec lookup and mapping.
    monkeypatch.setattr(search, "_PLATFORM", replace(get_platform(), os_name="windows"))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(GetACP=lambda: 874)), raising=False)
    original_run = search._run_rg
    content_encodings: list[str] = []

    async def recording_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        if "--json" in args:
            content_encodings.append(args[args.index("-E") + 1])
        return await original_run(args, timeout=timeout, cwd=cwd, consume=consume)

    monkeypatch.setattr(search, "_run_rg", recording_run)
    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        result = await search.grep("สวัสดี", path=str(tmp_path), glob="*.py", context_lines=0)
    finally:
        tool_result_metadata.reset(token)

    assert content_encodings == ["utf-8", "gbk", "windows-874"]
    assert "Found 1 match(es)" in result and "สวัสดี" in result
    assert "Error:" not in result
    assert metadata.get(TOOL_FAILED_METADATA_KEY) is not True


async def test_utf8_fatal_error_in_a_later_batch_stops_all_remaining_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("NEEDLE\n", encoding="utf-8")
    original_run = search._run_rg
    content_calls: list[tuple[str, str]] = []

    async def failing_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        if "--json" in args:
            encoding = args[args.index("-E") + 1]
            content_calls.append((args[-1], encoding))
            if args[-1] == "b.py":
                return "", "fatal search startup failure", 2
        return await original_run(args, timeout=timeout, cwd=cwd, consume=consume)

    monkeypatch.setattr(search, "_get_encodings", lambda: ["utf-8", "gbk"])
    monkeypatch.setattr(search, "_file_batches", _single_file_batches)
    monkeypatch.setattr(search, "_run_rg", failing_run)
    result = await search.grep("NEEDLE|测试", path=str(tmp_path), glob="*.py", context_lines=0)

    assert content_calls == [("a.py", "utf-8"), ("a.py", "gbk"), ("b.py", "utf-8")]
    assert "Found 1 match(es)" in result and "a.py:1" in result
    assert "fatal search startup failure" in result
