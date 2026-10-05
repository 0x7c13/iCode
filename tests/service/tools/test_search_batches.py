# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Platform command budgets, relative search paths, and partial batch failures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.foundation.platform import get_platform
from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
)
from chrys.service.tools.builtins import search
from chrys.service.tools.result_metadata import tool_result_metadata


@pytest.mark.parametrize("os_name", ["windows", "macos", "linux"])
def test_batches_fill_platform_budget_including_fixed_command(monkeypatch: pytest.MonkeyPatch, os_name: str) -> None:
    monkeypatch.setattr(search, "_PLATFORM", replace(get_platform(), os_name=os_name))
    command = [r"C:\Program Files\ripgrep\rg.exe", "--json", "-e", "long pattern 😀 " * 1000, "-E", "shift_jis", "--"]
    paths = [f'dir 文件😀/{index} "quoted"\\name.py' for index in range(12000)]
    limit = 32767 - 512 if os_name == "windows" else 128 * 1024

    def actual_size(argv: list[str]) -> int:
        if os_name == "windows":
            return len(subprocess.list2cmdline(argv).encode("utf-16-le")) // 2 + 1
        strings = sum(len(os.fsencode(argument)) + 1 for argument in argv)
        return strings + ((len(argv) + 1) * 8 if os_name == "linux" else 0)

    batches = list(search._file_batches(paths, command=command))

    assert len(batches) > 1
    assert [path for batch in batches for path in batch] == paths
    for index, batch in enumerate(batches):
        assert actual_size([*command, *batch]) <= limit
        if index + 1 < len(batches):
            assert actual_size([*command, *batch, batches[index + 1][0]]) > limit


@pytest.mark.parametrize("os_name", ["windows", "macos", "linux"])
def test_overlong_fixed_command_or_single_path_is_rejected(monkeypatch: pytest.MonkeyPatch, os_name: str) -> None:
    monkeypatch.setattr(search, "_PLATFORM", replace(get_platform(), os_name=os_name))
    limit = 32767 - 512 if os_name == "windows" else 128 * 1024
    for command, paths in [(["rg", "-e", "x" * limit, "--"], ["a.py"]), (["rg", "--"], ["x" * limit])]:
        with pytest.raises(ValueError, match="command budget"):
            list(search._file_batches(paths, command=command))


def test_windows_batches_allow_a_shim_to_expand_the_executable_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "_PLATFORM", replace(get_platform(), os_name="windows"))
    command = [r"C:\shims\rg.exe", "--json", "-e", "needle", "--"]
    target = "C:\\" + "long package directory\\" * 15 + "rg.exe"
    paths = [f"src/module_{index}.py" for index in range(5000)]

    batches = list(search._file_batches(paths, command=command))

    assert len(batches) > 1
    for batch in batches:
        forwarded = [target, *command[1:], *batch]
        assert len(subprocess.list2cmdline(forwarded).encode("utf-16-le")) // 2 + 1 <= 32767


@pytest.mark.skipif(not get_platform().is_windows, reason="Requires native Windows process creation")
def test_windows_full_batches_round_trip_through_createprocess() -> None:
    # The venv launcher forwards argv to a potentially longer base interpreter path.
    command = [
        getattr(sys, "_base_executable", sys.executable),
        "-c",
        "import json,sys; print(json.dumps(sys.argv[2:]))",
        "pattern with spaces 😀 " * 600,
    ]
    paths = [f'dir 文件😀/{index} "quoted"\\tail\\' for index in range(1600)]
    batches = list(search._file_batches(paths, command=command))
    assert len(batches) > 1
    for batch in batches:
        result = subprocess.run(
            [*command, *batch], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True
        )
        assert json.loads(result.stdout) == batch


async def test_globbed_content_search_uses_relative_paths_from_search_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    for name in ("-option.py", "space 文件.py", "nested/module.py"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("NEEDLE\n", encoding="utf-8")
    monkeypatch.chdir(other)
    content_calls: list[tuple[list[str], str | None]] = []
    original_run = search._run_rg

    async def recording_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        if "--json" in args:
            content_calls.append((args, cwd))
        return await original_run(args, timeout=timeout, cwd=cwd, consume=consume)

    monkeypatch.setattr(search, "_run_rg", recording_run)
    result = await search.grep("NEEDLE", path=str(root), glob="*.py", context_lines=0)

    assert "Found 3 match(es)" in result
    assert "nested/module.py:1" in result
    assert "-option.py:1" in result
    assert "space 文件.py:1" in result
    assert len(content_calls) == 1
    args, cwd = content_calls[0]
    assert cwd == str(root)
    files = args[args.index("--") + 1 :]
    assert all(not os.path.isabs(name) for name in files)
    assert {Path(name).as_posix() for name in files} == {"-option.py", "space 文件.py", "nested/module.py"}


@pytest.mark.parametrize("encoded_path", [{"text": "./nested/module.py"}, {"bytes": "Li9uZXN0ZWQvbW9kdWxlLnB5"}])
def test_relative_json_paths_are_resolved_against_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoded_path: dict[str, str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    output = json.dumps(
        {"type": "match", "data": {"path": encoded_path, "line_number": 1, "lines": {"text": "NEEDLE\n"}}}
    )

    stream = search._GrepStream(str(root), 50)
    stream.feed(output.encode() + b"\n")

    assert stream.result.entries[0].rel_path == "nested/module.py"


@pytest.mark.parametrize("split_batches", [True, False])
@pytest.mark.parametrize("max_results", [1, 50])
async def test_deleted_candidate_preserves_readable_matches_and_reports_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_batches: bool, max_results: int
) -> None:
    for name in ("a.py", "b_deleted.py", "c.py"):
        (tmp_path / name).write_text("NEEDLE\n", encoding="utf-8")
    original_files = search._search_files

    async def delete_after_listing(root: str, pattern: str | None, respect_gitignore: bool) -> list[str]:
        files = await original_files(root, pattern, respect_gitignore)
        assert isinstance(files, list)
        (tmp_path / "b_deleted.py").unlink()
        return sorted(files)

    def split_files(files: list[str], *, command: list[str]) -> Iterator[list[str]]:
        yield files[:2]
        yield files[2:]

    monkeypatch.setattr(search, "_search_files", delete_after_listing)
    if split_batches:
        monkeypatch.setattr(search, "_file_batches", split_files)
    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        result = await search.grep("NEEDLE", path=str(tmp_path), glob="*.py", max_results=max_results, context_lines=0)
    finally:
        tool_result_metadata.reset(token)

    assert f"Found {min(max_results, 2)} match(es)" in result
    if max_results == 50:
        assert "a.py:1" in result and "c.py:1" in result
    else:
        assert "a.py:1" in result or "c.py:1" in result
    if max_results == 1 and not split_batches:
        # rg is stopped at the second match: the result says it is limited, and rg's errors are not reported.
        assert "(limited to 1)" in result and "Error" not in result
        assert TOOL_FAILED_METADATA_KEY not in metadata
        return
    assert "Error: search results may be incomplete" in result
    assert "b_deleted.py" in result
    assert metadata[PROCESS_EXIT_CODE_METADATA_KEY] == 2
    assert metadata[TOOL_FAILED_METADATA_KEY] is True
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "search_incomplete"
