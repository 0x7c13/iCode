# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Filesystem identities stay raw until search results become display text."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.service.tools.builtins import search


@pytest.mark.parametrize("operation", ["grep", "globbed_grep", "glob"])
async def test_rg_byte_paths_are_escaped_at_the_output_boundary_without_merging_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    try:
        raw_name = os.fsdecode(b"raw_\xff.py")
    except UnicodeError:
        pytest.skip("Filesystem codec does not support surrogateescaped bytes")
    names = [raw_name, r"raw_\udcff.py"]
    records = []
    for name, label in zip(names, ("RAW_ONLY", "LITERAL_ONLY"), strict=True):
        path_data = {"bytes": base64.b64encode(os.fsencode(name)).decode("ascii")}
        records.append(
            {"type": "match", "data": {"path": path_data, "line_number": 1, "lines": {"text": f"NEEDLE {label}\n"}}}
        )
        records.append({"type": "end", "data": {"path": path_data, "binary_offset": None}})

    async def raw_path_output(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        if "--files" in args:
            assert cwd == str(tmp_path)
            return "\0".join([*names, ""]), "", 0
        if cwd is not None:
            # A display-escaped path must never become a filesystem operand.
            assert args[args.index("--") + 1 :] == names
        assert consume is not None
        consume("".join(json.dumps(record) + "\n" for record in records).encode())
        return "", "", 0

    monkeypatch.setattr(search, "_run_rg", raw_path_output)
    result = (
        await search.glob("*.py", path=str(tmp_path))
        if operation == "glob"
        else await search.grep(
            "NEEDLE", path=str(tmp_path), glob="*.py" if operation == "globbed_grep" else None, context_lines=0
        )
    )

    result.encode("utf-8")
    json.dumps({"result": result}, ensure_ascii=False).encode("utf-8")
    assert r"raw_\udcff.py" in result
    if operation == "glob":
        assert "Found 2 file(s)" in result
    else:
        assert "Found 2 match(es)" in result
        assert "RAW_ONLY" in result and "LITERAL_ONLY" in result


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    try:
        root = tmp_path / os.fsdecode(b"root_\xff")
        root.mkdir()
    except OSError, UnicodeError:
        pytest.skip("Filesystem does not support undecodable path bytes")
    return root


@pytest.mark.parametrize("operation", ["grep", "globbed_grep", "glob"])
async def test_undecodable_names_are_distinct_during_search_and_safe_in_output(raw_root: Path, operation: str) -> None:
    raw_name = os.fsdecode(b"raw_\xff.py")
    literal_name = r"raw_\udcff.py"
    try:
        (raw_root / raw_name).write_text("NEEDLE RAW_ONLY " + "x" * 2100 + "\n", encoding="utf-8")
        (raw_root / literal_name).write_text("NEEDLE LITERAL_ONLY\n", encoding="utf-8")
    except OSError, UnicodeError:
        pytest.skip("Filesystem does not support distinct raw-byte and literal-backslash filenames")
    assert os.fsencode(raw_name) != os.fsencode(literal_name)

    files = await search._search_files(str(raw_root), "*.py", True)
    assert isinstance(files, list)
    assert set(files) == {str(raw_root / raw_name), str(raw_root / literal_name)}
    result = (
        await search.glob("*.py", path=str(raw_root))
        if operation == "glob"
        else await search.grep(
            "NEEDLE", path=str(raw_root), glob="*.py" if operation == "globbed_grep" else None, context_lines=0
        )
    )

    result.encode("utf-8")
    json.dumps({"result": result}, ensure_ascii=False).encode("utf-8")
    assert r"root_\udcff" in result and r"raw_\udcff.py" in result
    if operation == "glob":
        assert "Found 2 file(s)" in result
    else:
        assert "Found 2 match(es)" in result
        assert "RAW_ONLY" in result and "LITERAL_ONLY" in result
        assert r"Long lines truncated to 2048 chars: raw_\udcff.py:1" in result


@pytest.mark.parametrize("operation", ["grep", "glob"])
async def test_empty_results_escape_undecodable_root_names(raw_root: Path, operation: str) -> None:
    result = (
        await search.grep("NEEDLE", path=str(raw_root))
        if operation == "grep"
        else await search.glob("*.py", path=str(raw_root))
    )

    assert result.startswith("No ")
    assert r"root_\udcff" in result
    result.encode("utf-8")


@pytest.mark.parametrize("glob_pattern", ["-", "**", "{-,*.py}"])
async def test_a_file_named_dash_is_searched_instead_of_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, glob_pattern: str
) -> None:
    (tmp_path / "-").write_text("NEEDLE DASH_FILE\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("nothing\n", encoding="utf-8")
    original_run = search._run_rg
    content_paths: list[str] = []

    async def recording_run(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        if "--json" in args:
            assert cwd == str(tmp_path)
            content_paths.extend(args[args.index("--") + 1 :])
        return await original_run(args, timeout=timeout, cwd=cwd, consume=consume)

    monkeypatch.setattr(search, "_run_rg", recording_run)
    result = await search.grep("NEEDLE", path=str(tmp_path), glob=glob_pattern, context_lines=0)

    assert os.path.join(".", "-") in content_paths and "-" not in content_paths
    assert "Found 1 match(es)" in result and "-:1" in result
    assert "DASH_FILE" in result and "<stdin>" not in result


async def test_explicit_dash_file_keeps_its_identity_during_listing_and_search(tmp_path: Path) -> None:
    path = tmp_path / "-"
    path.write_text("NEEDLE\n", encoding="utf-8")

    files = await search._search_files(str(path), None, True)
    result = await search.grep("NEEDLE", path=str(path))

    assert files == [str(path)]
    assert "Found 1 match(es)" in result and "<stdin>" not in result


@pytest.mark.parametrize("component", ["!deep[1]{two}", "#deep[1]{two}", "😀" * 12], ids=["bang", "hash", "utf16"])
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("*.py", {"top.py", "src/a.py", "src/deep/b.py"}),
        ("src/*.py", {"src/a.py"}),
        ("/src/*.py", {"src/a.py"}),
        ("**/*.py", {"top.py", "src/a.py", "src/deep/b.py"}),
        ("!src/*.py", {"top.py", "src/deep/b.py", "-"}),
        ("{src/a.py,top.py}", {"src/a.py", "top.py"}),
        ("-", {"-"}),
    ],
)
async def test_windows_long_search_roots_preserve_globs_and_avoid_long_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str, pattern: str, expected: set[str]
) -> None:
    root = tmp_path
    while len(str(root).encode("utf-16-le")) // 2 < 280:
        root /= component
    if component.startswith("😀"):
        assert len(str(root)) < 258  # Python character counts would miss this case.
    for name in ("top.py", "src/a.py", "src/deep/b.py", "-"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NEEDLE\n", encoding="utf-8")
    monkeypatch.setattr(search, "_PLATFORM", replace(search._PLATFORM, os_name="windows"))
    original_run = search._run_rg

    async def windows_cwd_limit(
        args: list[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        consume: Callable[[bytes], bool] | None = None,
    ) -> tuple[str, str, int]:
        if cwd is not None and len(cwd.encode("utf-16-le")) // 2 >= 258:
            raise NotADirectoryError("CreateProcess cannot use a long working directory")
        if "--json" in args:
            assert cwd is None
            assert all(os.path.isabs(name) for name in args[args.index("--") + 1 :])
        return await original_run(args, timeout=timeout, cwd=cwd, consume=consume)

    monkeypatch.setattr(search, "_run_rg", windows_cwd_limit)
    files = await search._search_files(str(root), pattern, True)
    result = await search.grep("NEEDLE", path=str(root), glob=pattern, context_lines=0)

    assert isinstance(files, list)
    assert {Path(name).relative_to(root).as_posix() for name in files} == expected
    assert f"Found {len(expected)} match(es)" in result
    assert all(f"{name}:1" in result for name in expected)
    if pattern == "-":
        assert await search._search_files(str(root / "-"), None, True) == [str(root / "-")]
