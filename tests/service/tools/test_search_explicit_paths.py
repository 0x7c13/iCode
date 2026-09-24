# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit search targets override discovery exclusions without widening globs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chrys.service.tools.builtins import search


@pytest.fixture
def project(tmp_path: Path, git_repo_factory) -> Path:
    root = git_repo_factory(tmp_path / "repo")
    (root / ".gitignore").write_text("ignored/\n.env\n", encoding="utf-8")
    for name in (
        ".env",
        "nested/.env",
        ".env.example",
        ".github/workflows/ci.yml",
        "ignored/file.py",
        "ignored/deep/file.py",
        "ignored/skip.py",
        "visible.py",
        ".hidden.py",
        ".git/probe.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NEEDLE\n", encoding="utf-8")
    (root / "ignored/.gitignore").write_text("skip.py\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (".env", {".env", "nested/.env"}),
        (".env*", {".env.example"}),
        (".github/workflows/*.yml", {".github/workflows/ci.yml"}),
        ("ignored/file.py", {"ignored/file.py"}),
        ("/ignored/file.py", {"ignored/file.py"}),
        ("ignored/*.py", {"ignored/file.py", "ignored/skip.py"}),
        ("ignored/**/*.py", {"ignored/file.py", "ignored/deep/file.py", "ignored/skip.py"}),
        ("ignored/skip.py", {"ignored/skip.py"}),
        ("*.py", {"visible.py"}),
        ("!ignored/*.py", {"README.md", "visible.py"}),
    ],
)
async def test_explicit_targets_and_broad_discovery_have_distinct_scope(
    project: Path, pattern: str, expected: set[str]
) -> None:
    files = await search._search_files(str(project), pattern, True)
    assert isinstance(files, list)
    assert {Path(name).relative_to(project).as_posix() for name in files} == expected

    listed = await search.glob(pattern, path=str(project))
    assert f"Found {len(expected)} file(s)" in listed
    assert all(f"  {name}" in listed for name in expected)

    result = await search.grep("NEEDLE", path=str(project), glob=pattern, context_lines=0)
    matching = expected - {"README.md"}
    assert f"Found {len(matching)} match(es)" in result
    assert all(f"{name}:1" in result for name in matching)


@pytest.mark.parametrize("pattern", [".git/probe.py", ".git/*.py", "**/.git/*.py"])
async def test_explicit_patterns_keep_git_metadata_excluded(project: Path, pattern: str) -> None:
    assert await search._search_files(str(project), pattern, False) == []


@pytest.mark.parametrize("glob_pattern", [None, "*.py"])
async def test_explicit_path_inside_ignored_directory_is_searchable(project: Path, glob_pattern: str | None) -> None:
    result = await search.grep("NEEDLE", path=str(project / "ignored/file.py"), glob=glob_pattern)
    assert "Found 1 match(es)" in result
    files = await search._search_files(str(project / "ignored"), glob_pattern, True)
    assert isinstance(files, list)
    assert {Path(name).relative_to(project).as_posix() for name in files} == {
        "ignored/file.py",
        "ignored/deep/file.py",
        "ignored/skip.py",
    }


@pytest.mark.parametrize("folder", ["!named[1]", "#named{two}"])
async def test_escaped_directory_names_keep_literal_identity(tmp_path: Path, folder: str) -> None:
    target = tmp_path / folder / "file.py"
    target.parent.mkdir()
    target.write_text("NEEDLE\n", encoding="utf-8")
    escaped = "".join("\\" + char if char in "!#[]{}" else char for char in folder)
    result = await search.grep("NEEDLE", path=str(tmp_path), glob=f"{escaped}/*.py")
    assert "Found 1 match(es)" in result
    assert f"{folder}/file.py:1" in result


async def test_hidden_directory_scope_survives_windows_long_cwd_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    while len(str(root).encode("utf-16-le")) // 2 < 280:
        root /= "deep[1]"
    target = root / ".github/workflows/ci.yml"
    target.parent.mkdir(parents=True)
    target.write_text("NEEDLE\n", encoding="utf-8")
    monkeypatch.setattr(search, "_PLATFORM", replace(search._PLATFORM, os_name="windows"))

    result = await search.grep("NEEDLE", path=str(root), glob=".github/workflows/*.yml")

    assert "Found 1 match(es)" in result
    assert ".github/workflows/ci.yml:1" in result


@pytest.mark.parametrize(
    "pattern",
    ["{.envrc,x}", "[.]envrc", r"[\.]envrc", r"\.envrc", "**/{.envrc,x}", "**/[a.]envrc", "{x,{.envrc,y}}"],
)
async def test_explicit_dotfile_glob_syntax(project: Path, pattern: str) -> None:
    (project / ".envrc").write_text("NEEDLE\n", encoding="utf-8")
    files = await search._search_files(str(project), pattern, True)
    result = await search.grep("NEEDLE", path=str(project), glob=pattern, context_lines=0)

    assert files == [str(project / ".envrc")]
    assert "Found 1 match(es)" in result and ".envrc:1" in result


async def test_mixed_brace_glob_finds_dotfiles_inside_a_named_directory(project: Path) -> None:
    (project / "src").mkdir()
    for name in (".envrc", "main.py"):
        (project / "src" / name).write_text("NEEDLE\n", encoding="utf-8")
    files = await search._search_files(str(project), "src/{.envrc,main.py}", True)

    assert isinstance(files, list)
    assert {Path(name).name for name in files} == {".envrc", "main.py"}


@pytest.mark.parametrize("pattern", ["*.py", "*.{py,txt}", "[!x]*.py", "[!.]*.py"])
async def test_extension_dots_do_not_enable_hidden_discovery(project: Path, pattern: str) -> None:
    files = await search._search_files(str(project), pattern, True)

    assert files == [str(project / "visible.py")]


@pytest.mark.parametrize("pattern", [".GIT/**", ".Git/config", ".git./**", "SRC/*.py", "src./*.py"])
async def test_directory_prefix_must_match_the_actual_entry_name(project: Path, pattern: str) -> None:
    (project / "src").mkdir()
    (project / "src/a.py").write_text("NEEDLE\n", encoding="utf-8")

    assert await search._search_files(str(project), pattern, True) == []
    result = await search.grep("NEEDLE", path=str(project), glob=pattern)
    assert result.startswith("No matches found")
