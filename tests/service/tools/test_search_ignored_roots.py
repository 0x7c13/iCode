# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit ignored roots remain searchable without relaxing broad discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.service.tools.builtins import search


@pytest.fixture
def project(tmp_path: Path, git_repo_factory) -> Path:
    root = git_repo_factory(tmp_path / "repo")
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    for directory in (".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        target = root / directory
        (target / "lib").mkdir(parents=True)
        (target / ".gitignore").write_text("*\n", encoding="utf-8")
        (target / ".ignore").write_text("dot_excluded.py\n", encoding="utf-8")
        (target / ".rgignore").write_text("rg_excluded.py\n", encoding="utf-8")
        for name in ("lib/match.py", "lib/dot_excluded.py", "lib/rg_excluded.py", "lib/.hidden.py"):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("NEEDLE\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("directory", [".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"])
@pytest.mark.parametrize("inside_lib", [False, True])
async def test_explicit_path_into_a_self_ignored_directory(project: Path, directory: str, inside_lib: bool) -> None:
    root = project / directory
    if inside_lib:
        root /= "lib"
    files = await search._search_files(str(root), "*.py", True)
    result = await search.grep("NEEDLE", path=str(root), glob="*.py", context_lines=0)
    plain = await search.grep("NEEDLE", path=str(root), context_lines=0)

    assert files == [str(project / directory / "lib/match.py")]
    for output in (result, plain):
        assert "Found 1 match(es)" in output
        assert "match.py:1" in output
        assert "excluded.py" not in output and "hidden.py" not in output


async def test_explicit_venv_prefix_is_searchable_but_broad_globs_still_skip_it(project: Path) -> None:
    explicit = await search.glob(".venv/**/*.py", path=str(project))
    contents = await search.grep("NEEDLE", glob=".venv/**/*.py", path=str(project), context_lines=0)
    broad = await search.glob("**/*.py", path=str(project))

    assert "Found 1 file(s)" in explicit and ".venv/lib/match.py" in explicit
    assert "Found 1 match(es)" in contents and ".venv/lib/match.py:1" in contents
    assert broad.startswith("No files matching")


async def test_unignored_directory_retains_inherited_and_local_git_rules(project: Path) -> None:
    (project / ".gitignore").write_text("*\n!src/\n__pycache__/\n", encoding="utf-8")
    source = project / "src"
    source.mkdir()
    (source / ".gitignore").write_text("!*.py\nlocal_excluded.py\n", encoding="utf-8")
    for name in ("main.py", "local_excluded.py", "__pycache__/cached.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NEEDLE\n", encoding="utf-8")

    assert not await search._directory_is_gitignored(str(source))
    assert await search._search_files(str(source), "*.py", True) == [str(source / "main.py")]
    assert await search._search_files(str(project), "*.py", True) == [str(source / "main.py")]


async def test_negated_blanket_rule_does_not_disable_git_filtering(project: Path) -> None:
    root = project / ".venv"
    (root / ".gitignore").write_text("*\n!*\nexcluded.py\n", encoding="utf-8")
    (root / "lib/excluded.py").write_text("NEEDLE\n", encoding="utf-8")

    assert not await search._directory_is_gitignored(str(root))
    files = await search._search_files(str(root), "*.py", True)
    assert isinstance(files, list)
    assert str(root / "lib/match.py") in files
    assert str(root / "lib/excluded.py") not in files


@pytest.mark.parametrize(
    ("rule", "directory", "ignore_directory"),
    [
        ("/log/*", "log", "."),
        ("dist/**", "dist", "."),
        ("**/log/*", "log", "."),
        ("**/log/*", "nested/log", "."),
        ("/log/*", "src/log", "src"),
        (r"/log\ files/*", "log files", "."),
    ],
)
async def test_rules_naming_directory_contents_allow_explicit_search(
    project: Path, rule: str, directory: str, ignore_directory: str
) -> None:
    root = project / directory
    root.mkdir(parents=True)
    origin = project / ignore_directory
    relative = root.relative_to(origin).as_posix()
    (origin / ".gitignore").write_text(f"{rule}\n!/{relative}/.keep\n", encoding="utf-8")
    (root / ".keep").write_text("", encoding="utf-8")
    (root / "match.py").write_text("NEEDLE\n", encoding="utf-8")
    (root / "skip.py").write_text("NEEDLE\n", encoding="utf-8")
    (root / ".ignore").write_text("skip.py\n", encoding="utf-8")

    assert await search._search_files(str(root), "*.py", True) == [str(root / "match.py")]
    assert "match.py" in await search.glob(f"{directory}/*.py", path=str(project))
    for pattern in (None, "*.py"):
        result = await search.grep("NEEDLE", path=str(root), glob=pattern, context_lines=0)
        assert "Found 1 match(es)" in result and "match.py:1" in result
        assert "skip.py" not in result
    assert "match.py" not in await search.glob("**/*.py", path=str(project))


async def test_inherited_whitelist_does_not_release_unignored_directories(project: Path) -> None:
    (project / ".gitignore").write_text("*\n!*/\n!*.py\n", encoding="utf-8")
    source = project / "src"
    source.mkdir()
    (source / "main.py").write_text("NEEDLE\n", encoding="utf-8")
    (source / "k.txt").write_text("NEEDLE\n", encoding="utf-8")

    assert not await search._directory_is_gitignored(str(source))
    assert await search._search_files(str(source), "*", True) == [str(source / "main.py")]
