# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Validate the test tree package shape and ownership layout."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT, TESTS_ROOT

# Platform-independent test-tree layout scan: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

_TARGET_TOP_LEVEL = {
    "architecture",
    "support",
    "app",
    "orchestration",
    "service",
    "kernel",
    "foundation",
    "integration",
}


def _package_dirs_for(path: Path) -> set[Path]:
    dirs: set[Path] = set()
    current = path if path.is_dir() else path.parent
    while current != TESTS_ROOT.parent:
        dirs.add(current)
        current = current.parent
    return dirs


def test_test_package_paths_have_init_files() -> None:
    paths = list(TESTS_ROOT.rglob("test_*.py")) + [
        path for path in TESTS_ROOT.rglob("*.py") if path.parent == TESTS_ROOT / "support"
    ]
    missing = sorted(
        directory
        for path in paths
        for directory in _package_dirs_for(path)
        if not (directory / "__init__.py").is_file()
    )

    assert missing == []


def test_test_directories_use_allowed_top_level_layout() -> None:
    unexpected = sorted(
        path.relative_to(TESTS_ROOT).as_posix()
        for path in TESTS_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__") and path.name not in _TARGET_TOP_LEVEL
    )

    assert unexpected == []


def test_no_test_modules_live_directly_under_tests_root() -> None:
    direct_tests = sorted(path.name for path in TESTS_ROOT.glob("test_*.py"))

    assert direct_tests == []


def test_pytest_collects_orchestration_engine_build_tests() -> None:
    build_tests = sorted((TESTS_ROOT / "orchestration" / "engine" / "build").glob("test_*.py"))
    assert build_tests != [], (
        "tests/orchestration/engine/build has no test files in this checkout. "
        "The directory mirrors src/chrys/orchestration/engine/build and must be committed; "
        "check that .gitignore keeps build-artifact ignores root-anchored (/build/) "
        "and make sure tests/orchestration/engine/build/*.py is staged."
    )

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]
    assert "norecursedirs" in pytest_options
    assert "build" not in pytest_options["norecursedirs"]

    gitignore_lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/build/" in gitignore_lines
    # An unanchored build/ pattern would swallow the source dirs named build.
    assert "build/" not in gitignore_lines
