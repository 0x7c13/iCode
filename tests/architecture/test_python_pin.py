# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CI tests on, and the release binaries ship, the one CPython version ``.python-version`` pins."""

from __future__ import annotations

import re

from tests.support.paths import REPO_ROOT

_WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _pinned_version() -> str:
    return _read(".python-version").strip()


def test_python_version_pins_a_full_cpython_version() -> None:
    assert re.fullmatch(r"3\.\d+\.\d+", _pinned_version())


def test_workflows_install_the_project_python_from_the_pin() -> None:
    # A bare minor version takes whichever patch release the runner has
    # cached, so the shards of one run can test different interpreters.
    minor = _pinned_version().rsplit(".", 1)[0]
    offenders = [
        f"{workflow.name}:{number}: {line.strip()}"
        for workflow in sorted(_WORKFLOWS.glob("*.yml"))
        for number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), 1)
        if (match := re.match(r"\s*(?:-\s+)?python-version:\s*[\"']?([^\"'\s#]+)", line))
        and (match.group(1) == minor or match.group(1).startswith(f"{minor}."))
    ]
    assert offenders == []


def test_release_builds_take_the_pin_from_one_standalone_release() -> None:
    releases = {
        ".github/workflows/cd.yml": r'^  PYTHON_BUILD_STANDALONE_RELEASE: "(\d{8})"$',
        "scripts/build.sh": r"^PYTHON_BUILD_STANDALONE_RELEASE=(\d{8})$",
        "scripts/build.ps1": r'^\$PythonBuildStandaloneRelease = "(\d{8})"$',
    }
    found = {}
    for relative, pattern in releases.items():
        text = _read(relative)
        assert ".python-version" in text, f"{relative} must take the CPython version from .python-version"
        match = re.search(pattern, text, re.MULTILINE)
        assert match is not None, f"{relative} names no python-build-standalone release"
        found[relative] = match.group(1)
    assert len(set(found.values())) == 1, found
