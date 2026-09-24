# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hard cap on the length of every Python file under ``tests/``.

A test module that keeps growing past a couple of thousand lines has almost
always absorbed a second subject, a second scenario family, or copy-pasted
fixtures. The cap forces that to surface as a split into single-subject files
whose names say what they test. It covers ``conftest.py`` and private helper
modules too: a helper that large has stopped being one fixture set. The only
files allowed above the cap are the ones where a split would hurt: one
subject, one cohesive fixture set, an inherently large matrix, and no
duplicated code. Each exemption carries its reason in-line and is pinned so
that a stale entry fails the suite.
"""

from __future__ import annotations

from pathlib import Path

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT, TESTS_ROOT

# Platform-independent test-tree scan: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

MAX_TEST_FILE_LINES = 2000

# Files allowed above the cap. Every entry must satisfy ALL of: the file covers
# exactly ONE subject and is the only file that covers it (for a test module,
# one subject under test; for a conftest or helper module, one fixture family
# serving one subject); what makes it long is inherently large (many
# parametrized/adversarial cases or oracle pins over one contract, or one
# fixture set with many knobs); it is internally cohesive; nothing in it is
# duplicated elsewhere. "It is big" is not a reason — the entry has to say why
# a split would make the tests worse. An entry whose file drops back under the
# cap must be removed.
_OVERSIZED_EXEMPTIONS: dict[Path, str] = {}


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _test_module_line_counts() -> dict[Path, int]:
    return {path.relative_to(REPO_ROOT): _line_count(path) for path in sorted(TESTS_ROOT.rglob("*.py"))}


def test_test_modules_stay_under_line_cap() -> None:
    oversized = [
        f"{path}: {count} lines"
        for path, count in _test_module_line_counts().items()
        if count > MAX_TEST_FILE_LINES and path not in _OVERSIZED_EXEMPTIONS
    ]

    assert oversized == [], (
        f"Python files under tests/ over the {MAX_TEST_FILE_LINES}-line cap:\n  "
        + "\n  ".join(oversized)
        + "\nSplit each test module into single-subject files whose names say what they test (move shared "
        "fixtures to a conftest.py or a helper module), and fold any duplicated cases while you are there. "
        "Split an oversized conftest or helper module the same way, by the subject its helpers serve. Only if "
        "the file is the only one covering its single subject, is internally cohesive, duplicates nothing "
        "elsewhere, and is long because that subject's matrix or fixture set is inherently large, may it be "
        "added to _OVERSIZED_EXEMPTIONS in tests/architecture/test_test_file_size.py with the reason spelled out."
    )


def test_oversized_exemptions_are_live() -> None:
    """An exemption must still name a scanned Python file that the cap would reject."""
    line_counts = _test_module_line_counts()
    problems: list[str] = []
    for path, reason in sorted(_OVERSIZED_EXEMPTIONS.items()):
        if not reason.strip():
            problems.append(f"{path}: the exemption has no reason; say why a split would make the tests worse")
        # Membership in the scan is the test, not "the file exists under tests/": an
        # exemption for something the cap never looks at can never expire.
        if path not in line_counts:
            problems.append(f"{path}: is not one of the files the cap scans; remove the stale entry")
            continue
        if line_counts[path] <= MAX_TEST_FILE_LINES:
            problems.append(f"{path}: is under the cap again; remove the exemption")

    assert problems == [], "\n".join(problems)
