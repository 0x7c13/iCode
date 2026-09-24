# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Production assertions are forbidden, including type-only and lint-suppressed ones."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture._hygiene_core import _tree
from tests.support.ci import CI_LINUX_ONLY

pytestmark = CI_LINUX_ONLY


def _assert_no_source_asserts(sources: dict[Path, str]) -> None:
    violations = [
        f"{path.as_posix()}:{node.lineno}: src forbids assert; use an explicit check or a typed lifecycle accessor"
        for path, source in sources.items()
        for node in ast.walk(_tree(path, source))
        if isinstance(node, ast.Assert)
    ]
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "assert ready\n",
        "def run():\n    assert ready, 'not ready'\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    assert ready\n",
        "import typing\nif typing.TYPE_CHECKING:\n    assert ready\n",
        "assert ready  # noqa: S101\n",
        "# ruff: noqa: S101\nassert ready\n",
    ],
)
def test_source_assert_guard_rejects_all_assert_statements(source: str) -> None:
    with pytest.raises(AssertionError, match=r"src/example.py:\d+: src forbids assert"):
        _assert_no_source_asserts({Path("src/example.py"): source})


def test_source_assert_guard_allows_examples_and_explicit_checks() -> None:
    _assert_no_source_asserts(
        {
            Path("src/example.py"): '''"""Example: assert ready"""
# assert ready
if not ready:
    raise RuntimeError("not ready")
''',
        }
    )
