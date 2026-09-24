# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Src rule: the workflow SDK and worker host stay stdlib-only and on Python 3.9 syntax and APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT
from tests.support.workflow_floor import violations

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

SDK_DIR = Path("src/chrys/service/workflows/sdk")
HOST = Path("src/chrys/service/workflows/worker_host.py")


def worker_files() -> list[Path]:
    return [*sorted((REPO_ROOT / SDK_DIR).glob("*.py")), REPO_ROOT / HOST]


def test_worker_side_sources_stay_on_the_floor() -> None:
    problems: list[str] = []
    for path in worker_files():
        problems.extend(violations(path.read_text(encoding="utf-8"), path=str(path.relative_to(REPO_ROOT))))
    assert problems == [], "\n".join(problems)


def test_worker_files_are_the_expected_set() -> None:
    names = {path.name for path in worker_files()}
    assert {"__init__.py", "_builder.py", "_values.py", "worker_host.py"} <= names


@pytest.mark.parametrize(
    "source",
    [
        "import tomllib\n",
        "import typing_extensions\n",
        "from typing import Self\n",
        "import typing\nx: typing.TypeAlias = int\n",
        "from asyncio import TaskGroup\n",
        "import asyncio\nasync def f():\n    async with asyncio.timeout(1):\n        pass\n",
        "from enum import StrEnum\n",
        "from dataclasses import dataclass\n@dataclass(frozen=True, slots=True)\nclass A:\n    x: int\n",
        "import dataclasses\n@dataclasses.dataclass(slots=True)\nclass A:\n    x: int\n",
        "import chrys.foundation.errors\n",
        "from chrys.service.workflows import values\n",
        "from chrys.workflows._builder import Workflow\n",
        "import requests\n",
        "from yaml import safe_load\n",
        "match x:\n    case 1:\n        pass\n",
        "try:\n    pass\nexcept* ValueError:\n    pass\n",
    ],
    ids=[
        "tomllib",
        "typing_extensions",
        "typing.Self",
        "typing.TypeAlias",
        "asyncio.TaskGroup",
        "asyncio.timeout",
        "StrEnum",
        "dataclass-slots",
        "dataclasses.dataclass-slots",
        "chrys-import",
        "chrys-from-import",
        "chrys-sdk-internals",
        "third-party-import",
        "third-party-from-import",
        "match",
        "except-star",
    ],
)
def test_guard_rejects_counterexamples(source: str) -> None:
    assert violations(source) != []


def test_guard_accepts_floor_code() -> None:
    source = (
        "from __future__ import annotations\n"
        "import asyncio\nfrom dataclasses import dataclass\nfrom typing import Dict, Optional\n"
        "@dataclass(frozen=True)\nclass A:\n    x: Optional[int] = None\n"
        "def f(d: Dict[str, int]) -> int:\n    return sum(d.values())\n"
        "async def g():\n    await asyncio.sleep(0)\n"
        "import chrys.workflows\n"
    )
    assert violations(source) == []
