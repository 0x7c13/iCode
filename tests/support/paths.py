# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Stable path helpers for tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"


def fixture_path(*parts: str) -> Path:
    """Return a path under the test fixture tree."""

    return TESTS_ROOT / "fixtures" / Path(*parts)


def deny_path_probes(monkeypatch: pytest.MonkeyPatch, spellings: Iterable[str | os.PathLike[str]]) -> None:
    """Report *spellings* as missing from every ``os.path`` existence probe.

    Tests that feed foreign-platform or fabricated absolute paths (``C:\\notes.md``
    on POSIX, ``/abs/notes.md`` on Windows) must not stat the host's real drive
    root, home, or process cwd for them. ``pathlib.Path.exists``/``is_dir``/
    ``is_file``/``is_symlink`` delegate to these functions, so patching them
    covers both spellings of a probe. Every other path keeps its real answer.
    """
    denied = {os.path.normcase(os.fsdecode(spelling)) for spelling in spellings}
    for name in ("exists", "lexists", "isdir", "isfile", "islink"):
        original = getattr(os.path, name)

        def probe(path: object, *, _original: Callable[[object], bool] = original) -> bool:
            try:
                key = os.path.normcase(os.fsdecode(path))
            except TypeError, ValueError:
                return _original(path)
            return False if key in denied else _original(path)

        monkeypatch.setattr(os.path, name, probe)
