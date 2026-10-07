# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A rename that another program briefly blocks, as Windows refuses it, for atomic-write tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import ModuleType

import pytest

from chrys.foundation.platform import files


@dataclass
class LockedRename:
    """What the blocked rename saw: each attempt's target and each retry sleep."""

    attempts: list[str] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)


def briefly_locked_rename(monkeypatch: pytest.MonkeyPatch, *, failures: int, windows: bool = True) -> LockedRename:
    """Make ``os.replace`` raise ``PermissionError`` *failures* times, then rename.

    The retry helper sees the platform as Windows when *windows* is true, and
    records its sleeps instead of sleeping.
    """
    real_replace = os.replace
    record = LockedRename()

    def replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        record.attempts.append(os.fspath(target))
        if len(record.attempts) <= failures:
            raise PermissionError("the target is open in another process")
        real_replace(source, target)

    fake_time = ModuleType("time")
    fake_time.sleep = record.sleeps.append  # type: ignore[attr-defined]
    monkeypatch.setattr(files, "_is_windows", lambda: windows)
    monkeypatch.setattr(files.os, "replace", replace)
    monkeypatch.setattr(files, "time", fake_time)
    return record
