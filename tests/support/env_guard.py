# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Environment snapshot, restoration, and failure reporting for pytest."""

from __future__ import annotations

import os
from collections.abc import Mapping

import pytest


def snapshot_environment() -> dict[str, str]:
    """Capture the process environment before the test fixture lifecycle."""
    return dict(os.environ)


def restore_environment_and_describe_changes(before: Mapping[str, str]) -> str | None:
    """Restore *before* and describe exact keys changed by the test, if any."""
    after = dict(os.environ)
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
    if not (added or removed or changed):
        return None
    os.environ.clear()
    os.environ.update(before)
    return f"added={added}, removed={removed}, changed={changed}"


def report_environment_leak(item: pytest.Item, changes: str) -> None:
    """Attach a failed teardown report to the leaking test."""
    message = f"Environment leak after {item.nodeid}: {changes}"
    report = pytest.TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords=dict.fromkeys(item.keywords, 1),
        outcome="failed",
        longrepr=message,
        when="teardown",
    )
    item.ihook.pytest_runtest_logreport(report=report)
