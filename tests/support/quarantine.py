# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Collection-time quarantine marker enforcement."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime

import pytest

QUARANTINE_PROBLEM: pytest.StashKey[str] = pytest.StashKey()


def describe_quarantine_problem(marker: pytest.Mark, current_date: date) -> str | None:
    """Return why *marker* is unusable (invalid metadata or expired), if it is."""
    reason = marker.kwargs.get("reason")
    expires = marker.kwargs.get("expires")
    if not isinstance(reason, str) or not reason.strip():
        return "quarantine requires a non-empty reason="
    if not isinstance(expires, str):
        return "quarantine requires expires='YYYY-MM-DD'"
    try:
        expiry = date.fromisoformat(expires)
    except ValueError:
        return f"invalid quarantine expiry {expires!r}; use YYYY-MM-DD"
    if expiry.isoformat() != expires:
        return f"invalid quarantine expiry {expires!r}; use YYYY-MM-DD"
    if expiry < current_date:
        return f"quarantine expired on {expires}; remove it, extend it with justification, or fix the test"
    return None


def apply_quarantine_markers(items: Iterable[pytest.Item], *, today: date | None = None) -> None:
    """Validate quarantines, fail bad ones at setup, and apply non-strict xfail."""
    current_date = today or datetime.now(UTC).date()
    for item in items:
        marker = item.get_closest_marker("quarantine")
        if marker is None:
            continue
        problem = describe_quarantine_problem(marker, current_date)
        if problem is not None:
            # Raising here would surface as an opaque xdist INTERNALERROR (the
            # hook runs on the worker); stashing defers to a setup-time failure
            # attributed to the test itself, keeping the message readable.
            item.stash[QUARANTINE_PROBLEM] = f"{item.nodeid}: {problem}"
            continue
        reason = marker.kwargs["reason"]
        expires = marker.kwargs["expires"]
        item.add_marker(pytest.mark.xfail(reason=f"Quarantined until {expires}: {reason}", strict=False))


def enforce_quarantine_problem(item: pytest.Item) -> None:
    """Fail *item* at setup when its quarantine metadata was rejected."""
    problem = item.stash.get(QUARANTINE_PROBLEM, None)
    if problem is not None:
        pytest.fail(problem, pytrace=False)
