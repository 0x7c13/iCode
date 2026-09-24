# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Self-tests for collection-time quarantine enforcement."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.quarantine import apply_quarantine_markers, enforce_quarantine_problem

# Platform-independent test-infra self-checks: the Linux CI job covers them.
pytestmark = CI_LINUX_ONLY


class _Item:
    nodeid = "tests/test_flaky.py::test_flaky"

    def __init__(self, **kwargs: Any) -> None:
        self._marker = SimpleNamespace(kwargs=kwargs)
        self.added: list[Any] = []
        self.stash = pytest.Stash()

    def get_closest_marker(self, name: str) -> Any:
        assert name == "quarantine"
        return self._marker

    def add_marker(self, marker: Any) -> None:
        self.added.append(marker)


def test_unexpired_quarantine_becomes_non_strict_xfail() -> None:
    item = _Item(reason="intermittent provider response", expires="2026-07-17")

    apply_quarantine_markers([item], today=date(2026, 7, 16))  # type: ignore[list-item]

    assert len(item.added) == 1
    assert item.added[0].mark.name == "xfail"
    assert item.added[0].mark.kwargs["strict"] is False
    assert item.added[0].mark.kwargs["reason"] == "Quarantined until 2026-07-17: intermittent provider response"
    enforce_quarantine_problem(item)  # type: ignore[arg-type]  # no stashed problem → no failure


def test_expired_quarantine_guard_goes_red_at_setup() -> None:
    item = _Item(reason="intermittent provider response", expires="2026-07-15")

    apply_quarantine_markers([item], today=date(2026, 7, 16))  # type: ignore[list-item]

    assert item.added == []  # an expired quarantine must not be xfailed away
    with pytest.raises(pytest.fail.Exception, match="quarantine expired on 2026-07-15"):
        enforce_quarantine_problem(item)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "problem"),
    [
        ({"expires": "2099-01-01"}, "non-empty reason="),
        ({"reason": "known flake"}, "expires='YYYY-MM-DD'"),
        ({"reason": "known flake", "expires": "2099-1-1"}, "invalid quarantine expiry"),
    ],
)
def test_invalid_quarantine_metadata_guard_goes_red_at_setup(kwargs: dict[str, str], problem: str) -> None:
    item = _Item(**kwargs)

    apply_quarantine_markers([item], today=date(2026, 7, 16))  # type: ignore[list-item]

    assert item.added == []
    with pytest.raises(pytest.fail.Exception, match=problem):
        enforce_quarantine_problem(item)  # type: ignore[arg-type]
